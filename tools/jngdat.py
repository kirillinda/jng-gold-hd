#!/usr/bin/env python3
"""
jngdat — reader/writer for Jets'n'Guns Gold .dat archives.

Format (reverse-engineered from the jng_gold binary):
  Header (16 bytes):
    u32 magic          (per-file, ignored)
    u32 count          number of index entries
    u32 index_offset   absolute file offset of the index
    u32 adler32        adler32 of the first 12 header bytes
  Payload region [16, index_offset): concatenated LZO1X-compressed blocks.
  Index at index_offset, `count` records:
    u16 namelen                    (includes trailing NUL)
    u8  name[namelen]              XOR-obfuscated (see xorbuf)
    u32 uncompressed_size          XOR-obfuscated
    u32 data_offset                XOR-obfuscated (absolute)
  A record's compressed size is next_offset - this_offset (last uses
  index_offset - this_offset). Names use '\\' separators (Windows origin).

Obfuscation (xorbuf): per-entry key `seed`, starting at index_offset and
evolving `seed = (seed * 0x17BC3) & 0xFFFFFFFF` after each entry. Keystream
word i = (seed + i*0x732C2E17). The original routine only obfuscates the
first `len>>2` "units" of each buffer (a quirk of the shipped code); this is
replicated exactly so round-trips are byte-identical.
"""
import struct, zlib, os

KS_STEP = 0x732C2E17
KEY_MUL = 0x17BC3
MASK = 0xFFFFFFFF

# ---- LZO1X ------------------------------------------------------------------
# Two interchangeable back-ends, tried in order:
#   1. lzallright — a pip wheel bundling miniLZO (LZO1X). It ships prebuilt
#      wheels for Windows/macOS/Linux, so no system library or C toolchain is
#      needed. This is the portable default and what the Windows build uses.
#   2. ctypes -> system liblzo2 — the original zero-dependency Linux path, kept
#      as a fallback for machines that have liblzo2 but not the wheel.
# Both speak the same LZO1X bitstream the game's own decompressor reads.
_LZO_BACKEND = None

try:
    import lzallright as _lzal          # cross-platform prebuilt wheel
    _lzc = _lzal.LZOCompressor()
    _LZO_BACKEND = "lzallright"

    def lzo_decompress(src: bytes, out_len: int) -> bytes:
        out = _lzal.LZOCompressor.decompress(bytes(src), output_size_hint=out_len)
        if len(out) != out_len:
            raise ValueError(f"lzo decompress len {len(out)} != {out_len}")
        return bytes(out)

    def lzo_compress(src: bytes) -> bytes:
        return bytes(_lzc.compress(bytes(src)))

except ImportError:
    import ctypes, ctypes.util
    _lib = ctypes.util.find_library("lzo2") or "liblzo2.so.2"
    _lzo = ctypes.CDLL(_lib)
    _LZO_BACKEND = f"liblzo2 ({_lib})"
    _lzo.lzo1x_decompress_safe.restype = ctypes.c_int
    _lzo.lzo1x_decompress_safe.argtypes = [
        ctypes.c_char_p, ctypes.c_size_t,
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]
    # 1x-1 compressor (present in full liblzo2); wrkmem size for LZO1X-1
    _LZO1X_1_MEM = 16384 * ctypes.sizeof(ctypes.c_void_p)
    _has_compress = hasattr(_lzo, "lzo1x_1_compress")
    if _has_compress:
        _lzo.lzo1x_1_compress.restype = ctypes.c_int
        _lzo.lzo1x_1_compress.argtypes = [
            ctypes.c_char_p, ctypes.c_size_t,
            ctypes.c_char_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p]

    def lzo_decompress(src: bytes, out_len: int) -> bytes:
        dst = ctypes.create_string_buffer(out_len)
        dl = ctypes.c_size_t(out_len)
        r = _lzo.lzo1x_decompress_safe(src, len(src), dst, ctypes.byref(dl), None)
        if r != 0:
            raise ValueError(f"lzo1x_decompress_safe rc={r}")
        return dst.raw[:dl.value]

    def lzo_compress(src: bytes) -> bytes:
        if not _has_compress:
            raise RuntimeError("liblzo2 lacks lzo1x_1_compress")
        dst = ctypes.create_string_buffer(len(src) + len(src)//16 + 64 + 3)
        dl = ctypes.c_size_t(len(dst))
        wrk = ctypes.create_string_buffer(_LZO1X_1_MEM)
        r = _lzo.lzo1x_1_compress(src, len(src), dst, ctypes.byref(dl), wrk)
        if r != 0:
            raise ValueError(f"lzo1x_1_compress rc={r}")
        return dst.raw[:dl.value]

# ---- obfuscation -----------------------------------------------------------
def xorbuf(buf: bytearray, seed: int):
    L = len(buf); wc = L >> 2; seed &= MASK
    if wc > 3:
        edi = wc; j = 0
        while True:
            edi -= 4
            for k in range(4):
                ks = (seed + (j + k) * KS_STEP) & MASK
                o = (j + k) * 4
                buf[o:o+4] = (int.from_bytes(buf[o:o+4], 'little') ^ ks).to_bytes(4, 'little')
            j += 4
            if edi <= 3:
                break
        for k in range(wc - j):
            buf[j*4 + k] ^= ((seed + (j + k) * KS_STEP) & MASK) & 0xFF
    else:
        for k in range(wc):
            buf[k] ^= ((seed + k * KS_STEP) & MASK) & 0xFF
    return buf

# ---- archive ---------------------------------------------------------------
BLOCK = 0x8000  # uncompressed block size

class Entry:
    __slots__ = ("name", "uncomp", "bt_offset")
    def __init__(self, name, uncomp, bt_offset):
        self.name, self.uncomp, self.bt_offset = name, uncomp, bt_offset
    @property
    def nblocks(self):
        return (self.uncomp + BLOCK - 1) >> 15 if self.uncomp else 0
    def __repr__(self):
        return f"Entry({self.name!r} uncomp={self.uncomp} bt=0x{self.bt_offset:x})"

class DatArchive:
    """Read-only view of a JnG .dat. Files are stored as a sequence of
    LZO1X-compressed 32 KB blocks; each entry points at a block table."""
    def __init__(self, path):
        self.path = path
        self.data = open(path, 'rb').read()
        self.magic, self.count, self.index_offset, self.adler = struct.unpack_from('<IIII', self.data, 0)
        self.entries = self._parse_index()
        self.by_name = {e.name: e for e in self.entries}

    def _parse_index(self):
        d = self.data; pos = self.index_offset; seed = self.index_offset & MASK
        ents = []
        for _ in range(self.count):
            namelen = struct.unpack_from('<H', d, pos)[0]; pos += 2
            name = bytearray(d[pos:pos+namelen]); pos += namelen
            xorbuf(name, seed)
            rec = bytearray(d[pos:pos+8]); pos += 8
            xorbuf(rec, seed)
            uncomp, bt_offset = struct.unpack('<II', rec)
            nm = name.rstrip(b'\x00').decode('latin1').replace('\\', '/')
            ents.append(Entry(nm, uncomp, bt_offset))
            seed = (seed * KEY_MUL) & MASK
        return ents

    def read(self, entry: Entry) -> bytes:
        d = self.data; out = bytearray()
        for b in range(entry.nblocks):
            delta, _adler, comp, _flag = struct.unpack_from('<IIHH', d, entry.bt_offset + b*12)
            want = min(BLOCK, entry.uncomp - b*BLOCK)
            if comp == 0:                       # stored raw 32 KB block
                blk = bytearray(d[delta:delta+BLOCK]); xorbuf(blk, 0)
                out += blk[:want]
            else:
                blk = bytearray(d[delta:delta+comp]); xorbuf(blk, 0)
                out += lzo_decompress(bytes(blk), want)
        assert len(out) == entry.uncomp, f"{entry.name}: {len(out)} != {entry.uncomp}"
        return bytes(out)


def pack(files: dict, out_path: str, magic: int = 0x30444800):
    """Write a JnG .dat archive. `files` maps archive path (either '/' or '\\'
    separators) -> bytes. Suitable as an override overlay listed first in Data.ini
    (lookup is first-match-wins, so overlay entries win over jng.dat)."""
    out = bytearray(16)  # header filled in later
    recs = []            # (name, uncomp, bt_offset)
    for name, content in files.items():
        content = bytes(content)
        uncomp = len(content)
        nblocks = (uncomp + BLOCK - 1) >> 15 if uncomp else 0
        bt_offset = len(out)
        table_pos = len(out)
        out += b'\x00' * (nblocks * 12)     # reserve block table
        table = bytearray()
        for i in range(nblocks):
            chunk = content[i*BLOCK : i*BLOCK + min(BLOCK, uncomp - i*BLOCK)]
            c = bytes(lzo_compress(chunk))
            if len(c) >= BLOCK:               # would overflow game's 0x8000 read buffer -> store raw
                raw = bytearray(chunk) + bytes(BLOCK - len(chunk))   # pad to 0x8000; game uses only `want` bytes
                xorbuf(raw, 0)
                delta = len(out); out += raw
                table += struct.pack('<IIHH', delta, zlib.adler32(bytes(raw), 0) & MASK, 0, 0)  # comp=0 => raw
            else:
                xc = bytearray(c); xorbuf(xc, 0)  # block data XOR'd with seed 0
                delta = len(out); out += xc
                table += struct.pack('<IIHH', delta, zlib.adler32(c, 0) & MASK, len(c), 0)
        out[table_pos:table_pos + len(table)] = table
        recs.append((name, uncomp, bt_offset))

    index_offset = len(out)
    struct.pack_into('<III', out, 0, magic, len(recs), index_offset)
    struct.pack_into('<I', out, 12, zlib.adler32(bytes(out[0:12]), 0) & MASK)  # game seeds adler with 0
    seed = index_offset & MASK
    for name, uncomp, bt in recs:
        raw = (name.replace('/', '\\') + '\x00').encode('latin1')  # backslash + NUL like original
        nb = bytearray(raw); xorbuf(nb, seed)
        rec = bytearray(struct.pack('<II', uncomp, bt)); xorbuf(rec, seed)
        out += struct.pack('<H', len(raw)) + nb + rec
        seed = (seed * KEY_MUL) & MASK
    with open(out_path, 'wb') as f:
        f.write(out)
    return len(out)

if __name__ == "__main__":
    import sys
    a = DatArchive(sys.argv[1])
    print(f"{a.path}: magic=0x{a.magic:08x} count={a.count} index_off={a.index_offset}")
    okc = badc = 0
    for e in a.entries:
        try:
            b = a.read(e)
            assert len(b) == e.uncomp
            okc += 1
        except Exception as ex:
            badc += 1
            if badc <= 10:
                print(f"  FAIL {e}: {ex}")
    print(f"decompressed OK={okc} FAIL={badc}")
