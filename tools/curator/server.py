# Curator server: browse every GSR-upscaled asset, paint a mask, have FLUX
# repaint only inside the mask (feathered, alpha-weighted), pick one of six
# candidates or keep the plain GSR upscale. All state lives under
# lab-curator/ (gitignored) so a session can be stopped and resumed.
#
# Runs inside the ROCm container (see run.sh); torch is only imported by the
# generation worker thread, so the UI is browsable before the GPU warms up.
import base64
import glob
import io
import json
import os
import queue
import shutil
import struct
import threading
import time
import uuid

import numpy as np
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = os.environ.get("CURATOR_ROOT", "/work")
CACHE = os.path.join(ROOT, os.environ.get(
    "CURATOR_CACHE", "upscaled_gsr/4x-UltraSharpV2_Lite_aa"))
ASSETS_DIR = os.path.join(ROOT, "assets")
DATA = os.path.join(ROOT, "lab-curator")
EXTS = {".bmp", ".tga", ".jpg"}

DEFAULTS = {  # the values validated by the lab-flux test batches
    "strength": 0.25, "steps": 32, "guidance": 3.5, "seed": 7, "feather": 12,
    "v2_scale": 1.0, "v3_scale": 1.0, "use_v2": True, "use_v3": True,
    "prompt": "",
}
SEEDS_PER_FAMILY = 3

app = FastAPI()


# ---- disk helpers (container runs as root; keep host-user-writable) ---------
def _mkdirs(path):
    made = []
    p = path
    while p and not os.path.isdir(p):
        made.append(p)
        p = os.path.dirname(p)
    os.makedirs(path, exist_ok=True)
    for p in made:
        try:
            os.chmod(p, 0o777)
        except OSError:
            pass


def _save_bytes(path, data):
    _mkdirs(os.path.dirname(path))
    with open(path, "wb") as f:
        f.write(data)
    os.chmod(path, 0o666)


def _save_img(path, im, **kw):
    buf = io.BytesIO()
    im.save(buf, **kw)
    _save_bytes(path, buf.getvalue())


def _save_json(path, obj):
    _save_bytes(path, json.dumps(obj, indent=1).encode())


# ---- asset list + state -----------------------------------------------------
def list_assets():
    out = []
    for dirpath, _, files in os.walk(CACHE):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                out.append(os.path.relpath(os.path.join(dirpath, f), CACHE))
    return sorted(out)

ASSET_LIST = list_assets()
ASSET_SET = set(ASSET_LIST)

STATE_PATH = os.path.join(DATA, "state.json")
STATE_LOCK = threading.Lock()
try:
    with open(STATE_PATH) as f:
        STATE = json.load(f)
except (OSError, ValueError):
    STATE = {"assets": {}}


def state_get(rel):
    return STATE["assets"].get(rel, {})


def state_update(rel, **kw):
    with STATE_LOCK:
        ent = STATE["assets"].setdefault(rel, {})
        ent.update(kw)
        ent["updated"] = int(time.time())
        _save_json(STATE_PATH, STATE)


def check_rel(rel):
    if rel not in ASSET_SET:
        raise HTTPException(404, f"unknown asset {rel!r}")
    return rel


def relkey(rel):
    return rel.replace("/", "__")


def mask_path(rel):
    return os.path.join(DATA, "masks", rel + ".png")


def gen_dir(rel):
    return os.path.join(DATA, "gen", relkey(rel))


def chosen_path(rel):
    return os.path.join(DATA, "chosen", rel)


def history_path(rel, stage):
    return os.path.join(DATA, "history", rel + f".stage{stage}")


def clear_history(rel):
    for f in glob.glob(os.path.join(DATA, "history",
                                    glob.escape(rel) + ".stage*")):
        os.remove(f)


# ---- imaging ----------------------------------------------------------------
def load_cache(rel):
    return Image.open(os.path.join(CACHE, rel))


def load_work(rel):
    """Current working image: the last saved stage, or the plain GSR upscale."""
    p = chosen_path(rel)
    return Image.open(p if os.path.exists(p) else os.path.join(CACHE, rel))


def box16(mask, W, H, pad=48, minside=320):
    """Padded, /16-aligned crop box around the painted mask."""
    ys, xs = np.nonzero(mask > 8)
    if xs.size == 0:
        raise ValueError("the mask is empty — paint the region to refine first")

    def axis(lo, hi, limit):
        lo = max(0, int(lo) - pad)
        hi = min(limit, int(hi) + 1 + pad)
        need = max(minside, hi - lo)
        need = min(need + (-need) % 16, limit - limit % 16 or limit)
        lo = max(0, min(lo, limit - need))
        return lo, lo + need

    x0, x1 = axis(xs.min(), xs.max(), W)
    y0, y1 = axis(ys.min(), ys.max(), H)
    return x0, y0, x1, y1


def feathered_weight(mask_l, alpha, feather):
    """float32 HxW blend weight: painted mask, blurred, killed on transparency."""
    m = mask_l
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    w = np.asarray(m, np.float32) / 255.0
    return w * (np.asarray(alpha, np.float32) / 255.0)


def composite_crop(gsr_rgba, flux_rgb, mask_l, feather):
    """Blend the FLUX crop into the GSR crop inside the feathered mask."""
    a = np.asarray(gsr_rgba, np.float32)
    w = feathered_weight(mask_l, gsr_rgba.getchannel("A"), feather)[..., None]
    rgb = a[..., :3] * (1 - w) + np.asarray(flux_rgb, np.float32) * w
    out = np.concatenate([rgb, a[..., 3:]], axis=2)
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8), "RGBA")


def enc_bmp32(rgba):
    """32-bit BGRA BMP with a BITMAPV4HEADER (explicit alpha masks) — Pillow's
    BMP writer drops alpha. Keep in sync with tools/gsr/build_hd_gsr.py."""
    h, w = rgba.shape[:2]
    data = np.ascontiguousarray(rgba[::-1, :, [2, 1, 0, 3]]).tobytes()
    dib = struct.pack(
        "<IiiHHIIiiII" "IIII" "I" "36x" "III",
        108, w, h, 1, 32, 3, len(data), 2835, 2835, 0, 0,  # 3 = BI_BITFIELDS
        0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,    # R, G, B, A masks
        0x73524742,                                        # 'sRGB'
        0, 0, 0)
    off = 14 + len(dib)
    return struct.pack("<2sIHHI", b"BM", off + len(data), 0, 0, off) + dib + data


def save_like_cache(rel, im_rgba, path):
    """Write in the same format/mode the pipeline uses for this cache file."""
    orig = load_cache(rel)
    ext = os.path.splitext(rel)[1].lower()
    _mkdirs(os.path.dirname(path))
    if ext == ".jpg":
        _save_img(path, im_rgba.convert("RGB"), format="JPEG",
                  quality=94, subsampling=1)
    elif ext == ".tga":
        _save_img(path, im_rgba.convert(orig.mode), format="TGA")
    elif orig.mode == "RGBA":
        _save_bytes(path, enc_bmp32(np.asarray(im_rgba.convert("RGBA"),
                                               np.uint8)))
    else:
        _save_img(path, im_rgba.convert(orig.mode), format="BMP")


def decode_mask(data_url, size):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    m = Image.open(io.BytesIO(raw)).convert("L")
    if m.size != size:
        m = m.resize(size, Image.NEAREST)
    return m


# ---- generation worker ------------------------------------------------------
JOBS = {}
JOB_Q = queue.Queue()
ENGINE = None


def _run_job(job):
    rel, p = job["rel"], job["params"]
    jid = job["id"]

    def msg(text):
        JOBS[jid]["message"] = text

    global ENGINE
    if ENGINE is None:
        msg("loading FLUX pipeline (first job — takes a minute)")
        import engine as engine_mod
        ENGINE = engine_mod.Engine()

    im = load_work(rel).convert("RGBA")
    mask_l = Image.open(mask_path(rel)).convert("L")
    box = box16(np.asarray(mask_l), im.width, im.height)
    crop = im.crop(box)
    mask_crop = mask_l.crop(box)
    flux_in = Image.alpha_composite(
        Image.new("RGBA", crop.size, (96, 96, 96, 255)), crop).convert("RGB")

    gd = gen_dir(rel)
    _mkdirs(gd)
    _save_img(os.path.join(gd, "mask.png"), mask_l, format="PNG")
    _save_img(os.path.join(gd, "gsr.png"), crop, format="PNG")
    _save_img(os.path.join(gd, "input.png"), flux_in, format="PNG")

    families = []
    if p["use_v2"]:
        families.append(("v2", p["v2_scale"]))
    if p["use_v3"]:
        families.append(("v3", p["v3_scale"]))
    if not families:
        raise ValueError("both LoRA families are unchecked")
    n_seeds = int(p.get("_n", SEEDS_PER_FAMILY))
    JOBS[jid]["total"] = n_seeds * len(families)

    ENGINE.encode(p["prompt"], log=msg)
    options = []
    for fam, scale in families:
        ENGINE.ensure(fam, scale, log=msg)
        for k in range(n_seeds):
            seed = int(p["seed"]) + k
            msg(f"{fam} ×{scale} · seed {seed} · "
                f"{crop.width}×{crop.height}px · {p['steps']} steps")
            out = ENGINE.generate(flux_in, p["prompt"], p["strength"],
                                  p["steps"], p["guidance"], seed)
            raw_fn = f"raw_{fam}_s{seed}.png"
            opt_fn = f"opt_{fam}_s{seed}.png"
            _save_img(os.path.join(gd, raw_fn), out, format="PNG")
            preview = composite_crop(crop, out, mask_crop, p["feather"])
            _save_img(os.path.join(gd, opt_fn), preview, format="PNG")
            options.append({"file": opt_fn, "raw": raw_fn, "family": fam,
                            "scale": scale, "seed": seed,
                            "label": f"{fam} ×{scale} · seed {seed}"})
            JOBS[jid]["done"] += 1
            JOBS[jid]["options"] = list(options)
            # progressive, so an option can be chosen mid-batch
            _save_json(os.path.join(gd, "meta.json"),
                       {"box": list(map(int, box)), "params": p,
                        "base_stage": int(p.get("_stage", 0)),
                        "options": options})
    msg("done")


def _worker():
    while True:
        job = JOB_Q.get()
        JOBS[job["id"]]["status"] = "running"
        try:
            _run_job(job)
            JOBS[job["id"]]["status"] = "done"
        except Exception as e:  # surfaced in the UI, server keeps serving
            JOBS[job["id"]]["status"] = "error"
            JOBS[job["id"]]["message"] = f"{type(e).__name__}: {e}"


threading.Thread(target=_worker, daemon=True).start()


# ---- API models -------------------------------------------------------------
class MaskIn(BaseModel):
    rel: str
    mask: str  # data URL


class GenerateIn(MaskIn):
    params: dict


class ChooseIn(BaseModel):
    rel: str
    option: str  # "gsr" or an opt_*.png filename from the last batch


class RelIn(BaseModel):
    rel: str


# ---- API --------------------------------------------------------------------
@app.get("/api/assets")
def api_assets():
    assets = [{"rel": rel, "status": state_get(rel).get("status", "todo")}
              for rel in ASSET_LIST]
    counts = {}
    for a in assets:
        counts[a["status"]] = counts.get(a["status"], 0) + 1
    return {"assets": assets, "counts": counts}


@app.get("/api/asset")
def api_asset(rel: str):
    check_rel(rel)
    im = load_cache(rel)
    ent = state_get(rel)
    stage = ent.get("stage", 0)
    d = {"rel": rel, "w": im.width, "h": im.height, "stage": stage,
         "status": ent.get("status", "todo"), "choice": ent.get("choice"),
         "params": {**DEFAULTS, **ent.get("params", {})},
         "has_mask": os.path.exists(mask_path(rel)), "gen": None}
    meta_p = os.path.join(gen_dir(rel), "meta.json")
    if os.path.exists(meta_p):
        with open(meta_p) as f:
            d["gen"] = json.load(f)
        d["gen"]["stale"] = d["gen"].get("base_stage", 0) != stage
    return d


@app.post("/api/mask")
def api_mask(body: MaskIn):
    check_rel(body.rel)
    im = load_cache(body.rel)
    m = decode_mask(body.mask, (im.width, im.height))
    _save_img(mask_path(body.rel), m, format="PNG")
    return {"ok": True}


@app.post("/api/generate")
def api_generate(body: GenerateIn):
    check_rel(body.rel)
    api_mask(body)
    ent = state_get(body.rel)
    stage = ent.get("stage", 0)
    gens = ent.get("gens", 0)
    p = {**DEFAULTS, **body.params}
    state_update(body.rel, params=p, gens=gens + 1)
    run_p = dict(p)
    run_p["_stage"] = stage
    if stage > 0:
        # refinement pass: one sample per checked family, and roll the seed
        # each press so repeated generates give new samples, not the same one
        run_p["_n"] = 1
        run_p["seed"] = int(p["seed"]) + gens
    else:
        run_p["_n"] = SEEDS_PER_FAMILY
    jid = uuid.uuid4().hex[:12]
    JOBS[jid] = {"status": "queued", "done": 0, "total": 0,
                 "message": "queued", "options": [], "rel": body.rel}
    JOB_Q.put({"id": jid, "rel": body.rel, "params": run_p})
    return {"job": jid}


@app.get("/api/job")
def api_job(id: str):
    if id not in JOBS:
        raise HTTPException(404, "unknown job")
    return JOBS[id]


@app.post("/api/choose")
def api_choose(body: ChooseIn):
    rel = check_rel(body.rel)
    ent = state_get(rel)
    stage = ent.get("stage", 0)
    if body.option == "gsr":
        cp = chosen_path(rel)
        if os.path.exists(cp):
            os.remove(cp)
        clear_history(rel)
        state_update(rel, status="gsr", choice="gsr", stage=0)
        return {"ok": True, "stage": 0}
    gd = gen_dir(rel)
    with open(os.path.join(gd, "meta.json")) as f:
        meta = json.load(f)
    if meta.get("base_stage", 0) != stage:
        raise HTTPException(409, "these options were generated against a "
                            "previous stage — generate again first")
    opt = next((o for o in meta["options"] if o["file"] == body.option), None)
    if opt is None:
        raise HTTPException(404, f"option {body.option!r} not in last batch")
    im = load_work(rel).convert("RGBA")
    box = tuple(meta["box"])
    flux = Image.open(os.path.join(gd, opt["raw"])).convert("RGB")
    mask_l = Image.open(os.path.join(gd, "mask.png")).convert("L").crop(box)
    blended = composite_crop(im.crop(box), flux, mask_l,
                             meta["params"]["feather"])
    im.paste(blended, box)
    cp = chosen_path(rel)
    if os.path.exists(cp):  # keep the outgoing stage for undo
        hp = history_path(rel, stage)
        _mkdirs(os.path.dirname(hp))
        shutil.copy2(cp, hp)
    save_like_cache(rel, im, cp)
    mp = mask_path(rel)  # the mask is consumed; the next stage starts fresh
    if os.path.exists(mp):
        os.remove(mp)
    state_update(rel, status="flux", choice=body.option, stage=stage + 1)
    return {"ok": True, "stage": stage + 1}


@app.post("/api/undo")
def api_undo(body: RelIn):
    rel = check_rel(body.rel)
    stage = state_get(rel).get("stage", 0)
    if stage == 0:
        raise HTTPException(400, "nothing to undo")
    cp = chosen_path(rel)
    hp = history_path(rel, stage - 1)
    if stage > 1 and os.path.exists(hp):
        shutil.copy2(hp, cp)
        os.remove(hp)
        state_update(rel, stage=stage - 1, status="flux", choice=None)
    else:
        if os.path.exists(cp):
            os.remove(cp)
        state_update(rel, stage=0, status="todo", choice=None)
    return {"ok": True, "stage": max(0, stage - 1)}


@app.post("/api/skip")
def api_skip(body: RelIn):
    state_update(check_rel(body.rel), status="skip")
    return {"ok": True}


@app.post("/api/reset")
def api_reset(body: RelIn):
    rel = check_rel(body.rel)
    cp = chosen_path(rel)
    if os.path.exists(cp):
        os.remove(cp)
    clear_history(rel)
    state_update(rel, status="todo", choice=None, stage=0)
    return {"ok": True}


# ---- images -----------------------------------------------------------------
def _png_response(im):
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.get("/img/orig")
def img_orig(rel: str):
    check_rel(rel)
    p = os.path.join(ASSETS_DIR, rel)
    if not os.path.exists(p):
        raise HTTPException(404, "no original")
    return _png_response(Image.open(p).convert("RGBA"))


@app.get("/img/up")
def img_up(rel: str):
    return _png_response(load_cache(check_rel(rel)).convert("RGBA"))


@app.get("/img/work")
def img_work(rel: str):
    return _png_response(load_work(check_rel(rel)).convert("RGBA"))


@app.get("/img/mask")
def img_mask(rel: str):
    # stored as L internally; served white-with-alpha so the browser canvas
    # can stamp it straight back into its alpha-based mask layer
    p = mask_path(check_rel(rel))
    if not os.path.exists(p):
        raise HTTPException(404, "no mask")
    m = Image.open(p).convert("L")
    white = Image.new("L", m.size, 255)
    return _png_response(Image.merge("RGBA", (white, white, white, m)))


@app.get("/img/gen")
def img_gen(rel: str, f: str):
    check_rel(rel)
    if "/" in f or ".." in f:
        raise HTTPException(400, "bad filename")
    p = os.path.join(gen_dir(rel), f)
    if not os.path.exists(p):
        raise HTTPException(404, "no such file")
    return FileResponse(p, headers={"Cache-Control": "no-store"})


@app.get("/img/chosen")
def img_chosen(rel: str):
    p = chosen_path(check_rel(rel))
    if not os.path.exists(p):
        raise HTTPException(404, "nothing chosen")
    return _png_response(Image.open(p).convert("RGBA"))


@app.get("/")
def index():
    return RedirectResponse("/static/index.html")


app.mount("/static", StaticFiles(
    directory=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "static")), name="static")
