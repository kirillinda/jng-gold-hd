<#
install.ps1 - install the Jets'n'Guns Gold HD + widescreen mod (Windows).

Run from your JnG Gold game folder (where jng_gold.exe lives), or pass its path:
  powershell -ExecutionPolicy Bypass -File install.ps1 ["C:\path\to\JnG Gold"]

It:
  * backs up the stock jng_gold.exe / Data.ini to *.orig (once),
  * installs the patched jng_gold.exe, hd.dat and an overlay-first Data.ini,
  * enables Hor+ widescreen (Width=1067, Height=600, ratio43=0) in your
    Documents\JnGGold\Game.ini, backing it up to Game.ini.orig first.

Reverse everything with uninstall.ps1. Close the game (and Steam's overlay)
before running so the files aren't locked.
#>
[CmdletBinding()]
param([string]$GameDir = $PWD.Path)

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path (Join-Path $GameDir "jng_gold.exe"))) {
    Write-Host "Run from your JnG Gold folder, or pass its path: install.ps1 'C:\path\to\JnG Gold'" -ForegroundColor Red
    exit 1
}

# 1. Back up originals (once), then install the changed files.
foreach ($f in @("jng_gold.exe", "Data.ini")) {
    $dst = Join-Path $GameDir $f
    $bak = "$dst.orig"
    if ((Test-Path $dst) -and -not (Test-Path $bak)) {
        Copy-Item $dst $bak
        Write-Host "backed up $f -> $f.orig"
    }
}
foreach ($f in @("jng_gold.exe", "hd.dat", "Data.ini")) {
    Copy-Item (Join-Path $Here $f) (Join-Path $GameDir $f) -Force
    Write-Host "installed $f"
}

# 2. Enable Hor+ widescreen in Documents\JnGGold\Game.ini.
$iniDir = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "JnGGold"
$ini    = Join-Path $iniDir "Game.ini"
$desired = [ordered]@{ Width = "1067"; Height = "600"; ratio43 = "0" }

if (-not (Test-Path $ini)) {
    New-Item -ItemType Directory -Force $iniDir | Out-Null
    "[VIDEO]" | Out-File $ini -Encoding ascii
    Write-Host "created $ini"
} elseif (-not (Test-Path "$ini.orig")) {
    Copy-Item $ini "$ini.orig"
    Write-Host "backed up Game.ini -> Game.ini.orig"
}

# Line-based [VIDEO] editor: update keys in place, append any that are missing.
$lines = [System.Collections.Generic.List[string]](Get-Content $ini)
$inVideo = $false; $videoStart = -1; $videoEnd = $lines.Count
$seen = @{}
for ($i = 0; $i -lt $lines.Count; $i++) {
    $l = $lines[$i].Trim()
    if ($l -match '^\[(.+)\]$') {
        if ($inVideo) { $videoEnd = $i; break }
        if ($matches[1] -ieq "VIDEO") { $inVideo = $true; $videoStart = $i }
        continue
    }
    if ($inVideo -and $l -match '^\s*([^;=]+?)\s*=') {
        $key = $matches[1].Trim()
        if ($desired.Contains($key)) {
            $lines[$i] = "$key=$($desired[$key])"
            $seen[$key] = $true
        }
    }
}
if ($videoStart -lt 0) {          # no [VIDEO] section at all
    $lines.Add("[VIDEO]"); $videoStart = $lines.Count - 1; $videoEnd = $lines.Count
}
$insertAt = $videoEnd
foreach ($k in $desired.Keys) {
    if (-not $seen[$k]) {
        $lines.Insert($insertAt, "$k=$($desired[$k])"); $insertAt++
    }
}
Set-Content -Path $ini -Value $lines -Encoding ascii
Write-Host "widescreen enabled in $ini (Width=1067 Height=600 ratio43=0)"

Write-Host "`nDone. Launch the game normally (Steam). Uninstall with uninstall.ps1." -ForegroundColor Green
