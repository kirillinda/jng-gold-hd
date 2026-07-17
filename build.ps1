<#
build.ps1 - one-shot builder for the Jets'n'Guns Gold HD + widescreen mod (Windows).

Usage:
  .\build.ps1 [-Model MODEL_NAME] [-GameDir "PATH"] [-Gpu N]

  -Model    Name of a realesrgan-ncnn model (a MODEL.param/MODEL.bin pair) under
            tools\upscaler\models. If omitted, the default model the mod was
            released with (4x_NMKD-Siax_200k) is downloaded and used.
  -GameDir  Path to the game install. Defaults to $env:JNG_GAME_DIR, else the
            standard Steam location. A D:\ (or other) Steam library is common, so
            pass this if the game is not on C:.
  -Gpu      Vulkan device id for the upscaler (default 0). Run the upscaler once
            with -h to list devices if 0 is not your discrete GPU.

What it does (mirrors build.sh on Linux):
  1. sets up a Python venv with Pillow + numpy + lzallright
  2. downloads realesrgan-ncnn-vulkan (Vulkan GPU upscaler) if missing
  3. resolves / downloads the upscale model
  4. unpacks the game's assets from YOUR copy (first run only)
  5. upscales every asset 4x and packs the HD override archive  (build\hd.dat)
     and re-authors the 800-wide level defs for 16:9            (build\ws.dat)
  6. binary-patches the game executable                          (build\jng_gold.exe)
  7. assembles two deliverables under dist\:
       dist\mod-dropin\    only the changed files (+ install/uninstall scripts)
       dist\patched-game\  a full, ready-to-run copy of the patched game

Assumes the Windows Steam build of Jets'n'Guns Gold, version 1.308 ST.
#>
[CmdletBinding()]
param(
    [string]$Model   = "4x_NMKD-Siax_200k",
    [string]$GameDir = $(if ($env:JNG_GAME_DIR) { $env:JNG_GAME_DIR } else { "${env:ProgramFiles(x86)}\Steam\steamapps\common\JnG Gold" }),
    [int]$Gpu        = 0
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Repo

$DefaultModel = "4x_NMKD-Siax_200k"
$UpscDir   = Join-Path $Repo "tools\upscaler"
$ModelsDir = Join-Path $UpscDir "models"
$Venv      = Join-Path $Repo "tools\venv"
$Py        = Join-Path $Venv  "Scripts\python.exe"
$UpscExe   = Join-Path $UpscDir "realesrgan-ncnn-vulkan.exe"

function Log($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Die($m) { Write-Host "Error: $m" -ForegroundColor Red; exit 1 }

if (-not (Test-Path $GameDir))                       { Die "game not found at '$GameDir' (pass -GameDir or set JNG_GAME_DIR)" }
if (-not (Test-Path (Join-Path $GameDir "jng_gold.exe"))) { Die "'$GameDir\jng_gold.exe' missing - is this the Windows build?" }

# Env consumed by the Python tools (config.py / build_batch.py).
$env:JNG_GAME_DIR = $GameDir
$env:HD_MODEL     = $Model
$env:JNG_GPU      = "$Gpu"

# 1. Python environment ------------------------------------------------------
Log "Python venv + dependencies"
if (-not (Test-Path $Py)) {
    $sys = (Get-Command python -ErrorAction SilentlyContinue).Source
    if (-not $sys) { $sys = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" }
    if (-not (Test-Path $sys)) { Die "Python not found. Install Python 3.10+ (winget install Python.Python.3.12) and re-run." }
    & $sys -m venv $Venv
}
& $Py -m pip install --quiet --upgrade pip
& $Py -m pip install --quiet -r (Join-Path $Repo "tools\requirements.txt")
if (-not $?) { Die "pip install failed" }

# 2. Upscaler binary ---------------------------------------------------------
if (-not (Test-Path $UpscExe)) {
    Log "Downloading realesrgan-ncnn-vulkan (Windows)"
    New-Item -ItemType Directory -Force $UpscDir | Out-Null
    $url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip"
    $zip = Join-Path $env:TEMP "realesrgan-windows.zip"
    Invoke-WebRequest -Uri $url -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $UpscDir -Force
    Remove-Item $zip -Force
    if (-not (Test-Path $UpscExe)) { Die "upscaler exe not found after extract at $UpscExe" }
}

# 3. Model -------------------------------------------------------------------
New-Item -ItemType Directory -Force $ModelsDir | Out-Null
if (-not (Test-Path (Join-Path $ModelsDir "$Model.param"))) {
    if ($Model -ne $DefaultModel) {
        Die "model '$Model' not found in $ModelsDir (drop the .param/.bin there, or omit for the default)"
    }
    Log "Downloading default model $DefaultModel"
    $base = "https://github.com/upscayl/custom-models/raw/main/models"
    foreach ($ext in @("param", "bin")) {
        Invoke-WebRequest -Uri "$base/$DefaultModel.$ext" -OutFile (Join-Path $ModelsDir "$DefaultModel.$ext")
    }
}
Log "Using model: $Model"

# 4. Unpack the game's assets (from YOUR copy) if not already present --------
$assetsData = Join-Path $Repo "assets\DATA"
if (-not (Test-Path $assetsData) -or -not (Get-ChildItem $assetsData -ErrorAction SilentlyContinue)) {
    Log "Unpacking assets from your game into assets\"
    & $Py (Join-Path $Repo "tools\extract.py")
    if (-not $?) { Die "asset extraction failed" }
}

# 5. Build the HD override archive ------------------------------------------
Log "Upscaling assets and packing build\hd.dat (this uses the GPU, device $Gpu)"
& $Py (Join-Path $Repo "tools\build_batch.py")
if (-not $?) { Die "hd.dat build failed" }

# 6. Patch the game binary ---------------------------------------------------
# Always patch from the STOCK binary. If the mod is already installed, the
# original is preserved as jng_gold.exe.orig; patching an already-patched binary
# would fail the safety check in patch_hd.py.
$srcBin = Join-Path $GameDir "jng_gold.exe"
if (Test-Path (Join-Path $GameDir "jng_gold.exe.orig")) { $srcBin = Join-Path $GameDir "jng_gold.exe.orig" }
$buildBin = Join-Path $Repo "build\jng_gold.exe"
New-Item -ItemType Directory -Force (Join-Path $Repo "build") | Out-Null
Log "Patching game binary ($srcBin) -> build\jng_gold.exe"
& $Py (Join-Path $Repo "tools\patch_hd.py") $srcBin $buildBin
if (-not $?) { Die "binary patch failed" }

# 7. Assemble deliverables ---------------------------------------------------
Log "Assembling dist\"
$Dropin = Join-Path $Repo "dist\mod-dropin"
$Full   = Join-Path $Repo "dist\patched-game"
Remove-Item -Recurse -Force $Dropin, $Full -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $Dropin, $Full | Out-Null

# ws.dat: the level defs whose coordinates were authored for an 800-wide screen,
# re-authored for the target Width. Pure data, so it applies on Windows too.
Log "Building build\ws.dat (resolution-dependent defs)"
& $Py (Join-Path $Repo "tools\make_widescreen_defs.py") (Join-Path $Repo "build\ws.dat")
if (-not $?) { Die "ws.dat build failed" }

# Data.ini that loads the overlays first (first match wins).
"data_file = ws.dat`r`ndata_file = hd.dat`r`ndata_file = update.dat`r`ndata_file = jng.dat`r`n" |
    Out-File -FilePath (Join-Path $Dropin "Data.ini") -Encoding ascii -NoNewline
Copy-Item $buildBin, (Join-Path $Repo "build\hd.dat"), (Join-Path $Repo "build\ws.dat") $Dropin
Copy-Item (Join-Path $Repo "tools\install.ps1"), (Join-Path $Repo "tools\uninstall.ps1") $Dropin

@"
Jets'n'Guns Gold HD + Widescreen - drop-in mod (Windows)
Copy these files into your game folder:
  $GameDir
Then run install.ps1 from inside that folder (backs up originals), e.g.:
  powershell -ExecutionPolicy Bypass -File install.ps1
It installs jng_gold.exe, hd.dat, Data.ini and enables Hor+ widescreen in
your Documents\JnGGold\Game.ini. Uninstall with uninstall.ps1.
"@ | Out-File -FilePath (Join-Path $Dropin "README.txt") -Encoding utf8

# Full ready-to-run copy of the patched game.
Copy-Item -Recurse -Force (Join-Path $GameDir "*") $Full
Get-ChildItem $Full -Filter "*.orig" | Remove-Item -Force -ErrorAction SilentlyContinue
Remove-Item -Force -ErrorAction SilentlyContinue (Join-Path $Full "hd_test.dat"), (Join-Path $Full "game.log")
Copy-Item -Force $buildBin, (Join-Path $Repo "build\hd.dat"), (Join-Path $Repo "build\ws.dat"), (Join-Path $Dropin "Data.ini") $Full

Log "Done."
Write-Host "  dist\mod-dropin\    -> copy into your game folder, then run install.ps1"
Write-Host "  dist\patched-game\  -> a complete, ready-to-run patched game (widescreen still needs install.ps1's Game.ini edit)"
