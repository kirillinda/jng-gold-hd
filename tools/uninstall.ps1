<#
uninstall.ps1 - restore the stock game after the HD + widescreen mod (Windows).

Run from your JnG Gold folder, or pass its path:
  powershell -ExecutionPolicy Bypass -File uninstall.ps1 ["C:\path\to\JnG Gold"]

Restores the *.orig backups (jng_gold.exe, Data.ini), removes hd.dat / ws.dat, and
restores Documents\JnGGold\Game.ini from its .orig backup. Close the game first.
#>
[CmdletBinding()]
param([string]$GameDir = $PWD.Path)

$ErrorActionPreference = "Stop"

foreach ($f in @("jng_gold.exe", "Data.ini")) {
    $dst = Join-Path $GameDir $f
    $bak = "$dst.orig"
    if (Test-Path $bak) {
        Move-Item $bak $dst -Force
        Write-Host "restored $f"
    }
}
foreach ($f in @("hd.dat", "ws.dat")) {
    $p = Join-Path $GameDir $f
    if (Test-Path $p) { Remove-Item $p -Force; Write-Host "removed $f" }
}

$ini = Join-Path ([Environment]::GetFolderPath("MyDocuments")) "JnGGold\Game.ini"
if (Test-Path "$ini.orig") {
    Move-Item "$ini.orig" $ini -Force
    Write-Host "restored Game.ini (widescreen reverted)"
} else {
    Write-Host "note: no Game.ini.orig backup; leaving Game.ini as-is." -ForegroundColor Yellow
    Write-Host "      To revert widescreen manually, remove Width/Height from [VIDEO] in $ini"
}

Write-Host "`nStock game restored." -ForegroundColor Green
