$ErrorActionPreference = 'Stop'

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$target = Join-Path $scriptDir 'Launch_TME_Spatial.bat'
$icon = Join-Path $scriptDir 'TME_Spatial.ico'
$shortcutPath = Join-Path $scriptDir 'Launch TME Spatial.lnk'

if (-not (Test-Path $target)) {
    throw "Launch_TME_Spatial.bat was not found in $scriptDir"
}

if (-not (Test-Path $icon)) {
    throw "TME_Spatial.ico was not found in $scriptDir"
}

$shell = New-Object -ComObject WScript.Shell
$shortcut = $shell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = $target
$shortcut.WorkingDirectory = $scriptDir
$shortcut.IconLocation = "$icon,0"
$shortcut.Description = 'Launch the TME Spatial Streamlit app'
$shortcut.Save()

Write-Host "Created shortcut: $shortcutPath"
