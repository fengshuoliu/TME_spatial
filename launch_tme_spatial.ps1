$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvName = "TME_spatial"
$AppPath = Join-Path $ScriptRoot "app.py"
$RequirementsPath = Join-Path $ScriptRoot "requirements.txt"
$VenvPath = Join-Path $ScriptRoot $EnvName
$LauncherLogPath = Join-Path $ScriptRoot "launcher_log.txt"
$MinimumPythonMajor = 3
$MinimumPythonMinor = 11
$PythonDownloadUrl = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"

function Write-Status {
    param(
        [Parameter(Mandatory = $true)][string]$Message,
        [ConsoleColor]$Color = [ConsoleColor]::Yellow
    )

    $timestamp = Get-Date -Format "HH:mm:ss"
    Write-Host "[$timestamp] $Message" -ForegroundColor $Color
}

function Write-Section {
    param([string]$Message)
    Write-Host ""
    Write-Host "==== $Message ====" -ForegroundColor Cyan
}

function Set-LauncherWindowTitle {
    param([string]$Message)
    try {
        $host.UI.RawUI.WindowTitle = "TME Spatial Launcher - $Message"
    }
    catch {
    }
}

function Format-CommandForDisplay {
    param(
        [string]$FilePath,
        [string[]]$Arguments = @()
    )

    $parts = @($FilePath) + @($Arguments)
    return ($parts | ForEach-Object {
        if ($_ -match '\s') {
            '"{0}"' -f $_
        }
        else {
            $_
        }
    }) -join ' '
}

function Test-CommandExists {
    param([string]$CommandName)
    return $null -ne (Get-Command $CommandName -ErrorAction SilentlyContinue)
}

function Refresh-ProcessPath {
    $machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
    $userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
    $pathParts = @($machinePath, $userPath) | Where-Object { -not [string]::IsNullOrWhiteSpace($_) }

    if ($pathParts.Count -gt 0) {
        $env:Path = $pathParts -join ";"
    }
}

function Start-LauncherTranscript {
    try {
        Start-Transcript -Path $LauncherLogPath -Force | Out-Null
    }
    catch {
    }
}

function Stop-LauncherTranscript {
    try {
        Stop-Transcript | Out-Null
    }
    catch {
    }
}

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory = $ScriptRoot,
        [string]$StatusMessage = "Running command"
    )

    $displayCommand = Format-CommandForDisplay -FilePath $FilePath -Arguments $Arguments
    Set-LauncherWindowTitle $StatusMessage
    Write-Status $StatusMessage
    Write-Host "Command: $displayCommand" -ForegroundColor DarkGray

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        $joined = if ($Arguments.Count -gt 0) { $Arguments -join ' ' } else { '' }
        throw "Command failed: $FilePath $joined"
    }

    Write-Status "Completed: $StatusMessage" ([ConsoleColor]::Green)
}

function Get-CondaCommand {
    foreach ($name in @("conda", "conda.exe", "conda.bat")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }
    return $null
}

function Ensure-CondaEnvironment {
    param([string]$CondaCommand)

    Set-LauncherWindowTitle "Using Conda"
    Write-Section "Using Conda"
    Write-Host "Conda detected at: $CondaCommand"
    Write-Status "Checking existing Conda environments"

    $envInfoJson = & $CondaCommand info --envs --json
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query conda environments."
    }

    $envInfo = $envInfoJson | ConvertFrom-Json
    $envNames = @($envInfo.envs | ForEach-Object { Split-Path $_ -Leaf })
    if ($envNames -notcontains $EnvName) {
        Write-Section "Creating Conda environment"
        Invoke-Step -FilePath $CondaCommand -Arguments @("create", "-y", "-n", $EnvName, "python=3.11") -StatusMessage "Creating Conda environment $EnvName"
    } else {
        Write-Host "Conda environment '$EnvName' already exists."
    }

    Write-Section "Installing Python dependencies"
    Invoke-Step -FilePath $CondaCommand -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -StatusMessage "Upgrading pip tools in Conda environment"
    Invoke-Step -FilePath $CondaCommand -Arguments @("run", "-n", $EnvName, "python", "-m", "pip", "install", "-r", $RequirementsPath) -StatusMessage "Installing app requirements in Conda environment"

    Write-Section "Launching Streamlit"
    Write-Status "If Streamlit starts successfully, the app URL will appear below."
    Invoke-Step -FilePath $CondaCommand -Arguments @("run", "--live-stream", "-n", $EnvName, "python", "-m", "streamlit", "run", $AppPath) -StatusMessage "Starting Streamlit app"
}

function Test-PythonVersionSupported {
    param([Version]$Version)

    if (-not $Version) {
        return $false
    }

    if ($Version.Major -gt $MinimumPythonMajor) {
        return $true
    }

    return ($Version.Major -eq $MinimumPythonMajor -and $Version.Minor -ge $MinimumPythonMinor)
}

function Get-PythonProbeResult {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$PrefixArgs = @()
    )

    try {
        $probeOutput = & $FilePath @PrefixArgs -c "import sys; print(sys.executable); print(f'{sys.version_info[0]}.{sys.version_info[1]}')" 2>$null
    }
    catch {
        return $null
    }

    if ($LASTEXITCODE -ne 0 -or -not $probeOutput) {
        return $null
    }

    $outputLines = @($probeOutput | ForEach-Object { "$_".Trim() } | Where-Object { $_ })
    if ($outputLines.Count -lt 2) {
        return $null
    }

    try {
        $version = [Version]::Parse(("{0}.0" -f $outputLines[-1]))
    }
    catch {
        return $null
    }

    return @{
        ExecutablePath = $outputLines[0]
        Version = $version
    }
}

function Get-PythonExecutablesFromRegistry {
    $registryRoots = @(
        "HKCU:\Software\Python\PythonCore",
        "HKLM:\Software\Python\PythonCore",
        "HKLM:\Software\Wow6432Node\Python\PythonCore"
    )
    $pythonExecutables = New-Object System.Collections.Generic.List[string]

    foreach ($root in $registryRoots) {
        if (-not (Test-Path $root)) {
            continue
        }

        foreach ($versionKey in Get-ChildItem -Path $root -ErrorAction SilentlyContinue) {
            $installPathKey = Join-Path $versionKey.PSPath "InstallPath"
            if (-not (Test-Path $installPathKey)) {
                continue
            }

            $installDir = (Get-Item $installPathKey).GetValue("")
            if ([string]::IsNullOrWhiteSpace($installDir)) {
                continue
            }

            $pythonExe = Join-Path $installDir "python.exe"
            if (Test-Path $pythonExe) {
                $pythonExecutables.Add($pythonExe)
            }
        }
    }

    return @($pythonExecutables | Select-Object -Unique)
}

function Get-CommonPythonExecutables {
    $candidates = @(
        (Join-Path $env:LocalAppData "Programs\Python\Python311\python.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"),
        (Join-Path $env:LocalAppData "Programs\Python\Python313\python.exe"),
        (Join-Path $env:ProgramFiles "Python311\python.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Python311\python.exe")
    )

    return @($candidates | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)
}

function Get-InstalledPythonCommand {
    Refresh-ProcessPath
    $commandCandidates = @()

    foreach ($name in @("python", "python3")) {
        if (Test-CommandExists $name) {
            $commandCandidates += @{ FilePath = $name; PrefixArgs = @() }
        }
    }

    foreach ($pythonExe in (Get-PythonExecutablesFromRegistry) + (Get-CommonPythonExecutables)) {
        $commandCandidates += @{ FilePath = $pythonExe; PrefixArgs = @() }
    }

    foreach ($candidate in $commandCandidates) {
        $probe = Get-PythonProbeResult -FilePath $candidate.FilePath -PrefixArgs $candidate.PrefixArgs
        if (-not $probe) {
            continue
        }

        if (Test-PythonVersionSupported -Version $probe.Version) {
            return @{
                FilePath = $probe.ExecutablePath
                PrefixArgs = @()
                ExecutablePath = $probe.ExecutablePath
                Version = $probe.Version
                Source = $candidate.FilePath
            }
        }
    }

    return $null
}

function Install-PythonFromOfficialInstaller {
    Set-LauncherWindowTitle "Installing Python"
    Write-Section "Downloading official Python installer"

    $installerPath = Join-Path $env:TEMP "python-3.11.9-amd64.exe"
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    Write-Status "Downloading Python 3.11 from python.org"
    Invoke-WebRequest -Uri $PythonDownloadUrl -OutFile $installerPath

    if (-not (Test-Path $installerPath)) {
        throw "Could not download the Python installer from $PythonDownloadUrl"
    }

    try {
        Invoke-Step -FilePath $installerPath -Arguments @("/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1", "SimpleInstall=1") -StatusMessage "Installing Python 3.11 for the current user"
    }
    finally {
        Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
    }

    Refresh-ProcessPath
}

function Install-PythonIfNeeded {
    $pythonCommand = Get-InstalledPythonCommand
    if ($pythonCommand) {
        $versionText = $pythonCommand.Version.ToString(2)
        $sourceText = if ($pythonCommand.Source) { " via $($pythonCommand.Source)" } else { "" }
        Write-Status "Python detected: $($pythonCommand.ExecutablePath) (version $versionText)$sourceText" ([ConsoleColor]::Green)
        return $pythonCommand
    }

    Set-LauncherWindowTitle "Installing Python"
    Write-Section "Python not found"
    Write-Host "Python $MinimumPythonMajor.$MinimumPythonMinor or newer is required."
    Write-Host "The launcher will try to install a local copy for the current Windows user."

    if (Test-CommandExists "winget") {
        Write-Status "Trying Python installation through winget"
        & winget install --exact --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -eq 0) {
            Refresh-ProcessPath
            $pythonCommand = Get-InstalledPythonCommand
            if ($pythonCommand) {
                return $pythonCommand
            }
        }
    }

    Write-Status "winget was unavailable or did not produce a usable Python install. Falling back to python.org." ([ConsoleColor]::Yellow)
    Install-PythonFromOfficialInstaller
    $pythonCommand = Get-InstalledPythonCommand
    if ($pythonCommand) {
        return $pythonCommand
    }

    throw "Python 3.11 could not be found after automatic setup. Please install Python 3.11 manually, then run Launch_TME_Spatial.bat again."
}

function Ensure-VenvEnvironment {
    Set-LauncherWindowTitle "Using Python virtual environment"
    Write-Section "Using Python virtual environment"

    $venvPython = Join-Path $VenvPath "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        $PythonCommand = Install-PythonIfNeeded
        $pythonFile = $PythonCommand.FilePath
        $pythonPrefixArgs = @($PythonCommand.PrefixArgs)

        Write-Section "Creating virtual environment"
        Invoke-Step -FilePath $pythonFile -Arguments ($pythonPrefixArgs + @("-m", "venv", $VenvPath)) -StatusMessage "Creating Python virtual environment $EnvName"
    } else {
        Write-Host "Virtual environment already exists at: $VenvPath"
    }

    if (-not (Test-Path $venvPython)) {
        throw "Virtual environment creation did not produce $venvPython"
    }

    Write-Section "Installing Python dependencies"
    Invoke-Step -FilePath $venvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel") -StatusMessage "Upgrading pip tools in virtual environment"
    Invoke-Step -FilePath $venvPython -Arguments @("-m", "pip", "install", "-r", $RequirementsPath) -StatusMessage "Installing app requirements in virtual environment"

    Write-Section "Launching Streamlit"
    Write-Status "If Streamlit starts successfully, the app URL will appear below."
    Invoke-Step -FilePath $venvPython -Arguments @("-m", "streamlit", "run", $AppPath) -StatusMessage "Starting Streamlit app"
}

try {
    Start-LauncherTranscript
    if (-not (Test-Path $AppPath)) {
        throw "Could not find app.py at $AppPath"
    }
    if (-not (Test-Path $RequirementsPath)) {
        throw "Could not find requirements.txt at $RequirementsPath"
    }

    Set-Location $ScriptRoot
    Set-LauncherWindowTitle "Preparing launcher"
    Write-Section "TME Spatial launcher"
    Write-Host "Project folder: $ScriptRoot"
    Write-Status "Checking app files and environment setup"

    $condaCommand = Get-CondaCommand
    if ($condaCommand) {
        Ensure-CondaEnvironment -CondaCommand $condaCommand
    } else {
        Ensure-VenvEnvironment
    }
}
catch {
    Write-Host "" 
    Write-Host "Launcher failed:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}
finally {
    Stop-LauncherTranscript
}
