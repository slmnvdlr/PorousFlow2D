param(
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$OutputRoot = Join-Path $ProjectRoot "release"
$Launcher = Join-Path $ProjectRoot "porous_flow_gui.py"

$PythonExecutable = $null
$PythonPrefix = @()

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $PyLauncher) {
    foreach ($Version in @("-3.12", "-3.13", "-3.11")) {
        try {
            & $PyLauncher.Source $Version -c "import sys; assert (3, 11) <= sys.version_info < (3, 14)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                $PythonExecutable = $PyLauncher.Source
                $PythonPrefix = @($Version)
                break
            }
        } catch {
            # Continue with the next registered version or a direct python.exe.
        }
    }
}

$PythonCandidates = Get-Command python -All -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty Source -Unique
foreach ($Candidate in $PythonCandidates) {
    if ($null -ne $PythonExecutable) {
        break
    }
    try {
        & $Candidate -c "import sys; assert (3, 11) <= sys.version_info < (3, 14)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonExecutable = $Candidate
            break
        }
    } catch {
        # Windows may register an unusable WindowsApps alias.
    }
}
if ($null -eq $PythonExecutable) {
    throw "Python 3.11, 3.12, or 3.13 is required. Python 3.12 is recommended because PyAMG and Numba do not yet provide all required wheels for Python 3.14."
}

& $PythonExecutable @PythonPrefix -c "import PyInstaller" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'PyInstaller is not installed. Run: python -m pip install -e ".[distribution]"'
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$Mode = if ($OneFile) { "onefile" } else { "onedir" }
$BuildRoot = Join-Path $ProjectRoot "build\pyinstaller"
$SpecRoot = Join-Path $BuildRoot "spec"
New-Item -ItemType Directory -Force -Path $SpecRoot | Out-Null

$Arguments = @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--$Mode",
    "--windowed",
    "--name", "PorousFlow2D",
    "--distpath", $OutputRoot,
    "--workpath", $BuildRoot,
    "--specpath", $SpecRoot,
    "--paths", (Join-Path $ProjectRoot "src"),
    "--hidden-import", "numba",
    "--hidden-import", "llvmlite",
    "--hidden-import", "pyamg",
    "--hidden-import", "xlsxwriter",
    "--exclude-module", "numba.tests",
    "--exclude-module", "numba.cuda.tests",
    "--exclude-module", "llvmlite.tests",
    "--exclude-module", "pyamg.tests",
    $Launcher
)

Write-Host "Building PorousFlow 2D ($Mode)..."
& $PythonExecutable @PythonPrefix @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE."
}

if ($OneFile) {
    Write-Host "Done. Distribute release\PorousFlow2D.exe."
} else {
    $Distribution = Join-Path $OutputRoot "PorousFlow2D"
    if (-not (Test-Path -LiteralPath $Distribution -PathType Container)) {
        throw "PyInstaller finished successfully, but the release directory was not found."
    }
    Write-Host "Done. Distribute the complete directory: $Distribution"
}
