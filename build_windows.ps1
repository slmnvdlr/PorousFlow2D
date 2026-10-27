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
        & $PyLauncher.Source $Version -c "import sys; assert (3, 11) <= sys.version_info < (3, 14)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $PythonExecutable = $PyLauncher.Source
            $PythonPrefix = @($Version)
            break
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

& $PythonExecutable @PythonPrefix -c "import nuitka" 2>$null
if ($LASTEXITCODE -ne 0) {
    throw 'Nuitka is not installed. Run: python -m pip install -e ".[distribution]"'
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
$env:PYTHONPATH = Join-Path $ProjectRoot "src"

$Mode = if ($OneFile) { "onefile" } else { "standalone" }
$Arguments = @(
    "-m", "nuitka",
    "--mode=$Mode",
    "--enable-plugin=tk-inter",
    "--module-parameter=numba-disable-jit=no",
    "--windows-console-mode=disable",
    "--output-dir=$OutputRoot",
    "--output-filename=PorousFlow2D.exe",
    "--include-package=porous_fvm",
    "--include-package=pyamg",
    "--include-package=numba",
    "--include-package=llvmlite",
    "--nofollow-import-to=numba.tests",
    "--nofollow-import-to=llvmlite.tests",
    "--include-package=xlsxwriter",
    "--product-name=PorousFlow 2D",
    "--file-description=PorousFlow 2D simulator",
    "--file-version=0.1.0.0",
    "--product-version=0.1.0.0",
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
    $Distribution = Get-ChildItem $OutputRoot -Directory -Filter "*.dist" |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($null -eq $Distribution) {
        throw "Nuitka finished successfully, but no *.dist directory was found."
    }
    Write-Host "Done. Distribute the complete directory: $($Distribution.FullName)"
}
