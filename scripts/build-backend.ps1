param(
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$BuildVenv = Join-Path $ProjectRoot ".build-venv"
$VenvPython = Join-Path $BuildVenv "Scripts\python.exe"
$PythonVersionArgument = @()
if ($Python -eq "py") {
    $PythonVersionArgument = @("-3.12")
}

& $Python @PythonVersionArgument -m venv $BuildVenv
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt") pyinstaller

Push-Location $ProjectRoot
try {
    & $VenvPython -m PyInstaller --noconfirm --clean --onedir --name clotho-backend `
        --workpath backend/build --distpath backend/dist --specpath backend `
        --add-data "$ProjectRoot\checkpoints;checkpoints" `
        --add-data "$ProjectRoot\static;static" `
        --collect-all torch --collect-all torchvision --collect-all cv2 --collect-all multipart `
        --hidden-import multipart.multipart `
        --hidden-import uvicorn.logging `
        --hidden-import uvicorn.loops.auto `
        --hidden-import uvicorn.protocols.http.auto `
        --hidden-import uvicorn.protocols.websockets.auto `
        web.py
} finally {
    Pop-Location
}
