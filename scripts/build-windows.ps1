param(
    [string]$Python = "py"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

& (Join-Path $PSScriptRoot "build-backend.ps1") -Python $Python
Push-Location (Join-Path $ProjectRoot "desktop")
try {
    npm install
    npm run dist
} finally {
    Pop-Location
}
