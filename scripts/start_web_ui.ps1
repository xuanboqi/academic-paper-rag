$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Frontend = Join-Path $Root "frontend"
$Logs = Join-Path $Root "logs"
$Conda = Join-Path $env:USERPROFILE "anaconda3\Scripts\conda.exe"

if (-not (Test-Path $Conda)) {
    throw "Cannot find conda.exe at $Conda"
}

New-Item -ItemType Directory -Force $Logs | Out-Null

$BackendLog = Join-Path $Logs "backend.log"
$FrontendLog = Join-Path $Logs "frontend.log"

Write-Host "Starting academic-paper-rag backend on http://127.0.0.1:8002"
$BackendCommand = "Set-Location -LiteralPath '$Root'; & '$Conda' run -n LLM uvicorn backend.main:app --host 127.0.0.1 --port 8002 *> '$BackendLog'"
$Backend = Start-Process -WindowStyle Hidden `
    -FilePath "powershell.exe" `
    -WorkingDirectory $Root `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $BackendCommand) `
    -PassThru

Write-Host "Starting academic-paper-rag frontend on http://127.0.0.1:5174"
$FrontendCommand = "Set-Location -LiteralPath '$Frontend'; npm.cmd run dev -- --host 127.0.0.1 *> '$FrontendLog'"
$FrontendProcess = Start-Process -WindowStyle Hidden `
    -FilePath "powershell.exe" `
    -WorkingDirectory $Frontend `
    -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $FrontendCommand) `
    -PassThru

Write-Host ""
Write-Host "Backend PID:  $($Backend.Id)"
Write-Host "Frontend PID: $($FrontendProcess.Id)"
Write-Host "Backend log:  $BackendLog"
Write-Host "Frontend log: $FrontendLog"
Write-Host "Open http://127.0.0.1:5174 in your browser."
Write-Host "Use .\scripts\stop_web_ui.ps1 to stop the background services."
