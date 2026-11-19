# The GPU vision server (YOLOE + depth on DirectML) that the robot stack in WSL uses, on 127.0.0.1:8090.
#   .\tools\win\vision.ps1          # start it in the background if it is not running (log: %TEMP%\sprocket-vision.log)
#   .\tools\win\vision.ps1 -Stop
# Needs the venv %USERPROFILE%\sprocket-win (onnxruntime-directml, numpy); its models are in the repo's
# models\vision\ (made by tools/wsl/setup_models.sh --gpu-models); see docs/usage.md.
param([switch]$Stop, [switch]$Serve)
$health = "http://127.0.0.1:8090/health"
function Test-Ready { try { return (Invoke-RestMethod $health -TimeoutSec 2).status -eq "ok" } catch { return $false } }
function Get-Server {
    Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*vision_server.py*" }
}

if ($Stop) { Get-Server | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -Confirm:$false }; "stopped"; return }
$python = Join-Path $env:USERPROFILE "sprocket-win\Scripts\python.exe"
$models = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) "models\vision"
if ($Serve) {  # the server itself, in this window; the background start below runs this
    & $python (Join-Path $PSScriptRoot "vision_server.py") --models $models *> (Join-Path $env:TEMP "sprocket-vision.log")
    return
}
if (Test-Ready) { "vision server already running"; return }
if (-not (Test-Path $python)) { "no venv at ${python}; see docs/usage.md"; exit 1 }
# started through WMI, so it outlives this window and the session that ran it
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = "powershell -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`" -Serve" } |
    Out-Null
for ($i = 0; $i -lt 60; $i++) {
    if (Test-Ready) { "vision server ready on 127.0.0.1:8090"; return }
    Start-Sleep -Seconds 1
}
"vision server did not start: see $(Join-Path $env:TEMP 'sprocket-vision.log')"; exit 1
