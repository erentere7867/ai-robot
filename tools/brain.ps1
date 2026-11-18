# Start the robot's model (LFM3-VL-2B) with llama.cpp on Windows, on port 8081, on the GPU.
#   .\tools\brain.ps1            # models from $env:SPROCKET_MODELS_DIR, else Downloads\sprocket-portable-bundle\sprocket-models
#   .\tools\brain.ps1 -Stop
# Needs llama.cpp build 11960 or newer (LFM3 support): $env:LLAMA_SERVER, else C:\llama-vulkan\llama-server.exe, else the
# one on the PATH. WSL reaches it at http://127.0.0.1:8081 too.
param(
    [switch]$Stop,
    [string]$Models = $(if ($env:SPROCKET_MODELS_DIR) { $env:SPROCKET_MODELS_DIR } else {
        Join-Path $env:USERPROFILE "Downloads\sprocket-portable-bundle\sprocket-models" })
)
$health = "http://127.0.0.1:8081/health"
function Test-Ready { try { return (Invoke-RestMethod $health -TimeoutSec 2).status -eq "ok" } catch { return $false } }

if ($Stop) { Stop-Process -Name llama-server -ErrorAction SilentlyContinue; "stopped"; return }
$exe = @($env:LLAMA_SERVER, "C:\llama-vulkan\llama-server.exe", (Get-Command llama-server -ErrorAction SilentlyContinue).Source) |
    Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if (-not $exe) { throw "no llama-server found: set LLAMA_SERVER to a llama.cpp build 11960 or newer" }
$model = Join-Path $Models "llm\LFM3-VL-2B-QAD-Q4_0.gguf"
$proj = Join-Path $Models "llm\mmproj-LFM3-VL-2B-F16.gguf"
foreach ($f in $model, $proj) { if (-not (Test-Path $f)) { throw "missing $f (set SPROCKET_MODELS_DIR or pass -Models)" } }
if (Test-Ready) { "the model is already running on port 8081"; return }

$log = Join-Path $env:TEMP "sprocket-llama.log"
$serverArgs = @("-m", "`"$model`"", "--mmproj", "`"$proj`"", "-ngl", "99", "-c", "32768", "-np", "2", "--jinja",
                "--host", "127.0.0.1", "--port", "8081")
Start-Process -FilePath $exe -ArgumentList $serverArgs -WindowStyle Hidden -RedirectStandardError $log
"starting LFM3-VL-2B with $exe..."
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    if (Test-Ready) { "the model is ready on port 8081 (stop it with .\tools\brain.ps1 -Stop)"; return }
}
throw "the model did not start: see $log"
