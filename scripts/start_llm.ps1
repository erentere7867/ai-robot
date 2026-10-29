param(
    [int]$Port = 8080,
    [int]$GpuLayers = 99,
    [int]$CtxSize = 4096,
    [switch]$IncludeProjector
)

$ErrorActionPreference = "Stop"
$ServerBin = Join-Path $env:USERPROFILE "robot-runtime\llama.cpp\llama-server.exe"
$ModelPath = Join-Path $env:USERPROFILE "Downloads\LFM3-VL-2B-Q4_K_M.gguf"
$MmprojPath = Join-Path $env:USERPROFILE "Downloads\mmproj-LFM3-VL-2B-Q8_0.gguf"

if (-not (Test-Path $ServerBin)) {
    throw "llama-server.exe not found at $ServerBin"
}
if (-not (Test-Path $ModelPath)) {
    throw "Model not found at $ModelPath"
}

$ArgsList = @(
    "-m", $ModelPath,
    "--port", "$Port",
    "--host", "127.0.0.1",
    "-ngl", "$GpuLayers",
    "-c", "$CtxSize",
    "-np", "1",
    "--reasoning", "off",
    "--temp", "0"
)

if ($IncludeProjector -and (Test-Path $MmprojPath)) {
    $ArgsList += @("-mm", $MmprojPath)
}

Write-Host "Starting llama-server on http://127.0.0.1:$Port ..."
& $ServerBin @ArgsList
