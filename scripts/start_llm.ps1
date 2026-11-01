param(
    [int]$Port = 8080,
    [int]$GpuLayers = 99,
    [int]$CtxSize = 16384,
    # Kept for backward compatibility: the vision projector is now always loaded when present.
    [switch]$IncludeProjector,
    [string]$ServerBin = (Join-Path $env:USERPROFILE "robot-runtime\llama.cpp\llama-server.exe"),
    [string]$ModelPath = (Join-Path $env:USERPROFILE "Downloads\LFM3-VL-2B-Q4_K_M.gguf"),
    [string]$MmprojPath = (Join-Path $env:USERPROFILE "Downloads\mmproj-LFM3-VL-2B-F16.gguf")
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ServerBin)) {
    throw "llama-server.exe not found at $ServerBin"
}
if (-not (Test-Path $ModelPath)) {
    throw "Model not found at $ModelPath"
}

# --jinja enables the model's own chat template, which native tool calling requires.
$ArgsList = @(
    "-m", $ModelPath,
    "--port", "$Port",
    "--host", "127.0.0.1",
    "-ngl", "$GpuLayers",
    "-c", "$CtxSize",
    "-np", "2",  # one slot keeps the agent prompt cached, the other serves detection
    "--jinja",
    "--reasoning", "auto", "--reasoning-format", "deepseek", "--no-reasoning-preserve",
    "--reasoning-budget", "512",
    # Hybrid LFM3 models resume cached prompts only from checkpoints; default spacing is too coarse.
    "--swa-full", "--checkpoint-min-step", "256",
    "--temp", "0"
)

# The robot needs vision: the projector turns camera frames into tokens the model can see.
if (Test-Path $MmprojPath) {
    $ArgsList += @("--mmproj", $MmprojPath)
} else {
    Write-Warning "Vision projector not found at $MmprojPath; the robot will be blind."
}

Write-Host "Starting llama-server on http://127.0.0.1:$Port ..."
& $ServerBin @ArgsList
exit $LASTEXITCODE
