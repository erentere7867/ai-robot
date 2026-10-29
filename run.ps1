param(
    [switch]$FaceWindow,
    [switch]$StartLLM,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy = Join-Path $env:USERPROFILE "robot-runtime\venv\Scripts\python.exe"
$PyExe = if (Test-Path $VenvPy) { $VenvPy } else { "python" }

$env:PYTHONPATH = $ProjectRoot

$LlmProc = $null
if ($StartLLM) {
    $StartScript = Join-Path $ProjectRoot "scripts\start_llm.ps1"
    $LlmProc = Start-Process -FilePath "powershell" -ArgumentList @("-ExecutionPolicy", "Bypass", "-File", $StartScript) -PassThru
    Start-Sleep -Seconds 3
}

try {
    $CmdArgs = @("-m", "robot")
    if ($FaceWindow) {
        $CmdArgs += "--face-window"
    }
    if ($ExtraArgs) {
        $CmdArgs += $ExtraArgs
    }
    Push-Location $ProjectRoot
    & $PyExe @CmdArgs
} finally {
    Pop-Location -ErrorAction SilentlyContinue
    if ($LlmProc -and -not $LlmProc.HasExited) {
        Stop-Process -Id $LlmProc.Id -Force
    }
}
