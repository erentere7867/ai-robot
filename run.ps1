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
$LlmHealthUrl = "http://127.0.0.1:8080/health"

$env:PYTHONPATH = $ProjectRoot

function Test-LlmHealthy {
    try {
        return (Invoke-WebRequest -Uri $LlmHealthUrl -UseBasicParsing -TimeoutSec 2).StatusCode -eq 200
    } catch {
        return $false
    }
}

$LlmProc = $null
if ($StartLLM) {
    if (Test-LlmHealthy) {
        Write-Host "llama-server is already running at $LlmHealthUrl; reusing it."
    } else {
        $StartScript = Join-Path $ProjectRoot "scripts\start_llm.ps1"
        # Start-Process joins arguments with spaces and does not quote them; the project path may contain spaces.
        $LlmProc = Start-Process -FilePath "powershell" -PassThru -ArgumentList @(
            "-ExecutionPolicy", "Bypass", "-File", "`"$StartScript`""
        )
        $null = $LlmProc.Handle  # Without an open handle ExitCode reads as empty once the process exits.
        # Loading a model takes far longer than a fixed sleep on some machines; wait for /health instead.
        $Deadline = (Get-Date).AddSeconds(120)
        while ((Get-Date) -lt $Deadline -and -not $LlmProc.HasExited -and -not (Test-LlmHealthy)) {
            Start-Sleep -Milliseconds 500
        }
        if ($LlmProc.HasExited) {
            Write-Warning "llama-server exited during startup (exit code $($LlmProc.ExitCode))."
        } elseif (-not (Test-LlmHealthy)) {
            Write-Warning "llama-server is still loading; starting the robot anyway."
        }
    }
}

$ExitCode = 0
Push-Location $ProjectRoot
try {
    $CmdArgs = @("-m", "robot")
    if ($FaceWindow) {
        $CmdArgs += "--face-window"
    }
    if ($ExtraArgs) {
        $CmdArgs += $ExtraArgs
    }
    & $PyExe @CmdArgs
    $ExitCode = $LASTEXITCODE
} finally {
    Pop-Location
    if ($LlmProc -and -not $LlmProc.HasExited) {
        # Stop-Process would kill only the wrapper shell and orphan llama-server with its VRAM and port.
        $ErrorActionPreference = "Continue"
        & taskkill.exe /PID $LlmProc.Id /T /F | Out-Null
    }
}
exit $ExitCode
