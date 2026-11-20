# Try Sprocket from Windows: start the model and the GPU vision server if needed, then the simulated home and the live
# page (in WSL).
#   powershell -ExecutionPolicy Bypass -File tools\demo.ps1                  # then open http://127.0.0.1:8765
#   powershell -ExecutionPolicy Bypass -File tools\demo.ps1 --scene "two rooms"
# Extra arguments go to `python -m robot.interfaces.demo` (--case O3, --cameras all, --port 8766, ...). Ctrl+C stops it.
$repo = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "brain.ps1")
if (-not $?) { exit 1 }
& (Join-Path $PSScriptRoot "win\vision.ps1")
if (-not $?) { exit 1 }
$extra = ($args | ForEach-Object { if ("$_" -match "\s") { "'$_'" } else { "$_" } }) -join " "
"open http://127.0.0.1:8765 (Ctrl+C here stops the demo; the model keeps running)"
# env.sh first: the 3D apartment, the demo's first scene, is the robot's ROS software and needs ROS on the path
wsl --cd "$repo" -e bash -lc ". tools/linux/env.sh && exec python -m robot.interfaces.demo $extra"
