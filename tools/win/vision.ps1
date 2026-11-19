# Start the GPU vision server (YOLOE + depth on DirectML) for the robot stack in WSL, on 127.0.0.1:8090.
# Needs the venv C:\Users\<you>\sprocket-win (onnxruntime-directml, numpy) and its models folder.
$python = Join-Path $env:USERPROFILE "sprocket-win\Scripts\python.exe"
$models = Join-Path $env:USERPROFILE "sprocket-win\models"
$log = Join-Path $env:TEMP "sprocket-vision.log"
& $python (Join-Path $PSScriptRoot "vision_server.py") --models $models *> $log
