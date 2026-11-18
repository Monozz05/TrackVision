<#
Install YOLO11 in its own environment and run it over every real video, from the repository root:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_yolo.ps1

Add -InstallOnly to stop once the environment and the weights are ready, without running any detection.

The environment lives outside the repository, so the research `.venv` is not touched. This downloads PyTorch for NVIDIA
GPUs, Ultralytics and its dependencies, and the YOLO11 weights from Ultralytics. An interrupted run can be started again;
finished scenes are skipped. Add -Device 0 to name the GPU, or -Device cpu to run without one, which takes many hours.
#>
param(
    [string]$Environment = "$env:USERPROFILE\tv-envs\yolo-py312",
    [string]$Model = 'yolo11m',
    [string]$Device = '',
    [switch]$InstallOnly
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $Environment 'Scripts\python.exe'
$weightsDir = Join-Path $Environment 'weights'
$weights = Join-Path $weightsDir "$Model.pt"

if (-not (Test-Path $python)) { py -3.12 -m venv $Environment }
# PyPI's Windows build of PyTorch has no GPU support, so PyTorch comes from its own index first.
& $python -m pip install --disable-pip-version-check --timeout 120 --retries 10 torch torchvision --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { throw 'Installing PyTorch failed.' }
& $python -m pip install --disable-pip-version-check ultralytics
if ($LASTEXITCODE -ne 0) { throw 'Installing ultralytics failed.' }

New-Item -ItemType Directory -Force $weightsDir | Out-Null
$env:YOLO_CONFIG_DIR = Join-Path $weightsDir 'ultralytics-config'

# Ultralytics downloads the weights to these paths on first use. The larger model is there for very small people, as in ETH.
foreach ($name in $Model, 'yolo11x') {
    & $python -c "import sys, torch; from ultralytics import YOLO; YOLO(sys.argv[1]); print(sys.argv[1], 'ready. GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none found')" (Join-Path $weightsDir "$name.pt")
    if ($LASTEXITCODE -ne 0) { throw 'Loading the YOLO11 weights failed.' }
}
if ($InstallOnly) { exit 0 }

$arguments = @('--weights', $weights)
if ($Device) { $arguments += @('--device', $Device) }
foreach ($script in 'yolo_detect_mot17.py', 'yolo_detect_videos.py') {
    & $python (Join-Path $root "scripts\$script") @arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
exit 0
