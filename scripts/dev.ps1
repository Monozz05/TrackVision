<#
Start the read-only results API and the frontend together, from the repository root:

    .\scripts\dev.ps1
    .\scripts\dev.ps1 -BackendPort 8010 -FrontendPort 3001

Stop with Ctrl+C. If a port is taken, the script names what holds it and stops; it never touches another process.
#>
param([int]$BackendPort = 8000, [int]$FrontendPort = 3000)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'

$busy = $false
foreach ($port in @($BackendPort, $FrontendPort)) {
    $listener = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($listener) {
        $owner = (Get-CimInstance Win32_Process -Filter "ProcessId=$($listener.OwningProcess)").CommandLine
        Write-Host "Port $port is in use by process $($listener.OwningProcess): $owner"
        $busy = $true
    }
}
if ($busy) {
    Write-Host 'Stop that process yourself, or choose free ports, for example: .\scripts\dev.ps1 -BackendPort 8010 -FrontendPort 3001'
    exit 1
}

$env:TRACKVISION_CORS_ORIGINS = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"
$env:NEXT_PUBLIC_API_BASE_URL = "http://127.0.0.1:$BackendPort/api"
$env:PYTHONDONTWRITEBYTECODE = '1'

$backend = Start-Process -FilePath $python -WorkingDirectory $root -NoNewWindow -PassThru -ArgumentList @(
    '-m', 'uvicorn', 'trackvision.platform.app:app', '--host', '127.0.0.1', '--port', $BackendPort)
try {
    Write-Host "Backend:  http://127.0.0.1:$BackendPort/api/health"
    Write-Host "Frontend: http://localhost:$FrontendPort"
    Set-Location (Join-Path $root 'web')
    pnpm dev --port $FrontendPort
} finally {
    Set-Location $root
    taskkill /T /F /PID $backend.Id | Out-Null
}
