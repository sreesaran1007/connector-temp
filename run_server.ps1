<#
.SYNOPSIS
  Starts the local analysis server that the app talks to.

.DESCRIPTION
  Listens on port 4445 (change with -Port). Without -Lan only this PC can
  connect, which is what a Cloudflare tunnel on the same PC needs.

.EXAMPLE
  .\run_server.ps1            # only this PC can connect (tunnel, Chrome, Android emulator)
  .\run_server.ps1 -Lan       # phones on the same Wi-Fi can connect too
#>
param(
    [switch]$Lan,
    [int]$Port = 4445
)

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "No virtual environment found. Create it first - see requirements.txt."
    exit 1
}

$bind = if ($Lan) { '0.0.0.0' } else { '127.0.0.1' }
if ($Lan) {
    Write-Host "Reachable from other devices on your network at:" -ForegroundColor Cyan
    Get-NetIPAddress -AddressFamily IPv4 |
        Where-Object { $_.IPAddress -notmatch '^(127|169\.254)\.' -and $_.PrefixOrigin -ne 'WellKnown' } |
        ForEach-Object { Write-Host ("  http://{0}:{1}" -f $_.IPAddress, $Port) }
    Write-Host "Windows may ask to allow Python through the firewall: allow it on Private networks." -ForegroundColor Yellow
    Write-Host "Anyone on the same network can use this server while it runs. Stop it with Ctrl+C." -ForegroundColor Yellow
}

Set-Location $PSScriptRoot
& $python -m uvicorn server:app --host $bind --port $Port

