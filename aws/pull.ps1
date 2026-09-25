# Download outputs, logs and the tuned decision config from EC2 back to this repo.
#   .\aws\pull.ps1 -Ip 13.233.10.20 -Key C:\Users\heman\.ssh\amazon-ml.pem
param(
    [Parameter(Mandatory = $true)][string]$Ip,
    [Parameter(Mandatory = $true)][string]$Key,
    [string]$User = "ubuntu"
)
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force (Join-Path $proj "output"), (Join-Path $proj "logs"), (Join-Path $proj "work") | Out-Null
scp -i $Key "${User}@${Ip}:~/er/output/*.tsv" (Join-Path $proj "output")
scp -i $Key "${User}@${Ip}:~/er/logs/*.log" (Join-Path $proj "logs")
scp -i $Key "${User}@${Ip}:~/er/work/decision.json" (Join-Path $proj "work")
Write-Host "pulled into $proj\output, \logs, \work"
