# Upload the code (and optionally a dataset zip) from Windows to the EC2 instance.
#   .\aws\push.ps1 -Ip 13.233.10.20 -Key C:\Users\heman\.ssh\amazon-ml.pem
#   .\aws\push.ps1 -Ip 13.233.10.20 -Key ...\amazon-ml.pem -Dataset C:\Users\heman\Downloads\dataset.zip
param(
    [Parameter(Mandatory = $true)][string]$Ip,
    [Parameter(Mandatory = $true)][string]$Key,
    [string]$Dataset = "",
    [string]$User = "ubuntu"
)
$ErrorActionPreference = "Stop"
$proj = Split-Path -Parent $PSScriptRoot

# ssh refuses keys that other Windows users can read
icacls $Key /inheritance:r | Out-Null
icacls $Key /grant:r "$($env:USERNAME):(R)" | Out-Null

$tgz = Join-Path $env:TEMP "er_code.tgz"
tar -czf $tgz -C $proj src aws requirements.txt README.md
ssh -i $Key -o StrictHostKeyChecking=accept-new "$User@$Ip" "mkdir -p ~/er"
scp -i $Key $tgz "${User}@${Ip}:~/er_code.tgz"
ssh -i $Key "$User@$Ip" "tar -xzf ~/er_code.tgz -C ~/er && sed -i 's/\r$//' ~/er/aws/*.sh && echo code uploaded"

if ($Dataset -ne "") {
    Write-Host "uploading dataset (this can take a while on a slow connection)..."
    scp -i $Key $Dataset "${User}@${Ip}:~/dataset.zip"
}
Write-Host "next: ssh -i $Key $User@$Ip   then   cd ~/er && bash aws/setup_ec2.sh ~/dataset.zip"
