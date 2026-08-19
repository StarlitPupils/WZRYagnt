# Detached bootstrap: waits for port 3080 to free, then starts the dsh web server
# (same command the user originally ran: npx @deepseek-ai/dsh web)
$ErrorActionPreference = 'Continue'
$log = 'E:\WZRYagent\dsh-web-restart.log'
$errLog = 'E:\WZRYagent\dsh-web-restart.err.log'
$bin = 'C:\Users\15261\AppData\Local\npm-cache\_npx\1e7f6d9597241db0\node_modules\@deepseek-ai\dsh\lib\bin.js'

"$(Get-Date -Format o) bootstrap started, waiting for port 3080 to free" | Out-File $log -Append

$deadline = (Get-Date).AddMinutes(3)
$portFree = $false
while ((Get-Date) -lt $deadline) {
    $c = Get-NetTCPConnection -LocalPort 3080 -State Listen -ErrorAction SilentlyContinue
    if (-not $c) { $portFree = $true; break }
    Start-Sleep -Seconds 1
}
Start-Sleep -Seconds 2

if (-not $portFree) {
    "$(Get-Date -Format o) TIMEOUT: port 3080 still busy, aborting" | Out-File $log -Append
    exit 1
}

"$(Get-Date -Format o) port free, starting: node `"$bin`" web" | Out-File $log -Append
$p = Start-Process -FilePath 'E:\node\node.exe' -ArgumentList "`"$bin`"", 'web' -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError $errLog -PassThru
"$(Get-Date -Format o) dsh web started, pid $($p.Id)" | Out-File $log -Append
