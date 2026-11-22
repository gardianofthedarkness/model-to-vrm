# Start the Model-to-VRM web app and a public tunnel to it, then print the phone link.
#   powershell -ExecutionPolicy Bypass -File E:\code\model-to-vrm\webapp\start_web.ps1
# Leave this window open: closing it stops the tunnel (the server keeps running in its own window).
# Tunnel: Cloudflare quick tunnel (E:\tools\cloudflared\cloudflared.exe, signed by Cloudflare); its address is new each
# time you run this script. Falls back to localhost.run over ssh if cloudflared is missing.
$py = 'E:\splat\.venv\Scripts\python.exe'; $here = Split-Path -Parent $MyInvocation.MyCommand.Path; $port = 8765
$cf = 'E:\tools\cloudflared\cloudflared.exe'
$up = $false; try { Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$port/" -TimeoutSec 2 | Out-Null; $up = $true } catch { if ($_.Exception.Response) { $up = $true } }
if (-not $up) { Start-Process -FilePath $py -ArgumentList "`"$here\server.py`" $port" -WorkingDirectory $here -WindowStyle Minimized; Start-Sleep 3 }
$key = (Get-Content 'E:\figures\runs\.webkey' -Raw).Trim()
Write-Host "Server on http://127.0.0.1:$port/?key=$key"
Write-Host "Opening the tunnel... wait for PHONE LINK below." -ForegroundColor Cyan
if (Test-Path $cf) {
    & $cf tunnel --no-autoupdate --url "http://127.0.0.1:$port" 2>&1 | ForEach-Object {
        $l = "$_"; if ($l -match '(https://[a-z0-9-]+\.trycloudflare\.com)') { Write-Host ""; Write-Host "PHONE LINK:  $($matches[1])/?key=$key" -ForegroundColor Green; Write-Host "" }
    }
} else {
    ssh -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=15 -o ExitOnForwardFailure=yes -R "80:127.0.0.1:$port" nokey@localhost.run 2>&1 | ForEach-Object {
        if ("$_" -match '(https://[a-z0-9]+\.[a-z0-9.]*lhr\.life)') { Write-Host ""; Write-Host "PHONE LINK:  $($matches[1])/?key=$key" -ForegroundColor Green; Write-Host "" }
    }
}
