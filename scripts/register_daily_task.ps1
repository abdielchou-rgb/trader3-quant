# 注册 3号交易员 每日定时任务（工作日 17:30 收盘后）
# 用法: powershell -ExecutionPolicy Bypass -File scripts\register_daily_task.ps1
# 删除: Unregister-ScheduledTask -TaskName "Trader3Daily" -Confirm:$false

$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = (Get-Command py).Source

$action = New-ScheduledTaskAction -Execute $python `
    -Argument "-3.11 -W ignore tools\daily_routine.py" `
    -WorkingDirectory $project

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At 17:30

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive

Register-ScheduledTask -TaskName "Trader3Daily" `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "3号交易员: 数据增量更新+触发扫描+纸面交易+日报推送(工作日17:30)" `
    -Force | Out-Null

Write-Host "[OK] 计划任务已注册: Trader3Daily (工作日 17:30)"
Write-Host "     项目目录: $project"
Write-Host "     手动试跑: Start-ScheduledTask -TaskName Trader3Daily"
Write-Host "     查看状态: Get-ScheduledTask -TaskName Trader3Daily | Get-ScheduledTaskInfo"
