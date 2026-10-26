param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Python = 'python',
    [int]$VideoLength = 600
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (!(Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw 'Missing demonstration checkpoint' }
if ($VideoLength -lt 1) { throw 'VideoLength must be positive' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$jobDir = Join-Path $projectRoot ('logs/video_jobs/rubble_dense_' + $stamp)
New-Item -ItemType Directory -Path $jobDir | Out-Null
$videoDir = Join-Path $projectRoot 'video'
$prefix = 'go2_rubble_dense_' + [IO.Path]::GetFileNameWithoutExtension($Checkpoint) + '_' + $stamp
$state = [ordered]@{
    phase='starting'; checkpoint=$Checkpoint; task='Go2-Demo-Rubble-Play-v0'
    rocks=600; rock_height_m=@(0.04,0.08); seed=42
    video_length_steps=$VideoLength; video=(Join-Path $videoDir ($prefix + '-step-0.mp4'))
    started_at=(Get-Date -Format o); active_pid=$null; error=$null
}
function Save-Status {
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $jobDir 'status.json') -Encoding UTF8
}
Save-Status
try {
    $arguments = @('-u', 'scripts/rsl_rl/play.py', '--task', $state.task,
        '--num_envs', '1', '--device', 'cuda:0', '--rendering_mode', 'quality', '--headless',
        '--seed', '42', '--video', '--video_length', "$VideoLength",
        '--video_folder', ('"' + $videoDir + '"'), '--video_name', $prefix,
        '--checkpoint', ('"' + $Checkpoint + '"'))
    $recording = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $jobDir 'play.stdout.log') `
        -RedirectStandardError (Join-Path $jobDir 'play.stderr.log')
    $nativeHandle = $recording.Handle
    $state.active_pid = $recording.Id
    $state.phase = 'recording'
    Save-Status
    $recording.WaitForExit()
    $recording.Refresh()
    if ($null -eq $recording.ExitCode -or $recording.ExitCode -ne 0) { throw 'Playback failed; inspect job logs' }
    if (!(Test-Path -LiteralPath $state.video -PathType Leaf)) { throw 'Recording was not produced' }
    $state.phase = 'completed'
    $state.active_pid = $null
    Save-Status
} catch {
    $state.phase = 'failed'
    $state.error = $_.Exception.Message
    Save-Status
    throw
}
