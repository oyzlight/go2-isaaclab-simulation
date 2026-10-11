param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [int]$VideoLength = 600,
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (!(Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw 'Missing demonstration checkpoint' }
if ($VideoLength -lt 1) { throw 'VideoLength must be positive' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$jobDir = Join-Path $projectRoot ('logs/video_jobs/terrain_demos_' + $stamp)
New-Item -ItemType Directory -Path $jobDir | Out-Null
$videoDir = Join-Path $projectRoot 'video'
$modelTag = [IO.Path]::GetFileNameWithoutExtension($Checkpoint)
$state = [ordered]@{
    phase='starting'; checkpoint=$Checkpoint; video_directory=$videoDir
    video_length_steps=$VideoLength; started_at=(Get-Date -Format o)
    active_case=$null; active_pid=$null; completed=@(); error=$null
}
function Save-Status {
    $state | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $jobDir 'status.json') -Encoding UTF8
}
$cases = @(
    @('flat', 'Flat'), @('rubble', 'Rubble'), @('uphill_10deg', 'Uphill'),
    @('downhill_10deg', 'Downhill'), @('stairs_up_8cm', 'Stairs-Up8cm'), @('stairs_down_8cm', 'Stairs-Down8cm')
)
try {
    foreach ($case in $cases) {
        $prefix = 'go2_' + $case[0] + '_' + $modelTag + '_' + $stamp
        $arguments = @('-u', 'scripts/rsl_rl/play.py', '--task', ('Go2-Demo-' + $case[1] + '-Play-v0'),
            '--num_envs', '1', '--device', 'cuda:0', '--rendering_mode', 'quality', '--headless',
            '--video', '--video_length', "$VideoLength", '--video_folder', ('"' + $videoDir + '"'),
            '--video_name', $prefix, '--checkpoint', ('"' + $Checkpoint + '"'))
        $stdout = Join-Path $jobDir ($case[0] + '.stdout.log')
        $stderr = Join-Path $jobDir ($case[0] + '.stderr.log')
        $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot `
            -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
        $nativeHandle = $process.Handle
        $state.phase = 'recording'
        $state.active_case = $case[0]
        $state.active_pid = $process.Id
        Save-Status
        $process.WaitForExit()
        $process.Refresh()
        if ($null -eq $process.ExitCode -or $process.ExitCode -ne 0) {
            throw "Recording $($case[0]) failed; inspect $stderr"
        }
        $videoPath = Join-Path $videoDir ($prefix + '-step-0.mp4')
        if (!(Test-Path -LiteralPath $videoPath -PathType Leaf)) { throw "Video not produced: $videoPath" }
        $state.completed += [ordered]@{ case=$case[0]; task=('Go2-Demo-' + $case[1] + '-Play-v0'); path=$videoPath }
        Save-Status
    }
    $state.phase = 'completed'
    $state.active_case = $null
    $state.active_pid = $null
    Save-Status
} catch {
    $state.phase = 'failed'
    $state.error = $_.Exception.Message
    Save-Status
    throw
}
