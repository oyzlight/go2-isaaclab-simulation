param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Python = 'python',
    [double[]]$Candidates = @(20,15,12.5),
    [int]$NumEnvs = 16,
    [double]$Duration = 18
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (!(Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw 'Missing checkpoint' }
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$jobDir = Join-Path $projectRoot ('logs/video_jobs/steeper_slopes_' + $stamp)
New-Item -ItemType Directory -Path $jobDir | Out-Null
$videoDir = Join-Path $projectRoot 'video'
$state = [ordered]@{phase='starting'; checkpoint=$Checkpoint; job_directory=$jobDir; selected=@(); active_pid=$null; active_case=$null; error=$null}
function Save-Status { $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $jobDir 'status.json') -Encoding UTF8 }
function Invoke-Slope([string]$direction,[double]$degrees,[bool]$record) {
    $angle = $degrees.ToString('0.##',[Globalization.CultureInfo]::InvariantCulture)
    $caseName = $direction + '_' + $angle + 'deg_' + $(if ($record) {'record'} else {'evaluate'})
    $output = Join-Path $jobDir ($caseName + '.json')
    $envCount = $(if ($record) {1} else {$NumEnvs})
    $arguments = @('-u','scripts/rsl_rl/evaluate_demo_slope.py','--checkpoint',('"' + $Checkpoint + '"'),
        '--direction',$direction,'--slope_degrees',$angle,'--num_envs',"$envCount",'--duration',"$Duration",
        '--seed','42','--headless','--device','cuda:0','--output',('"' + $output + '"'))
    if ($record) {
        $prefix = 'go2_' + $direction + '_' + $angle + 'deg_model_12493_' + $stamp
        $arguments += @('--enable_cameras','--rendering_mode','quality','--video_folder',('"' + $videoDir + '"'),'--video_name',$prefix)
    }
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -PassThru -RedirectStandardOutput (Join-Path $jobDir ($caseName + '.stdout.log')) `
        -RedirectStandardError (Join-Path $jobDir ($caseName + '.stderr.log'))
    $nativeHandle = $process.Handle
    $state.phase = $(if ($record) {'recording'} else {'evaluating'})
    $state.active_pid = $process.Id
    $state.active_case = $caseName
    Save-Status
    $process.WaitForExit()
    $process.Refresh()
    if ($null -eq $process.ExitCode -or $process.ExitCode -ne 0) { throw "Slope process failed: $caseName" }
    return (Get-Content -LiteralPath $output -Raw -Encoding UTF8 | ConvertFrom-Json)
}
Save-Status
try {
    foreach ($direction in @('uphill','downhill')) {
        $selected = $false
        foreach ($degrees in $Candidates) {
            $trial = Invoke-Slope $direction $degrees $false
            if ($trial.passed -ne $NumEnvs) { continue }
            $video = Invoke-Slope $direction $degrees $true
            if ($video.passed -ne 1 -or $video.whole_run_reset_count[0] -ne 0 -or $video.whole_run_max_tilt_deg[0] -ge 45) { continue }
            if (!(Test-Path -LiteralPath $video.video -PathType Leaf)) { throw 'Video missing' }
            $state.selected += [ordered]@{direction=$direction; degrees=$degrees; evaluation_passed=$trial.passed; evaluation_total=$NumEnvs; video=$video.video; trials=$trial.trials; recording_trial=$video.trials[0]}
            Save-Status
            $selected = $true
            break
        }
        if (!$selected) { throw "No tested steeper slope met the criterion for $direction" }
    }
    $state.phase='completed'; $state.active_pid=$null; $state.active_case=$null
    Save-Status
} catch {
    $state.phase='failed'; $state.error=$_.Exception.Message
    Save-Status
    throw
}
