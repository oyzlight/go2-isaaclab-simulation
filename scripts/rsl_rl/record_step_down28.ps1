param(
    [Parameter(Mandatory=$true)][string]$Checkpoint19491,
    [Parameter(Mandatory=$true)][string]$Checkpoint12493,
    [string]$Python = 'python',
    [double[]]$Speeds = @(0.8,0.3,0.5),
    [switch]$Only12493
)
$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
foreach ($checkpoint in @($Checkpoint19491,$Checkpoint12493)) {
    if (!(Test-Path -LiteralPath $checkpoint -PathType Leaf)) { throw 'Missing candidate checkpoint' }
}
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$jobDir = Join-Path $projectRoot ('logs/video_jobs/step_down28_' + $stamp)
New-Item -ItemType Directory -Path $jobDir | Out-Null
$videoDir = Join-Path $projectRoot 'video'
$state = [ordered]@{phase='starting'; active_pid=$null; active_case=$null; candidates=@(); selected=$null; error=$null}
function Save-Status { $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $jobDir 'status.json') -Encoding UTF8 }
function Invoke-Candidate([string]$name,[string]$checkpoint,[bool]$record,[double]$speed) {
    $caseName = $name + '_' + $(if ($record) {'record'} else {'evaluate'})
    $output = Join-Path $jobDir ($caseName + '.json')
    $numEnvs = $(if ($record) {1} else {16})
    $arguments = @('-u','scripts/rsl_rl/evaluate_demo_slope.py','--checkpoint',('"' + $checkpoint + '"'),
        '--direction','downhill','--step_down_height','0.28','--num_envs',"$numEnvs",'--duration','12',
        '--seed','42','--headless','--device','cuda:0','--output',('"' + $output + '"'),
        '--forward_speed',$speed.ToString('0.##',[Globalization.CultureInfo]::InvariantCulture))
    if ($record) {
        $prefix = 'go2_step_down28_' + $name + '_' + $stamp
        $arguments += @('--enable_cameras','--rendering_mode','quality','--video_folder',('"' + $videoDir + '"'),'--video_name',$prefix)
    }
    $process = Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden `
        -PassThru -RedirectStandardOutput (Join-Path $jobDir ($caseName + '.stdout.log')) `
        -RedirectStandardError (Join-Path $jobDir ($caseName + '.stderr.log'))
    $nativeHandle = $process.Handle
    $state.phase=$(if ($record) {'recording'} else {'evaluating'})
    $state.active_pid=$process.Id; $state.active_case=$caseName
    Save-Status
    $process.WaitForExit(); $process.Refresh()
    if ($null -eq $process.ExitCode -or $process.ExitCode -ne 0) { throw "Evaluation process failed: $caseName" }
    return (Get-Content -LiteralPath $output -Raw -Encoding UTF8 | ConvertFrom-Json)
}
Save-Status
try {
    $best=$null
    foreach ($candidate in @(@('model_19491',$Checkpoint19491),@('model_12493',$Checkpoint12493))) {
        if ($Only12493 -and $candidate[0] -ne 'model_12493') { continue }
        foreach ($speed in $Speeds) {
            $label=$candidate[0] + '_v' + $speed.ToString('0.##',[Globalization.CultureInfo]::InvariantCulture)
            $trial=Invoke-Candidate $label $candidate[1] $false $speed
            $entry=[ordered]@{name=$label;model=$candidate[0];speed=$speed;checkpoint=$candidate[1];passed=$trial.passed;total=16}
            $state.candidates += $entry
            if ($null -eq $best -or $trial.passed -gt $best.passed) { $best=$entry }
            Save-Status
        }
    }
    $record=Invoke-Candidate $best.name $best.checkpoint $true $best.speed
    if (!(Test-Path -LiteralPath $record.video -PathType Leaf)) { throw 'Recording was not produced' }
    $stable=($record.passed -eq 1 -and $record.whole_run_reset_count[0] -eq 0 -and $record.whole_run_max_tilt_deg[0] -lt 45)
    $state.selected=[ordered]@{name=$best.name;model=$best.model;speed=$best.speed;checkpoint=$best.checkpoint;batch_passed=$best.passed;batch_total=16;recording_stable=$stable;video=$record.video}
    $state.phase='completed';$state.active_pid=$null;$state.active_case=$null
    Save-Status
} catch {
    $state.phase='failed';$state.error=$_.Exception.Message
    Save-Status
    throw
}
