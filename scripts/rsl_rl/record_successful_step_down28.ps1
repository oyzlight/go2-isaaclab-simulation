param(
    [Parameter(Mandatory=$true)][string]$Checkpoint,
    [string]$Python='python',
    [int[]]$Seeds=(43..62)
)
$ErrorActionPreference='Stop'
$projectRoot=(Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (!(Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw 'Missing checkpoint' }
$stamp=Get-Date -Format 'yyyyMMdd_HHmmss'
$jobDir=Join-Path $projectRoot ('logs/video_jobs/step_down28_success_' + $stamp)
New-Item -ItemType Directory -Path $jobDir | Out-Null
$videoDir=Join-Path $projectRoot 'video'
$state=[ordered]@{phase='starting';active_pid=$null;seed=$null;attempts=@();selected=$null;error=$null}
function Save-Status { $state | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $jobDir 'status.json') -Encoding UTF8 }
function Invoke-Trial([int]$seed,[bool]$record) {
    $caseName='seed' + $seed + '_' + $(if($record){'record'}else{'evaluate'})
    $report=Join-Path $jobDir ($caseName + '.json')
    $arguments=@('-u','scripts/rsl_rl/evaluate_demo_slope.py','--checkpoint',('"'+$Checkpoint+'"'),
        '--direction','downhill','--step_down_height','0.28','--forward_speed','0.5',
        '--num_envs','1','--duration','12','--seed',"$seed",'--headless','--device','cuda:0','--output',('"'+$report+'"'))
    if($record){
        $prefix='go2_step_down28_success_model_12493_seed' + $seed + '_' + $stamp
        $arguments+=@('--enable_cameras','--rendering_mode','quality','--video_folder',('"'+$videoDir+'"'),'--video_name',$prefix)
    }
    $process=Start-Process -FilePath $Python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $jobDir ($caseName+'.stdout.log')) -RedirectStandardError (Join-Path $jobDir ($caseName+'.stderr.log'))
    $nativeHandle=$process.Handle
    $state.phase=$(if($record){'recording'}else{'evaluating'});$state.active_pid=$process.Id;$state.seed=$seed
    Save-Status
    $process.WaitForExit();$process.Refresh()
    if($null -eq $process.ExitCode -or $process.ExitCode -ne 0){throw "Trial process failed: $caseName"}
    return (Get-Content -LiteralPath $report -Raw -Encoding UTF8 | ConvertFrom-Json)
}
Save-Status
try {
    foreach($seed in $Seeds){
        $trial=Invoke-Trial $seed $false
        $state.attempts += [ordered]@{seed=$seed;passed=$trial.passed}
        Save-Status
        if($trial.passed -ne 1){continue}
        $video=Invoke-Trial $seed $true
        if($video.passed -ne 1 -or $video.whole_run_reset_count[0] -ne 0 -or $video.whole_run_max_tilt_deg[0] -ge 45){continue}
        if(!(Test-Path -LiteralPath $video.video -PathType Leaf)){throw 'Successful video missing'}
        $state.selected=[ordered]@{seed=$seed;checkpoint=$Checkpoint;video=$video.video;passed=1;resets=0;max_tilt_deg=$video.whole_run_max_tilt_deg[0]}
        $state.phase='completed';$state.active_pid=$null
        Save-Status
        return
    }
    throw 'No full successful recording found in the requested seeds'
} catch {
    $state.phase='failed';$state.error=$_.Exception.Message
    Save-Status
    throw
}
