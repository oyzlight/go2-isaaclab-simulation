param(
    [string]$Checkpoint = 'D:\RobotProject\go2_demo\logs\rsl_rl\go2_velocity_stairs_height_transfer\2026-10-01_10-58-57_stairs_lateral_speed\model_12493.pt',
    [int]$NumEnvs = 1024,
    [int]$Iterations = 5000,
    [string]$Python = 'C:\Users\26304\.conda\envs\env_isaaclab\python.exe',
    [string]$Task = 'Go2-High-Step-28cm-v0',
    [string]$EvaluationTask = 'Go2-High-Step-28cm-Play-v0',
    [string]$Experiment = 'go2_high_step_28cm',
    [string]$RunPrefix = '28cm_from12493',
    [double]$ActionNoiseStd = 0.0,
    [switch]$LoadOptimizer
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
if (!(Test-Path -LiteralPath $Checkpoint -PathType Leaf)) { throw "Missing checkpoint: $Checkpoint" }
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
if ($NumEnvs -lt 1 -or $Iterations -lt 1) { throw 'Environment/iteration counts must be positive' }
$runTag = $RunPrefix + '_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
$jobDir = Join-Path $projectRoot ('logs/high_step_jobs/' + $runTag)
New-Item -ItemType Directory -Path $jobDir | Out-Null
$statusPath = Join-Path $jobDir 'status.json'
$state = [ordered]@{
    phase = 'starting'; started_at = (Get-Date -Format o); updated_at = ''
    checkpoint = $Checkpoint; num_envs = $NumEnvs; additional_iterations = $Iterations
    task = $Task; evaluation_task = $EvaluationTask; action_noise_std_override = $ActionNoiseStd
    load_optimizer = [bool]$LoadOptimizer
    run_name = $runTag; training_pid = $null; evaluation_pid = $null
    training_stdout = (Join-Path $jobDir 'train.stdout.log')
    training_stderr = (Join-Path $jobDir 'train.stderr.log')
    training_directory = $null; final_checkpoint = $null; evaluation_result = $null
    error = $null
}

function Save-Status {
    $state.updated_at = Get-Date -Format o
    $state | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

try {
    $trainArgs = @('-u', 'scripts/rsl_rl/train.py', '--task', $Task,
        '--headless', '--num_envs', "$NumEnvs", '--resume',
        '--checkpoint', ('"' + $Checkpoint + '"'), '--max_iterations', "$Iterations", '--run_name', $runTag)
    if (!$LoadOptimizer) { $trainArgs += '--load_policy_only' }
    if ($ActionNoiseStd -gt 0) {
        $trainArgs += @('--reset_action_noise_std', $ActionNoiseStd.ToString([Globalization.CultureInfo]::InvariantCulture))
    }
    $trainProcess = Start-Process -FilePath $Python -ArgumentList $trainArgs -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $state.training_stdout `
        -RedirectStandardError $state.training_stderr
    $state.phase = 'training'
    $state.training_pid = $trainProcess.Id
    # Acquire and retain the native handle BEFORE a long wait. Start-Process
    # can otherwise lose the exited process handle and return a null ExitCode.
    $trainNativeHandle = $trainProcess.Handle
    Save-Status
    $trainProcess.WaitForExit()
    $trainProcess.Refresh()
    $state.training_exit_code = $trainProcess.ExitCode
    if ($null -eq $state.training_exit_code) { throw 'Cannot retrieve training exit code' }
    if ($state.training_exit_code -ne 0) { throw "Training exited with code $($state.training_exit_code)" }

    $experimentDir = Join-Path $projectRoot ('logs/rsl_rl/' + $Experiment)
    $runDir = Get-ChildItem -LiteralPath $experimentDir -Directory |
        Where-Object { $_.Name.EndsWith('_' + $runTag) } | Select-Object -First 1
    if (!$runDir) { throw 'Cannot locate completed training directory' }
    $finalModel = Get-ChildItem -LiteralPath $runDir.FullName -File -Filter 'model_*.pt' |
        Sort-Object { [int]($_.BaseName -replace '^model_', '') } -Descending | Select-Object -First 1
    if (!$finalModel) { throw 'Training produced no checkpoint' }
    $state.training_directory = $runDir.FullName
    $state.final_checkpoint = $finalModel.FullName

    # A trained model is not automatically a successful 28 cm climbing model.
    # Evaluate fixed 28 cm geometry, deterministic actions, 32 lanes x 4 attempts.
    $resultPath = Join-Path $runDir.FullName 'evaluation_28cm.json'
    $evalArgs = @('-u', 'scripts/rsl_rl/evaluate_high_step.py', '--headless',
        '--task', $EvaluationTask,
        '--checkpoint', ('"' + $finalModel.FullName + '"'), '--step_height', '0.28',
        '--num_envs', '32', '--episodes_per_env', '4', '--output', ('"' + $resultPath + '"'))
    $evalProcess = Start-Process -FilePath $Python -ArgumentList $evalArgs -WorkingDirectory $projectRoot `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $jobDir 'eval.stdout.log') `
        -RedirectStandardError (Join-Path $jobDir 'eval.stderr.log')
    $state.phase = 'evaluating'
    $state.evaluation_pid = $evalProcess.Id
    $evalNativeHandle = $evalProcess.Handle
    Save-Status
    $evalProcess.WaitForExit()
    $evalProcess.Refresh()
    $state.evaluation_exit_code = $evalProcess.ExitCode
    if ($null -eq $state.evaluation_exit_code) { throw 'Cannot retrieve evaluation exit code' }
    if ($state.evaluation_exit_code -ne 0) { throw "Evaluation exited with code $($state.evaluation_exit_code)" }
    $state.evaluation_result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
    $state.phase = 'completed'
    Save-Status
} catch {
    $state.phase = 'failed'
    $state.error = $_.Exception.Message
    Save-Status
    throw
}
