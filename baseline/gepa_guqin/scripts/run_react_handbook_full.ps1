# Full sealed-test rollout for the production single-stage ReAct prompt plus
# the fixed complex-fingering handbook. The run is resumable and evaluates
# automatically after all scores finish.
param(
    [string]$PythonPath = (Join-Path $PSScriptRoot '..\..\..\.venv-gepa\Scripts\python.exe'),
    [string]$Model = 'glm-5.3',
    [ValidateRange(1, 64)] [int]$MaxRounds = 8,
    [ValidateRange(1, 8)] [int]$Attempts = 2,
    [ValidateRange(0.0, 2.0)] [double]$SampleTemperature = 0.25,
    [ValidateRange(0.0, 60.0)] [double]$MinInterval = 0.5,
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')]
    [string]$RunName = 'react_handbook_full_test43',
    [switch]$SkipEvaluation
)

$ErrorActionPreference = 'Stop'
$repo = (Resolve-Path (Join-Path $PSScriptRoot '../../..')).Path
$inputDir = Join-Path $repo 'train/eval_inputs_v2_text_protocol/renderfix_20261007'
$runtimeInput = Join-Path $inputDir 'test_runtime.jsonl'
$referenceInput = Join-Path $inputDir 'evaluation_pairs_test.jsonl'
$sourceInput = Join-Path $repo 'ABC_J/agent_training/renderfix_train_test_20261007/pitch_eligible_two_or_half/inferred_trajectories_test.jsonl'
$envFile = Join-Path $repo '.env'
$runDir = Join-Path $repo "baseline/gepa_guqin/runs/$RunName"
$predictionPath = Join-Path $runDir 'predictions_test.jsonl'
$rolloutLog = Join-Path $runDir 'rollout.log'
$evaluationLog = Join-Path $runDir 'evaluation.log'
$configPath = Join-Path $runDir 'launch_config.json'
$pidPath = Join-Path $runDir 'run.pid'

foreach ($path in @($PythonPath, $sourceInput, $envFile)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Missing required file: $path"
    }
}
$envNames = Get-Content -LiteralPath $envFile |
    Where-Object { $_ -match '^\s*(GLM_API_KEY|GLM_BASE_URL|GLM_MODEL)\s*=' } |
    ForEach-Object { ($_ -split '=', 2)[0].Trim() }
foreach ($name in @('GLM_API_KEY', 'GLM_BASE_URL', 'GLM_MODEL')) {
    if ($envNames -notcontains $name) { throw "Missing $name in $envFile" }
}

$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

if (Test-Path -LiteralPath $pidPath -PathType Leaf) {
    $existingProcessId = 0
    $existingPidText = (Get-Content -LiteralPath $pidPath -Raw -Encoding UTF8).Trim()
    if ([int]::TryParse($existingPidText, [ref]$existingProcessId) -and
        (Get-Process -Id $existingProcessId -ErrorAction SilentlyContinue)) {
        throw "Run '$RunName' is already active as process $existingProcessId."
    }
    Remove-Item -LiteralPath $pidPath -Force
}

if (-not (Test-Path -LiteralPath $runtimeInput -PathType Leaf) -or
    -not (Test-Path -LiteralPath $referenceInput -PathType Leaf)) {
    Write-Host 'Exporting sealed public runtime input and private evaluation pairs...'
    & $PythonPath -X utf8 -m baseline.gepa_guqin.export_eval_inputs `
        --source $sourceInput --out-dir $inputDir
    if ($LASTEXITCODE -ne 0) { throw 'Sealed input export failed.' }
}

$config = [ordered]@{
    schema_version = 'react-handbook-launch-1.0'
    run_name = $RunName
    model = $Model
    prompt = 'production_seed'
    handbook = $true
    input = 'train/eval_inputs_v2_text_protocol/renderfix_20261007/test_runtime.jsonl'
    reference = 'train/eval_inputs_v2_text_protocol/renderfix_20261007/evaluation_pairs_test.jsonl'
    max_rounds = $MaxRounds
    attempts = $Attempts
    sample_temperature = $SampleTemperature
    min_interval = $MinInterval
}
$configJson = $config | ConvertTo-Json -Depth 4
if (Test-Path -LiteralPath $configPath -PathType Leaf) {
    $prior = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
    foreach ($field in @('model', 'prompt', 'handbook', 'input', 'reference',
                          'max_rounds', 'attempts', 'sample_temperature', 'min_interval')) {
        if ([string]$prior.$field -ne [string]$config.$field) {
            throw "Existing run uses a different '$field'; choose a new -RunName."
        }
    }
} else {
    [IO.File]::WriteAllText($configPath, $configJson + [Environment]::NewLine,
                            [Text.UTF8Encoding]::new($false))
}

[IO.File]::WriteAllText($pidPath, "$PID`n", [Text.UTF8Encoding]::new($false))
Push-Location -LiteralPath $repo
try {
    Write-Host "Starting ReAct + Handbook full test: $RunName"
    Write-Host "Predictions: $predictionPath"
    Write-Host "Live log: Get-Content -Wait -Tail 20 '$rolloutLog'"
    # Deliberately omit --prompt-json and --no-handbook: this selects the
    # production seed prompt plus the fixed handbook.
    & $PythonPath -X utf8 -m baseline.gepa_guqin.run_final_eval `
        --input $runtimeInput `
        --output $predictionPath `
        --model $Model `
        --max-rounds $MaxRounds `
        --attempts $Attempts `
        --sample-temperature $SampleTemperature `
        --min-interval $MinInterval `
        --resume 2>&1 | Tee-Object -FilePath $rolloutLog -Append
    $rolloutExit = $LASTEXITCODE
    if ($rolloutExit -ne 0) { throw "ReAct rollout failed with exit code $rolloutExit." }

    if (-not $SkipEvaluation) {
        Write-Host 'Rollout complete; starting standard event-level evaluation.'
        & $PythonPath -X utf8 -m evaluation.run_eval `
            --pred $predictionPath `
            --reference $referenceInput `
            --experiment $RunName `
            --model "$Model+handbook" 2>&1 |
            Tee-Object -FilePath $evaluationLog -Append
        $evaluationExit = $LASTEXITCODE
        if ($evaluationExit -ne 0) {
            throw "Evaluation failed with exit code $evaluationExit."
        }
    }
} finally {
    Pop-Location
    if (Test-Path -LiteralPath $pidPath) {
        Remove-Item -LiteralPath $pidPath -Force
    }
}
