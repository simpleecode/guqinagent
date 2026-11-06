param(
    [string]$ScoreKey = 'SWkES6xH',
    [string]$RunName = 'teacher_gepa_smoke_SWkES6xH',
    [ValidateSet('production', 'v9', 'v15', 'v27')]
    [string]$SeedPrompt = 'v9',
    [int]$FeedbackPhrases = 4,
    [int]$ParetoPhrases = 3,
    [int]$MinibatchSize = 1,
    [int]$MaxMetricCalls = 12,
    [int]$Seed = 20261027,
    [int]$MaxRounds = 8,
    [double]$Temperature = 0.0,
    [switch]$PrepareOnly
)

$ErrorActionPreference = 'Stop'
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$Workspace = (Resolve-Path (Join-Path $ScriptDirectory '..\..\..')).Path
Set-Location -LiteralPath $Workspace

$keyWasInjected = $false
$DotEnv = Join-Path $Workspace 'guqinagent\.env'
if (-not $PrepareOnly -and -not $env:GLM_API_KEY -and -not (Test-Path -LiteralPath $DotEnv)) {
    $secure = Read-Host '请输入 GLM API Key（字符会隐藏，仅本次进程使用）' -AsSecureString
    $env:GLM_API_KEY = [System.Net.NetworkCredential]::new('', $secure).Password
    Remove-Variable secure
    $keyWasInjected = $true
}
# Python loads the endpoint and model from guqinagent/.env.  Setting defaults
# here would shadow that file before load_dotenv gets a chance to read it.

$env:PYTHONPATH = $Workspace + '\.gepa_packages;' + $Workspace + '\guqinagent;' + $Workspace + '\guqinagent\baseline\gepa\src'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$PythonExe = Join-Path $Workspace '.venv-gepa-run\Scripts\python.exe'
$Runner = Join-Path $Workspace 'guqinagent\baseline\scripts\run_teacher_gepa_iteration.py'
$RunDir = Join-Path $Workspace ('guqinagent\baseline\runs\' + $RunName)
$Arguments = @(
    $Runner,
    '--score-key', $ScoreKey,
    '--run-dir', $RunDir,
    '--seed-prompt', $SeedPrompt,
    '--feedback-phrases', $FeedbackPhrases,
    '--pareto-phrases', $ParetoPhrases,
    '--minibatch-size', $MinibatchSize,
    '--max-metric-calls', $MaxMetricCalls,
    '--seed', $Seed,
    '--max-rounds', $MaxRounds,
    '--temperature', $Temperature
)
if ($PrepareOnly) { $Arguments += '--prepare-only' }

Write-Host ('GEPA run directory: ' + $RunDir)
Write-Host ('Live artifacts will be written after every GEPA checkpoint; final HTML: ' + (Join-Path $RunDir 'optimization_report.html'))
try {
    & $PythonExe @Arguments
    exit $LASTEXITCODE
}
finally {
    if ($keyWasInjected) { Remove-Item Env:\GLM_API_KEY -ErrorAction SilentlyContinue }
}
