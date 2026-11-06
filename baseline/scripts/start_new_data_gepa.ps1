param(
    [ValidateSet('production')]
    [string]$SeedPrompt = 'production',
    [ValidatePattern('^[A-Z]$')]
    [string]$Series = 'M',
    [string]$RunName = '',
    [int]$MaxMetricCalls = 60,
    [int]$MaxProposals = 6,
    [int]$MinibatchSize = 2,
    [int]$ConfirmationRepeats = 2,
    [switch]$SkipConfirmation,
    [switch]$CheckConfig,
    [switch]$CheckApi,
    [string]$ScoreKey = '',
    [int]$FeedbackPhrases = 2,
    [int]$ParetoPhrases = 1,
    [ValidateSet('exact', 'features')]
    [string]$FeedbackDetail = 'exact',
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$Workspace = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
Set-Location -LiteralPath $Workspace
$env:PYTHONPATH = "$Workspace\.gepa_packages;$Workspace\guqinagent;$Workspace\guqinagent\baseline\gepa\src"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$PythonExe = Join-Path $Workspace '.venv-gepa-run\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "找不到训练Python环境：$PythonExe"
}
if (-not $RunName) {
    if ($ScoreKey) { $RunName = $Series + '_single_' + $ScoreKey + '_from_original' }
    else { $RunName = $Series + '_whole10_from_original' }
}
$Runner = Join-Path $PSScriptRoot 'run_auto_prompt_training.py'
$Arguments = @($Runner, '--run-name', $RunName,
    '--seed-prompt', $SeedPrompt, '--max-metric-calls', $MaxMetricCalls,
    '--max-proposals', $MaxProposals, '--minibatch-size', $MinibatchSize,
    '--confirmation-repeats', $ConfirmationRepeats, '--feedback-detail', $FeedbackDetail, '--series', $Series)
if ($ScoreKey) { $Arguments += @('--score-key', $ScoreKey, '--feedback-phrases', $FeedbackPhrases, '--pareto-phrases', $ParetoPhrases) }
if ($PrepareOnly) { $Arguments += '--prepare-only' }
if ($CheckConfig) { $Arguments += '--check-config' }
if ($CheckApi) { $Arguments += '--check-api' }
if ($SkipConfirmation) { $Arguments += '--skip-confirmation' }
& $PythonExe @Arguments
exit $LASTEXITCODE
