param([switch]$RetryErrors)

$ErrorActionPreference = 'Stop'
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDirectory '..\..\..')).Path
Set-Location -LiteralPath $RepoRoot

if (-not $env:GLM_API_KEY) {
    throw 'GLM_API_KEY is not set in this PowerShell window.'
}
if (-not $env:GLM_BASE_URL) {
    $env:GLM_BASE_URL = 'https://open.bigmodel.cn/api/anthropic'
}
if (-not $env:GLM_MODEL) {
    $env:GLM_MODEL = 'glm-5.3'
}

$env:PYTHONPATH = $RepoRoot + '\.gepa_packages;' + $RepoRoot + '\guqinagent;' + $RepoRoot + '\guqinagent\baseline\gepa\src'
$PythonExe = Join-Path $RepoRoot '.venv-gepa-run\Scripts\python.exe'
$Arguments = @((Join-Path $RepoRoot 'guqinagent\baseline\scripts\run_v9_v27_prompt_iteration_baseline.py'), '--phase', 'diagnostic')
if ($RetryErrors) {
    $Arguments += '--retry-errors'
}

Write-Host 'Starting or resuming V9/V27 baseline: 10 samples, 20 paired calls.'
Write-Host ('Progress page: ' + (Join-Path $RepoRoot 'guqinagent\baseline\reports\v9_v27_diagnostic_baseline_comparison.html'))
& $PythonExe @Arguments
exit $LASTEXITCODE
