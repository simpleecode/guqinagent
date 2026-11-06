param(
    [ValidateSet('diagnostic', 'check')]
    [string]$Phase = 'diagnostic',
    [switch]$PrepareOnly,
    [switch]$RetryErrors
)

$ErrorActionPreference = 'Stop'
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDirectory '..\..\..')).Path
Set-Location -LiteralPath $RepoRoot
if (-not $PrepareOnly -and -not $env:GLM_API_KEY) { throw 'GLM_API_KEY is not set in this process.' }
if (-not $env:GLM_BASE_URL) { $env:GLM_BASE_URL = 'https://open.bigmodel.cn/api/anthropic' }
if (-not $env:GLM_MODEL) { $env:GLM_MODEL = 'glm-5.3' }
$env:PYTHONPATH = $RepoRoot + '\.gepa_packages;' + $RepoRoot + '\guqinagent;' + $RepoRoot + '\guqinagent\baseline\gepa\src'
$PythonExe = Join-Path $RepoRoot '.venv-gepa-run\Scripts\python.exe'
$Arguments = @(
    (Join-Path $RepoRoot 'guqinagent\baseline\scripts\run_v9_v27_prompt_iteration_baseline.py'),
    '--phase', $Phase, '--parent', 'v9', '--candidate-id', 'v34',
    '--candidate-prompt', (Join-Path $RepoRoot 'guqinagent\baseline\prompts\single_stage_v9_compound_coverage_v34_addendum.txt'),
    '--generation-manifest', (Join-Path $RepoRoot 'guqinagent\baseline\results\v34_generation_manifest.json')
)
if ($PrepareOnly) { $Arguments += '--prepare-only' }
if ($RetryErrors) { $Arguments += '--retry-errors' }
Write-Host "Starting or resuming V9/V34 $Phase comparison."
& $PythonExe @Arguments
exit $LASTEXITCODE
