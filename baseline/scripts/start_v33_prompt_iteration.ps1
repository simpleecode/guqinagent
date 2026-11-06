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

if (-not $PrepareOnly -and -not $env:GLM_API_KEY) {
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
$Runner = Join-Path $RepoRoot 'guqinagent\baseline\scripts\run_v9_v27_prompt_iteration_baseline.py'
$Prompt = Join-Path $RepoRoot 'guqinagent\baseline\prompts\single_stage_annotation_compact_v33_addendum.txt'
$Manifest = Join-Path $RepoRoot 'guqinagent\baseline\results\v33_generation_manifest.json'
$Arguments = @(
    $Runner,
    '--phase', $Phase,
    '--parent', 'v9',
    '--candidate-id', 'v33',
    '--candidate-prompt', $Prompt,
    '--generation-manifest', $Manifest
)
if ($PrepareOnly) { $Arguments += '--prepare-only' }
if ($RetryErrors) { $Arguments += '--retry-errors' }

Write-Host "Starting or resuming V9/V33 $Phase comparison."
Write-Host ("Progress page: " + (Join-Path $RepoRoot "guqinagent\baseline\reports\prompt_v33_${Phase}_comparison.html"))
& $PythonExe @Arguments
exit $LASTEXITCODE
