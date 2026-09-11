param(
  [Parameter(Mandatory=$true)][string]$ScoreKey,
  [Parameter(Mandatory=$true)][int]$ScoreId,
  [Parameter(Mandatory=$true)][string]$Title,
  [string]$Slug = "",
  [string]$FromKey = "",
  [string]$FromId = "",
  [string]$Package = "com.sitongli.app.gadget",
  [string]$Adb = "C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe",
  [string]$RootDir = "cases\sitongli-guanshanyue"
)

$ErrorActionPreference = "Stop"
if (-not $Slug) {
  $Slug = ($ScoreKey -replace '[^A-Za-z0-9_-]', '_')
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$scoreDir = Join-Path $RootDir ("batch\" + $Slug)
$evidenceDir = Join-Path $scoreDir "evidence"
$outDir = Join-Path $scoreDir "out"
New-Item -ItemType Directory -Force -Path $evidenceDir, $outDir | Out-Null
Remove-Item -LiteralPath (Join-Path $outDir "raw_data.json"), (Join-Path $outDir "data.json") -Force -ErrorAction SilentlyContinue

$jsonl = Join-Path $evidenceDir "memory_payload_$stamp.jsonl"
$candidateDir = Join-Path $evidenceDir "memory_json_candidates_$stamp"

& $Adb forward tcp:27042 tcp:27042 | Out-Null
$appPid = (& $Adb shell pidof $Package).Trim()
if (-not $appPid) {
  throw "$Package is not running. Launch the Frida Gadget clone app and open this score/editor page first."
}

python (Join-Path $RootDir "scripts\frida_scan_memory_strings.py") `
  --adb $Adb `
  --package $Package `
  --out $jsonl `
  --max-hits 80 `
  --context 8192 `
  --needle $ScoreKey `
  --needle $ScoreId `
  --needle notes `
  --needle jians `
  --needle sections `
  --needle dataops `
  --needle score_id `
  --needle score_key `
  --needle score_title `
  --needle $Title
if ($LASTEXITCODE -ne 0) {
  throw "memory scan failed with exit code $LASTEXITCODE. Keep the target score/editor page open in the Gadget clone app, then retry."
}

python (Join-Path $RootDir "scripts\extract_memory_json_candidates.py") `
  $jsonl `
  --out-dir $candidateDir
if ($LASTEXITCODE -ne 0) {
  throw "JSON candidate extraction failed with exit code $LASTEXITCODE."
}

$cloneSource = [pscustomobject]@{
  from_key = $FromKey
  from_id = $FromId
  from_title = $null
}
$cloneSourceJson = (& python (Join-Path $RootDir "scripts\find_clone_source.py") `
    $candidateDir `
    --score-key $ScoreKey `
    --score-id $ScoreId)
if ($LASTEXITCODE -eq 0 -and $cloneSourceJson) {
  $detectedCloneSource = $cloneSourceJson | ConvertFrom-Json
  if ($detectedCloneSource.from_key -or $detectedCloneSource.from_id) {
    $cloneSource = $detectedCloneSource
  }
}
elseif (-not $FromKey -and -not $FromId) {
  throw "clone source detection failed with exit code $LASTEXITCODE."
}

if ($cloneSource.from_key -or $cloneSource.from_id) {
  Write-Host "Clone source detected: from_key=$($cloneSource.from_key) from_id=$($cloneSource.from_id) from_title=$($cloneSource.from_title)"
  $sourceJsonl = Join-Path $evidenceDir "from_source_payload_$stamp.jsonl"
  $sourceCandidateDir = Join-Path $evidenceDir "from_source_candidates_$stamp"
  $sourceNeedles = @(
    $ScoreKey,
    $ScoreId,
    $cloneSource.from_key,
    $cloneSource.from_id,
    "notes",
    "jians",
    "sections",
    "tuning",
    "lyric"
  ) | Where-Object { $_ }

  $scanArgs = @(
    (Join-Path $RootDir "scripts\frida_scan_memory_strings.py"),
    "--adb", $Adb,
    "--package", $Package,
    "--out", $sourceJsonl,
    "--max-hits", "120",
    "--context", "16384"
  )
  foreach ($needle in $sourceNeedles) {
    $scanArgs += @("--needle", [string]$needle)
  }
  python @scanArgs
  if ($LASTEXITCODE -ne 0) {
    throw "clone source memory scan failed with exit code $LASTEXITCODE."
  }

  python (Join-Path $RootDir "scripts\extract_memory_json_candidates.py") `
    $sourceJsonl `
    --out-dir $sourceCandidateDir
  if ($LASTEXITCODE -ne 0) {
    throw "clone source JSON candidate extraction failed with exit code $LASTEXITCODE."
  }

  python (Join-Path $RootDir "scripts\normalize_score_export.py") `
    --input $sourceCandidateDir `
    --metadata-dir $candidateDir `
    --outdir $outDir `
    --score-key $ScoreKey `
    --score-id $ScoreId `
    --title $Title
  if ($LASTEXITCODE -ne 0) {
    throw "score data was not found after clone source fallback. Primary candidates: $candidateDir. Source candidates: $sourceCandidateDir."
  }
} else {
  python (Join-Path $RootDir "scripts\normalize_score_export.py") `
    --input $candidateDir `
    --metadata-dir $candidateDir `
    --outdir $outDir `
    --score-key $ScoreKey `
    --score-id $ScoreId `
    --title $Title
  if ($LASTEXITCODE -ne 0) {
    throw "score data was not found in JSON memory candidates. Candidate evidence was kept at: $candidateDir. If the phone is already on the score/editor page, this score may need the authorized API/data-response capture path instead of plain JSON memory extraction."
  }
}

$manifest = [ordered]@{
  score_key = $ScoreKey
  score_id = $ScoreId
  title = $Title
  slug = $Slug
  package = $Package
  pid = $appPid
  jsonl = $jsonl
  candidates = $candidateDir
  output = $outDir
  captured_at = (Get-Date).ToString("s")
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $scoreDir "manifest.json")

Write-Host "Score: $Title [$ScoreKey/$ScoreId]"
Write-Host "JSONL: $jsonl"
Write-Host "Candidates: $candidateDir"
Write-Host "Output: $outDir"
