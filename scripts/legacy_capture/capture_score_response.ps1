param(
  [Parameter(Mandatory=$true)][string]$ScoreKey,
  [Parameter(Mandatory=$true)][int]$ScoreId,
  [Parameter(Mandatory=$true)][string]$Title,
  [string]$Slug = "",
  [string]$FromKey = "",
  [string]$FromId = "",
  [int]$Duration = 75,
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
$captureDir = Join-Path $evidenceDir "http_capture_$stamp"
New-Item -ItemType Directory -Force -Path $evidenceDir, $outDir, $captureDir | Out-Null

& $Adb forward tcp:27042 tcp:27042 | Out-Null
$appPid = (& $Adb shell pidof $Package).Trim()
if (-not $appPid) {
  throw "$Package is not running. Launch the Frida Gadget clone app first."
}

$needles = @(
  $ScoreKey,
  $ScoreId,
  $FromKey,
  $FromId,
  "/v2/scores/$ScoreId/data",
  "notes",
  "jians",
  "sections",
  "dataops",
  "lyric"
) | Where-Object { $_ }

$args = @(
  (Join-Path $RootDir "scripts\frida_capture_http_responses.py"),
  "--adb", $Adb,
  "--package", $Package,
  "--out-dir", $captureDir,
  "--duration", [string]$Duration
)
foreach ($needle in $needles) {
  $args += @("--needle", [string]$needle)
}

Write-Host "HTTP capture is listening for $Duration seconds."
Write-Host "Now re-open or refresh this score in the phone app: $Title [$ScoreKey/$ScoreId]"
python @args
if ($LASTEXITCODE -ne 0) {
  throw "HTTP response capture failed with exit code $LASTEXITCODE."
}

$candidateDir = Join-Path $captureDir "candidates"
python (Join-Path $RootDir "scripts\normalize_score_export.py") `
  --input $candidateDir `
  --metadata-dir $candidateDir `
  --outdir $outDir `
  --score-key $ScoreKey `
  --score-id $ScoreId `
  --title $Title
if ($LASTEXITCODE -ne 0) {
  throw "captured HTTP responses did not contain a standardizable score body. Evidence kept at: $captureDir"
}

$manifest = [ordered]@{
  score_key = $ScoreKey
  score_id = $ScoreId
  title = $Title
  slug = $Slug
  package = $Package
  pid = $appPid
  capture_dir = $captureDir
  output = $outDir
  captured_at = (Get-Date).ToString("s")
  method = "http_response_capture"
}
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $scoreDir "manifest.json")

Write-Host "Score: $Title [$ScoreKey/$ScoreId]"
Write-Host "Capture: $captureDir"
Write-Host "Output: $outDir"
