param(
  [string]$Package = "com.sitongli.app.gadget",
  [string]$Adb = "C:\Users\93638\AppData\Local\Android\Sdk\platform-tools\adb.exe",
  [string]$OutDir = "cases\sitongli-guanshanyue\out",
  [string]$EvidenceDir = "cases\sitongli-guanshanyue\evidence"
)

$ErrorActionPreference = "Stop"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$jsonl = Join-Path $EvidenceDir "memory_payload_$stamp.jsonl"
$candidateDir = Join-Path $EvidenceDir "memory_json_candidates_$stamp"

& $Adb forward tcp:27042 tcp:27042 | Out-Null
$pid = (& $Adb shell pidof $Package).Trim()
if (-not $pid) {
  throw "$Package is not running. Launch the Frida Gadget clone app and open the Guanshanyue score/editor page first."
}

python cases\sitongli-guanshanyue\scripts\frida_scan_memory_strings.py `
  --adb $Adb `
  --package $Package `
  --out $jsonl `
  --max-hits 80 `
  --context 8192 `
  --needle SSG54sm8 `
  --needle 161577 `
  --needle notes `
  --needle jians `
  --needle sections `
  --needle dataops `
  --needle score_id `
  --needle score_key `
  --needle score_title `
  --needle 关山月

python cases\sitongli-guanshanyue\scripts\extract_memory_json_candidates.py `
  $jsonl `
  --out-dir $candidateDir

python cases\sitongli-guanshanyue\scripts\normalize_score_export.py `
  --input $candidateDir `
  --metadata-dir $candidateDir `
  --outdir $OutDir

Write-Host "JSONL: $jsonl"
Write-Host "Candidates: $candidateDir"
Write-Host "Output: $OutDir"
