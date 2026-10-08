param(
  [string]$Manifest = "cases\sitongli-guanshanyue\batch_scores.csv",
  [string]$Package = "com.sitongli.app.gadget",
  [string]$RootDir = "cases\sitongli-guanshanyue"
)

$ErrorActionPreference = "Stop"
$rows = Import-Csv $Manifest
if (-not $rows) {
  throw "No rows found in $Manifest"
}

function Convert-UnicodeEscapes([string]$Text) {
  if (-not $Text) {
    return $Text
  }
  return [regex]::Replace($Text, '\\u([0-9a-fA-F]{4})', {
    param($m)
    [char]([Convert]::ToInt32($m.Groups[1].Value, 16))
  })
}

foreach ($row in $rows) {
  $title = $row.title
  if ($row.title_unicode) {
    $title = Convert-UnicodeEscapes $row.title_unicode
  }
  if (-not $row.score_key -or -not $row.score_id -or -not $title) {
    throw "Each row must have score_key, score_id, and title or title_unicode. Bad row: $($row | ConvertTo-Json -Compress)"
  }

  Write-Host ""
  Write-Host "Open this score in the app, then press Enter here:"
  Write-Host ("  title={0}  score_key={1}  score_id={2}" -f $title, $row.score_key, $row.score_id)
  Read-Host | Out-Null

  $slug = $row.slug
  if (-not $slug) {
    $slug = $row.score_key
  }

  powershell -ExecutionPolicy Bypass -File (Join-Path $RootDir "scripts\export_score_from_memory.ps1") `
    -ScoreKey $row.score_key `
    -ScoreId ([int]$row.score_id) `
    -Title $title `
    -Slug $slug `
    -Package $Package `
    -RootDir $RootDir
}
