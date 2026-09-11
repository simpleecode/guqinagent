$ErrorActionPreference = 'Stop'
$project = 'D:\sitongli\sitongli-guanshanyue'
$root = 'C:\Users\30343\.codex\visualizations\2026\09\03\01a065f1-9f9a-77a2-97e1-effc6361e996\gqs_v12_train_run1'
$input = Join-Path $project 'ABC_J\agent_training\inferred_gqs_v12\inferred_trajectories_train.jsonl'
$ids = Join-Path $project 'ABC_J\agent_training\pitch_eligible_gqs_v12_train\pitch_eligible_phrase_ids.txt'
$shardDirs = 0..5 | ForEach-Object { Join-Path $root "messages_gqs_v12_train_run1_shard_$_" }

function Report-Ready {
    return @($shardDirs | Where-Object { Test-Path (Join-Path $_ 'generation_report.json') }).Count -eq 6
}

while (-not (Report-Ready)) { Start-Sleep -Seconds 30 }

# Retry only stages recorded as failed.  A bounded loop prevents an API outage
# from keeping the overnight supervisor alive forever.
for ($round = 1; $round -le 3; $round++) {
    $reports = @($shardDirs | ForEach-Object {
        $path = Join-Path $_ 'generation_report.json'
        if (Test-Path $path) { Get-Content $path -Raw | ConvertFrom-Json }
    })
    $failed = @($reports | ForEach-Object { $_.failures } | Where-Object { $_ })
    if ($failed.Count -eq 0) { break }
    foreach ($dir in $shardDirs) {
        $reportPath = Join-Path $dir 'generation_report.json'
        if (-not (Test-Path $reportPath)) { continue }
        $report = Get-Content $reportPath -Raw | ConvertFrom-Json
        if (@($report.failures).Count -eq 0) { continue }
        $name = Split-Path $dir -Leaf
        $idx = [int]($name -replace '.*_([0-5])$','$1')
        & python (Join-Path $project 'ABC_J\scripts\generate_teacher_tool_trajectories.py') `
            --input $input --trajectory-id-file $ids --limit 100000 `
            --output-dir $dir --score-shard-count 6 --score-shard-index $idx `
            --max-tool-rounds 24 --max-attempts 6 --allow-private-reasoning-leakage `
            --include-guqinizer-no-op --model glm-5.3 --min-interval 1 --resume --retry-failed
    }
}

$merged = Join-Path $root 'messages_gqs_v12_train_run1_merged_raw'
if (-not (Test-Path (Join-Path $merged 'generation_report.json'))) {
    $mergeArgs = @((Join-Path $project 'scripts\merge_teacher_retry_outputs.py'))
    foreach ($dir in $shardDirs) { $mergeArgs += '--input-dir'; $mergeArgs += $dir }
    $mergeArgs += '--output-dir'; $mergeArgs += $merged
    & python @mergeArgs
}

$redacted = Join-Path $root 'messages_gqs_v12_train_run1_final_redacted'
if (-not (Test-Path (Join-Path $redacted 'reasoning_redaction_report.json'))) {
    & python (Join-Path $project 'scripts\redact_teacher_reasoning.py') `
        --input-dir $merged --output-dir $redacted --model glm-5.3 --max-attempts 6
}
Write-Output "GQS v12 run1 complete: $redacted"
