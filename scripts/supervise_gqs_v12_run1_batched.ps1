$ErrorActionPreference = 'Stop'
$project = 'D:\sitongli\sitongli-guanshanyue'
$root = 'C:\Users\30343\.codex\visualizations\2026\09\03\01a065f1-9f9a-77a2-97e1-effc6361e996\gqs_v12_train_run1'
$launcher = Join-Path $project 'scripts\run_gqs_v12_worker.ps1'
$shardDirs = 0..5 | ForEach-Object { Join-Path $root "messages_gqs_v12_train_run1_shard_$_" }
$batches = @(@(0,1), @(2,3), @(4,5))
foreach ($batch in $batches) {
    $pids = @{}
    foreach ($i in $batch) {
        $dir = $shardDirs[$i]
        if (Test-Path (Join-Path $dir 'generation_report.json')) { continue }
        $out = Join-Path $root "worker_${i}.out.log"
        $err = Join-Path $root "worker_${i}.err.log"
        $p = Start-Process -FilePath 'pwsh.exe' -WorkingDirectory $project -WindowStyle Hidden `
            -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$launcher,"$i") `
            -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
        $pids[$i] = $p.Id
    }
    while (@($batch | Where-Object { -not (Test-Path (Join-Path $shardDirs[$_] 'generation_report.json')) }).Count -gt 0) {
        Start-Sleep -Seconds 30
        foreach ($i in @($pids.Keys)) {
            if (-not (Test-Path (Join-Path $shardDirs[$i] 'generation_report.json')) -and
                -not (Get-Process -Id $pids[$i] -ErrorAction SilentlyContinue)) {
                # One restart handles transient worker termination without
                # duplicating completed stages (the worker uses --resume).
                $out = Join-Path $root "worker_${i}.out.log"
                $err = Join-Path $root "worker_${i}.err.log"
                $p = Start-Process -FilePath 'pwsh.exe' -WorkingDirectory $project -WindowStyle Hidden `
                    -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',$launcher,"$i") `
                    -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
                $pids[$i] = $p.Id
            }
        }
    }
}

$merged = Join-Path $root 'messages_gqs_v12_train_run1_merged_raw'
if (-not (Test-Path (Join-Path $merged 'generation_report.json'))) {
    $args = @((Join-Path $project 'scripts\merge_teacher_retry_outputs.py'))
    foreach ($dir in $shardDirs) { $args += '--input-dir'; $args += $dir }
    $args += '--output-dir'; $args += $merged
    & python @args
}
$redacted = Join-Path $root 'messages_gqs_v12_train_run1_final_redacted'
if (-not (Test-Path (Join-Path $redacted 'reasoning_redaction_report.json'))) {
    & python (Join-Path $project 'scripts\redact_teacher_reasoning.py') `
        --input-dir $merged --output-dir $redacted --model glm-5.3 --max-attempts 6
}
Write-Output "GQS v12 run1 complete: $redacted"
