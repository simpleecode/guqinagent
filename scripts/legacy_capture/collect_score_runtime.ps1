param(
  [string]$Adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$Package = "com.sitongli.app.gadget",
  [int]$ScoreId = 161749,
  [string]$ScoreKey = "SWDFmDZX",
  [int]$FromId = 155861,
  [string]$FromKey = "",
  [string]$Title = "score",
  [Parameter(Mandatory = $true)]
  [string]$Tonic,
  [string]$TuningName = "",
  [string]$TuningValues = "",
  [int]$NotesLength = 140,
  [string]$Slug = "",
  [string]$RootDir = "cases\sitongli-guanshanyue",
  [string]$OutRoot = "",
  [switch]$LaunchDeepLink,
  [switch]$AutoTap,
  [switch]$StopAppOnExit,
  [int]$WarmupSeconds = 4,
  [int]$ScrollPasses = 8,
  [int]$PostGestureSeconds = 10
)

$ErrorActionPreference = "Stop"
chcp 65001 > $null
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONIOENCODING = "utf-8"

if (-not $Slug) {
  $Slug = ($ScoreKey -replace '[^A-Za-z0-9_-]', '_')
}

$RootDir = [System.IO.Path]::GetFullPath($RootDir)
if (-not $OutRoot) {
  $OutRoot = Join-Path $RootDir ("batch\" + $Slug)
}
$OutRoot = [System.IO.Path]::GetFullPath($OutRoot)
$EvidenceDir = Join-Path $OutRoot "evidence"
$Jsonl = Join-Path $EvidenceDir "runtime_note_slur.jsonl"
$HookStdout = Join-Path $EvidenceDir "frida_stdout.log"
$HookStderr = Join-Path $EvidenceDir "frida_stderr.log"
$ExportDir = Join-Path $OutRoot "out"
New-Item -ItemType Directory -Force -Path $EvidenceDir, $ExportDir | Out-Null
Remove-Item -LiteralPath $Jsonl, $HookStdout, $HookStderr -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath (Join-Path $ExportDir "raw_data.json"), (Join-Path $ExportDir "data.json") -Force -ErrorAction SilentlyContinue

& $Adb forward tcp:27042 tcp:27042 | Out-Null

try {
  Get-CimInstance Win32_Process -Filter "name='python.exe'" -ErrorAction Stop |
    Where-Object { $_.CommandLine -like "*frida_decode_score_jians.py*" -or $_.CommandLine -like "*frida_dump_score_bytes.py*" } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
catch {
  Write-Warning "Could not inspect existing Python command lines; continuing without stale-hook cleanup."
}

# `adb shell pidof` produces no pipeline object when the app is not running;
# normalize that case to an empty string so the launcher branch below can
# recover instead of calling `.Trim()` on `$null`.
$appPid = ((& $Adb shell pidof $Package | Out-String).Trim())
if ($LaunchDeepLink) {
  # A previously open score can render while Frida is warming up and be
  # mistaken for the next target.  Begin every deep-link capture from the app
  # home screen so the first score composition belongs to this request.
  & $Adb shell am force-stop $Package | Out-Null
  & $Adb shell monkey -p $Package -c android.intent.category.LAUNCHER 1 | Out-Null
  Start-Sleep -Seconds 2
  $appPid = ((& $Adb shell pidof $Package | Out-String).Trim())
}
if (-not $appPid) {
  throw "$Package is not running. Open the Gadget app first, then start runtime capture."
}

$hookArgs = @(
  (Join-Path $RootDir "scripts\frida_decode_score_jians.py"),
  "--adb", $Adb,
  "--blutter-js", (Join-Path $RootDir "work\blutter_out\blutter_frida.js"),
  "--host", "127.0.0.1:27042",
  "--out", $Jsonl,
  "--only", "note_slur",
  "--only", "note_render",
  "--only", "note_tuning",
  "--only", "view_get_jians",
  "--only", "jianzi_component",
  "--offset", "source_jab_event=0x63f220",
  "--entry-only",
  "--depth", "10",
  # Long scores can contain over 2,000 source events plus layout/control
  # entries.  1,024 silently truncated the source list before the ornament
  # objects near the later pages were decoded.
  "--array-limit", "4096",
  "--map-limit", "1024"
)
$hook = $null
$method = "runtime_source_event_list_with_wza_fallback"

function Quote-ProcessArg([string]$Value) {
  if ($null -eq $Value) { return '""' }
  if ($Value -notmatch '[\s"]') { return $Value }
  return '"' + ($Value -replace '"', '\"') + '"'
}

function Wait-ForSourceEvent([int]$MaximumSeconds) {
  # The source list is emitted as soon as the score body has been composed.
  # Waiting for that evidence is both faster and safer than a fixed sleep or
  # arbitrary full-page scrolling.
  $deadline = (Get-Date).AddSeconds([Math]::Max(0, $MaximumSeconds))
  do {
    if (Test-Path $Jsonl) {
      # The JSONL also contains a "hooked/source_jab_event" setup record.
      # Only an enter_decoded record proves that the current score's source
      # event list has actually been observed.
      $found = Select-String -LiteralPath $Jsonl -Pattern '"event": "enter_decoded", "name": "source_jab_event"' -Quiet -ErrorAction SilentlyContinue
      if ($found) {
        Start-Sleep -Milliseconds 800  # let the JSON line finish flushing
        return $true
      }
    }
    Start-Sleep -Milliseconds 400
  } while ((Get-Date) -lt $deadline)
  return $false
}

function Wait-ForHookReady([int]$MaximumSeconds) {
  # Starting the route before Frida has installed source_jab_event can lose the
  # only full-score callback (most visible with large scores).  The hook emits
  # both this marker and "attached" when it is ready to observe navigation.
  $deadline = (Get-Date).AddSeconds([Math]::Max(1, $MaximumSeconds))
  do {
    if (Test-Path $HookStdout) {
      $ready = Select-String -LiteralPath $HookStdout -Pattern '"name": "source_jab_event"' -Quiet -ErrorAction SilentlyContinue
      $attached = Select-String -LiteralPath $HookStdout -Pattern 'attached pid=' -Quiet -ErrorAction SilentlyContinue
      if ($ready -and $attached) { return $true }
    }
    Start-Sleep -Milliseconds 250
  } while ((Get-Date) -lt $deadline)
  return $false
}

function Get-ScorePageText() {
  $remoteXml = "/sdcard/codex_score_window.xml"
  & $Adb shell uiautomator dump $remoteXml | Out-Null
  if ($LASTEXITCODE -ne 0) { return "" }
  return ((& $Adb exec-out cat $remoteXml | Out-String))
}

function Dismiss-NotFoundDialog([string]$Xml) {
  # The route error is a modal dialog.  Retrying the deep link without first
  # pressing its confirmation button leaves the modal on top of the detail
  # page, so the retry cannot reach the score entry.
  foreach ($node in [regex]::Matches($Xml, '<node\b[^>]*/>')) {
    if ($node.Value -notmatch '(?:text|content-desc)="确定"') { continue }
    $bounds = [regex]::Match($node.Value, 'bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"')
    if (-not $bounds.Success) { continue }
    $x = [int](([int]$bounds.Groups[1].Value + [int]$bounds.Groups[3].Value) / 2)
    $y = [int](([int]$bounds.Groups[2].Value + [int]$bounds.Groups[4].Value) / 2)
    & $Adb shell input tap $x $y
    Start-Sleep -Milliseconds 350
    return $true
  }
  Write-Warning "Score route reported '未找到该琴谱' but its 确定 button was not found."
  return $false
}

function Open-TargetScore() {
  $deepLink = "sitongli://app/scores/$ScoreId"
  $adbArgs = @("shell", "am", "start", "-W", "-a", "android.intent.action.VIEW", "-d", $deepLink, $Package)
  Start-Process -FilePath $Adb -ArgumentList $adbArgs -WindowStyle Hidden | Out-Null
  # The following semantic wait handles the normal route transition.  Keep a
  # brief pause only so an immediate "not found" response can be retried.
  Start-Sleep -Milliseconds 600
  $pageXml = Get-ScorePageText
  if ($pageXml -match "未找到该琴谱") {
    # A transient route/render failure can show the same page; retry the
    # exact deep link once, after closing the modal.  Never tap the score-body
    # coordinate on this UI.
    Write-Warning "Score route reported '未找到该琴谱'; retrying once."
    Dismiss-NotFoundDialog $pageXml | Out-Null
    Start-Process -FilePath $Adb -ArgumentList $adbArgs -WindowStyle Hidden | Out-Null
    Start-Sleep -Milliseconds 900
    $retryXml = Get-ScorePageText
    if ($retryXml -match "未找到该琴谱") {
      Dismiss-NotFoundDialog $retryXml | Out-Null
      throw "score_not_found_after_retry: score_id=$ScoreId score_key=$ScoreKey"
    }
  }
}

function Enter-ScoreBody([int]$MaximumSeconds) {
  # Do not rely on a screen coordinate: wait for, and click, the actual
  # clickable row whose accessibility text includes "乐谱".  This avoids
  # accidentally interacting with clone/remarks controls while the detail
  # page is still rendering.
  $deadline = (Get-Date).AddSeconds([Math]::Max(1, $MaximumSeconds))
  do {
    $xml = Get-ScorePageText
    if ($xml -match "未找到该琴谱") {
      Dismiss-NotFoundDialog $xml | Out-Null
      throw "score_not_found_after_retry: score_id=$ScoreId score_key=$ScoreKey"
    }
    $entry = [regex]::Match(
      $xml,
      'content-desc="[^"]*乐谱[^"]*"[^>]*clickable="true"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'
    )
    if ($entry.Success) {
      $x = [int](([int]$entry.Groups[1].Value + [int]$entry.Groups[3].Value) / 2)
      $y = [int](([int]$entry.Groups[2].Value + [int]$entry.Groups[4].Value) / 2)
      & $Adb shell input tap $x $y
      return
    }
    Start-Sleep -Milliseconds 350
  } while ((Get-Date) -lt $deadline)
  throw "score_entry_not_found: score_id=$ScoreId score_key=$ScoreKey"
}

function Wait-ForVisibleTuningName([int]$MaximumSeconds) {
  # The score header exposes names such as "正调定弦：" even when the App
  # does not invoke its separate runtime tuning callback.
  $deadline = (Get-Date).AddSeconds([Math]::Max(1, $MaximumSeconds))
  do {
    $xml = Get-ScorePageText
    $match = [regex]::Match($xml, 'content-desc="([^"\r\n]*定弦)：?"')
    if ($match.Success) { return $match.Groups[1].Value }
    Start-Sleep -Milliseconds 300
  } while ((Get-Date) -lt $deadline)
  return ""
}

try {
  $hookArgLine = ($hookArgs | ForEach-Object { Quote-ProcessArg ([string]$_) }) -join " "
  $hook = Start-Process -FilePath "python" -ArgumentList $hookArgLine -RedirectStandardOutput $HookStdout -RedirectStandardError $HookStderr -PassThru -WindowStyle Hidden

  if (-not (Wait-ForHookReady ([Math]::Max($WarmupSeconds, 8)))) {
    throw "source_hook_not_ready: score_id=$ScoreId score_key=$ScoreKey"
  }
  if ($LaunchDeepLink) {
    Open-TargetScore
  }
  if ($AutoTap) {
    # Some score detail pages take longer to load their semantic "乐谱" row;
    # waiting is safe because we never fall back to a coordinate tap.
    Enter-ScoreBody 15
    if (-not $TuningName) { $TuningName = Wait-ForVisibleTuningName 5 }
  }
  if ($ScrollPasses -gt 0) {
    for ($i = 0; $i -lt $ScrollPasses; $i++) {
      & $Adb shell input swipe 720 1450 720 620 450 | Out-Null
      Start-Sleep -Milliseconds 650
    }
    for ($i = 0; $i -lt [Math]::Min(2, $ScrollPasses); $i++) {
      & $Adb shell input swipe 720 620 720 1450 450 | Out-Null
      Start-Sleep -Milliseconds 650
    }
  }
  $sourceEventSeen = Wait-ForSourceEvent $PostGestureSeconds
  if (-not $sourceEventSeen) {
    throw "source_event_timeout: score_id=$ScoreId score_key=$ScoreKey wait_seconds=$PostGestureSeconds"
  }

  if (!$hook.HasExited) {
    Stop-Process -Id $hook.Id -Force
  }

  $extractArgs = @(
    (Join-Path $RootDir "scripts\extract_score_runtime_windows.py"),
    "--input", $Jsonl,
    "--out-dir", $ExportDir,
    "--score-id", "$ScoreId",
    "--score-key", $ScoreKey,
    "--title", $Title,
    "--tonic", $Tonic,
    "--from-id", "$FromId",
    "--notes-length", "$NotesLength",
    "--assume-complete-runtime-window"
  )
  if ($FromKey) {
    $extractArgs += @("--from-key", $FromKey)
  }
  if ($TuningName) {
    $extractArgs += @("--tuning-name", $TuningName)
  }
  if ($TuningValues) {
    $extractArgs += @("--tuning-values", $TuningValues)
  }
  python @extractArgs
  $extractExit = $LASTEXITCODE
  if ($extractExit -ne 0) {
    throw "runtime source-event extraction failed with exit code ${extractExit}: score_id=$ScoreId score_key=$ScoreKey"
  }

  $manifest = [ordered]@{
    score_key = $ScoreKey
    score_id = $ScoreId
    from_key = $FromKey
    from_id = $FromId
    title = $Title
    slug = $Slug
    package = $Package
    method = $method
    jsonl = $Jsonl
    output = $ExportDir
    captured_at = (Get-Date).ToString("s")
  }
  $manifest | ConvertTo-Json -Depth 5 | Set-Content -Encoding UTF8 (Join-Path $OutRoot "manifest.json")

  Write-Host "raw_data.json: $(Join-Path $ExportDir 'raw_data.json')"
  Write-Host "data.json: $(Join-Path $ExportDir 'data.json')"
}
finally {
  if ($hook -and !$hook.HasExited) {
    Stop-Process -Id $hook.Id -Force -ErrorAction SilentlyContinue
  }
  if ($StopAppOnExit) {
    & $Adb shell am force-stop $Package | Out-Null
  }
}
