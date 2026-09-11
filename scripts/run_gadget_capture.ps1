param(
  [string]$Adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe",
  [string]$Package = "com.sitongli.app",
  [string]$Apk = ".\cases\sitongli-guanshanyue\out\sitongli-gadget-signed.apk",
  [switch]$InstallPatched,
  [switch]$SkipLaunch
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $Adb)) {
  throw "adb not found: $Adb"
}
if (!(Test-Path -LiteralPath $Apk)) {
  throw "patched APK not found: $Apk"
}

& $Adb devices

if ($InstallPatched) {
  Write-Host "[!] Installing the patched APK may fail if the original app has a different signature."
  Write-Host "[!] If Android reports INSTALL_FAILED_UPDATE_INCOMPATIBLE, use a test device/emulator or uninstall/reinstall manually after preserving account access."
  & $Adb install -r $Apk
}

& $Adb forward tcp:27042 tcp:27042 | Out-Host

if (!$SkipLaunch) {
  & $Adb shell monkey -p $Package -c android.intent.category.LAUNCHER 1 | Out-Host
  Start-Sleep -Seconds 2
}

python .\cases\sitongli-guanshanyue\scripts\frida_gadget_capture_score.py --adb $Adb --package $Package --out .\cases\sitongli-guanshanyue\evidence\gadget_capture_score.jsonl --no-launch
