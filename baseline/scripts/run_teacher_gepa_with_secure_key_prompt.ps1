$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = 'GLM API Key - 一曲谱 GEPA 测试'
$form.StartPosition = 'CenterScreen'
$form.Size = New-Object System.Drawing.Size(580, 190)
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.TopMost = $true
$label = New-Object System.Windows.Forms.Label
$label.Location = New-Object System.Drawing.Point(20, 18)
$label.Size = New-Object System.Drawing.Size(530, 42)
$label.Text = '输入新的 GLM API Key。密钥只传给本次 GEPA 进程，完成后立即清除。'
$form.Controls.Add($label)
$box = New-Object System.Windows.Forms.TextBox
$box.Location = New-Object System.Drawing.Point(20, 67)
$box.Size = New-Object System.Drawing.Size(530, 25)
$box.UseSystemPasswordChar = $true
$form.Controls.Add($box)
$ok = New-Object System.Windows.Forms.Button
$ok.Location = New-Object System.Drawing.Point(375, 108)
$ok.Size = New-Object System.Drawing.Size(80, 30)
$ok.Text = '开始'
$ok.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.AcceptButton = $ok
$form.Controls.Add($ok)
$cancel = New-Object System.Windows.Forms.Button
$cancel.Location = New-Object System.Drawing.Point(470, 108)
$cancel.Size = New-Object System.Drawing.Size(80, 30)
$cancel.Text = '取消'
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.CancelButton = $cancel
$form.Controls.Add($cancel)

$dialog = $form.ShowDialog()
if ($dialog -ne [System.Windows.Forms.DialogResult]::OK -or [string]::IsNullOrWhiteSpace($box.Text)) {
    throw '未输入 API key，GEPA 测试未启动。'
}
$env:GLM_API_KEY = $box.Text
$box.Text = ''
$form.Dispose()
try {
    & (Join-Path $PSScriptRoot 'start_teacher_gepa.ps1') `
        -ScoreKey 'SWkES6xH' `
        -RunName 'teacher_gepa_smoke_SWkES6xH' `
        -SeedPrompt 'v9' `
        -FeedbackPhrases 4 `
        -ParetoPhrases 3 `
        -MinibatchSize 1 `
        -MaxMetricCalls 12 `
        -Seed 20261027 `
        -MaxRounds 8 `
        -Temperature 0.0
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:\GLM_API_KEY -ErrorAction SilentlyContinue
}
