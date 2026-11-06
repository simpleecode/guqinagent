param([ValidateSet('v33', 'v34')][string]$Candidate = 'v33')

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$form = New-Object System.Windows.Forms.Form
$form.Text = "GLM API Key - 仅用于本次 $Candidate 测试"
$form.StartPosition = 'CenterScreen'
$form.Size = New-Object System.Drawing.Size(560, 185)
$form.FormBorderStyle = 'FixedDialog'
$form.MaximizeBox = $false
$form.MinimizeBox = $false
$form.TopMost = $true

$label = New-Object System.Windows.Forms.Label
$label.Location = New-Object System.Drawing.Point(20, 18)
$label.Size = New-Object System.Drawing.Size(510, 42)
$label.Text = '请输入新的 GLM API Key。它不会保存到文件或日志，只传给本次训练进程。'
$form.Controls.Add($label)

$box = New-Object System.Windows.Forms.TextBox
$box.Location = New-Object System.Drawing.Point(20, 65)
$box.Size = New-Object System.Drawing.Size(510, 25)
$box.UseSystemPasswordChar = $true
$form.Controls.Add($box)

$ok = New-Object System.Windows.Forms.Button
$ok.Location = New-Object System.Drawing.Point(355, 105)
$ok.Size = New-Object System.Drawing.Size(80, 30)
$ok.Text = '开始测试'
$ok.DialogResult = [System.Windows.Forms.DialogResult]::OK
$form.AcceptButton = $ok
$form.Controls.Add($ok)

$cancel = New-Object System.Windows.Forms.Button
$cancel.Location = New-Object System.Drawing.Point(450, 105)
$cancel.Size = New-Object System.Drawing.Size(80, 30)
$cancel.Text = '取消'
$cancel.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
$form.CancelButton = $cancel
$form.Controls.Add($cancel)

$result = $form.ShowDialog()
if ($result -ne [System.Windows.Forms.DialogResult]::OK -or [string]::IsNullOrWhiteSpace($box.Text)) {
    throw '未输入 API key，测试未启动。'
}

$env:GLM_API_KEY = $box.Text
$box.Text = ''
$form.Dispose()

try {
    $launcher = Join-Path $PSScriptRoot "start_${Candidate}_prompt_iteration.ps1"
    & $launcher -Phase diagnostic
    exit $LASTEXITCODE
}
finally {
    Remove-Item Env:\GLM_API_KEY -ErrorAction SilentlyContinue
}
