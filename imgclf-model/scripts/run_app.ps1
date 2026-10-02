# v6 launcher: pick a backbone and settings in a small window, then it runs the training.
# Save as imgclf-model\scripts\run_v6_app.ps1 and start it with:
#     powershell -ExecutionPolicy Bypass -File scripts\run_v6_app.ps1
# The "Save as" box auto-numbers: it finds the highest imgclf-model-vN.pt in weights\ and offers N+1.
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing
[System.Windows.Forms.Application]::EnableVisualStyles()

$root = Split-Path -Parent $PSScriptRoot                                  # imgclf-model folder
$venvPy = Join-Path (Split-Path -Parent $root) ".venv\Scripts\python.exe"   # no\.venv
$py = if (Test-Path $venvPy) { $venvPy } else { "python" }

# speeds measured on this laptop at 128 px (img/s), feature width, and a short note
$info = [ordered]@{
    "mobilenet_v2"    = @{ ips = 120; dim = 1280; note = "No download needed (weights already on disk). Fastest." }
    "efficientnet_b0" = @{ ips = 103; dim = 1280; note = "Needs a one-time ~20 MB download. Similar speed, usually a bit more accurate." }
    "convnext_tiny"   = @{ ips = 44;  dim = 768;  note = "Needs a one-time ~110 MB download. Slower, likely the most accurate." }
    "resnet50"        = @{ ips = 24;  dim = 2048; note = "Needs a one-time ~100 MB download. Slowest, not recommended." }
}
$presets = [ordered]@{
    "Quick (200 per class, 128 px)"    = @(200, 128)
    "Balanced (400 per class, 128 px)" = @(400, 128)
    "Best (500 per class, 160 px)"     = @(500, 160)
}

# highest version number handed out in this session (a launched run may not have written its file yet)
$script:lastLaunched = 0

function Get-NextName {
    $max = $script:lastLaunched
    $dir = Join-Path $root "weights"
    if (Test-Path $dir) {
        Get-ChildItem $dir -Filter "*.pt" | ForEach-Object {
            if ($_.Name -match '^imgclf[-_]model-v(\d+)\.pt$') {
                $n = [int]$Matches[1]
                if ($n -gt $max) { $max = $n }
            }
        }
    }
    return "imgclf-model-v$($max + 1).pt"
}

function New-Label($text, $x, $y, $w = 470, $h = 22) {
    $l = New-Object System.Windows.Forms.Label
    $l.Text = $text
    $l.Location = New-Object System.Drawing.Point($x, $y)
    $l.Size = New-Object System.Drawing.Size($w, $h)
    return $l
}

$form = New-Object System.Windows.Forms.Form
$form.Text = "Image classifier - launcher"
$form.ClientSize = New-Object System.Drawing.Size(510, 475)
$form.StartPosition = "CenterScreen"
$form.FormBorderStyle = "FixedDialog"
$form.MaximizeBox = $false
$form.Font = New-Object System.Drawing.Font("Segoe UI", 10)

$title = New-Label "Train imgclf (ImageNet-1k 64x64)" 20 12
$title.Font = New-Object System.Drawing.Font("Segoe UI", 12, [System.Drawing.FontStyle]::Bold)
$form.Controls.Add($title)

$form.Controls.Add((New-Label "Which backbone do you want to use?" 20 48))
$cbBackbone = New-Object System.Windows.Forms.ComboBox
$cbBackbone.DropDownStyle = "DropDownList"
$cbBackbone.Location = New-Object System.Drawing.Point(20, 72)
$cbBackbone.Size = New-Object System.Drawing.Size(470, 28)
foreach ($k in $info.Keys) { [void]$cbBackbone.Items.Add($k) }
$form.Controls.Add($cbBackbone)

$lblNote = New-Label "" 20 104 470 40
$lblNote.ForeColor = [System.Drawing.Color]::DimGray
$form.Controls.Add($lblNote)

$form.Controls.Add((New-Label "Preset:" 20 150))
$cbPreset = New-Object System.Windows.Forms.ComboBox
$cbPreset.DropDownStyle = "DropDownList"
$cbPreset.Location = New-Object System.Drawing.Point(20, 174)
$cbPreset.Size = New-Object System.Drawing.Size(470, 28)
foreach ($k in $presets.Keys) { [void]$cbPreset.Items.Add($k) }
$form.Controls.Add($cbPreset)

$form.Controls.Add((New-Label "Training images per class:" 20 218 220))
$nudPer = New-Object System.Windows.Forms.NumericUpDown
$nudPer.Location = New-Object System.Drawing.Point(260, 215)
$nudPer.Size = New-Object System.Drawing.Size(100, 28)
$nudPer.Minimum = 50; $nudPer.Maximum = 1300; $nudPer.Increment = 50; $nudPer.Value = 400
$form.Controls.Add($nudPer)

$form.Controls.Add((New-Label "Upscale size (px):" 20 254 220))
$nudSize = New-Object System.Windows.Forms.NumericUpDown
$nudSize.Location = New-Object System.Drawing.Point(260, 251)
$nudSize.Size = New-Object System.Drawing.Size(100, 28)
$nudSize.Minimum = 64; $nudSize.Maximum = 224; $nudSize.Increment = 32; $nudSize.Value = 128
$form.Controls.Add($nudSize)

$chkFlip = New-Object System.Windows.Forms.CheckBox
$chkFlip.Text = "Also use mirrored images (2x time, more memory)"
$chkFlip.Location = New-Object System.Drawing.Point(20, 288)
$chkFlip.Size = New-Object System.Drawing.Size(470, 26)
$form.Controls.Add($chkFlip)

$lblEst = New-Label "" 20 322 470 50
$lblEst.ForeColor = [System.Drawing.Color]::DarkSlateBlue
$form.Controls.Add($lblEst)

$form.Controls.Add((New-Label "Save as:" 20 384 70))
$txtSave = New-Object System.Windows.Forms.TextBox
$txtSave.Location = New-Object System.Drawing.Point(95, 381)
$txtSave.Size = New-Object System.Drawing.Size(395, 28)
$form.Controls.Add($txtSave)

$btnRun = New-Object System.Windows.Forms.Button
$btnRun.Text = "Run training"
$btnRun.Location = New-Object System.Drawing.Point(20, 425)
$btnRun.Size = New-Object System.Drawing.Size(150, 34)
$form.Controls.Add($btnRun)

$btnBench = New-Object System.Windows.Forms.Button
$btnBench.Text = "Speed test"
$btnBench.Location = New-Object System.Drawing.Point(180, 425)
$btnBench.Size = New-Object System.Drawing.Size(130, 34)
$form.Controls.Add($btnBench)

$btnClose = New-Object System.Windows.Forms.Button
$btnClose.Text = "Close"
$btnClose.Location = New-Object System.Drawing.Point(390, 425)
$btnClose.Size = New-Object System.Drawing.Size(100, 34)
$form.Controls.Add($btnClose)

function Update-Estimate {
    $b = $cbBackbone.SelectedItem
    if (-not $b) { return }
    $lblNote.Text = $info[$b].note
    $ips = $info[$b].ips * [math]::Pow(128.0 / [double]$nudSize.Value, 2)
    $mult = 1
    if ($chkFlip.Checked) { $mult = 2 }
    $nTrain = [double]$nudPer.Value * 1000 * $mult
    $hours = ($nTrain + 20000) / $ips / 3600
    $gb = $nTrain * $info[$b].dim * 2 / 1e9
    $text = "Rough time: {0:N1} h (your benchmark, real speed is usually a bit lower)`nFeature cache: ~{1:N1} GB on disk, ~{2:N1} GB of RAM while loading" -f $hours, $gb, (2 * $gb)
    if (2 * $gb -gt 4) { $text += "  - close other apps first!" }
    $lblEst.Text = $text
}

$cbBackbone.Add_SelectedIndexChanged({ Update-Estimate })
$nudPer.Add_ValueChanged({ Update-Estimate })
$nudSize.Add_ValueChanged({ Update-Estimate })
$chkFlip.Add_CheckedChanged({ Update-Estimate })
$cbPreset.Add_SelectedIndexChanged({
    $p = $presets[$cbPreset.SelectedItem]
    if ($p) { $nudPer.Value = $p[0]; $nudSize.Value = $p[1] }
    Update-Estimate
})

function Start-Console($command) {
    $full = "Set-Location '$root'; $command; Write-Host ''; Write-Host 'Finished - you can close this window.' -ForegroundColor Green"
    Start-Process powershell -ArgumentList @('-NoExit', '-ExecutionPolicy', 'Bypass', '-Command', $full)
}

$btnRun.Add_Click({
    $b = $cbBackbone.SelectedItem
    if (-not $b) {
        [void][System.Windows.Forms.MessageBox]::Show("Pick a backbone first.", "Launcher")
        return
    }
    if ($b -ne "mobilenet_v2") {
        $ans = [System.Windows.Forms.MessageBox]::Show("$b needs a one-time download of its pretrained weights. Continue?", "Download needed", "YesNo", "Question")
        if ($ans -ne "Yes") { return }
    }

    $fileName = $txtSave.Text.Trim()
    if (-not $fileName) { $fileName = Get-NextName }
    if (-not $fileName.EndsWith(".pt")) { $fileName += ".pt" }
    if ($fileName -match '[^\w\.\-]') {
        [void][System.Windows.Forms.MessageBox]::Show("Use only letters, numbers, dots, dashes and underscores in the file name.", "Launcher")
        return
    }
    $outRel = "weights/$fileName"
    if (Test-Path (Join-Path $root $outRel)) {
        $ans = [System.Windows.Forms.MessageBox]::Show("$outRel already exists. Overwrite it?", "File exists", "YesNo", "Warning")
        if ($ans -ne "Yes") { return }
    }

    $a = "--backbone $b --img-size $($nudSize.Value) --per-class $($nudPer.Value) --val-per-class 20 --out '$outRel'"
    if ($chkFlip.Checked) { $a += " --flip-train" }

    # small note next to the weights so you remember which settings made which version
    New-Item -ItemType Directory -Force -Path (Join-Path $root "weights") | Out-Null
    $noteFile = Join-Path $root ($outRel -replace '\.pt$', '.txt')
    Set-Content -Path $noteFile -Value "backbone=$b img-size=$($nudSize.Value) per-class=$($nudPer.Value) flip=$($chkFlip.Checked) started=$(Get-Date -Format 'yyyy-MM-dd HH:mm')"

    Start-Console "& '$py' scripts/train_imgclf.py $a"

    # move on to the next free version number for the next run
    if ($fileName -match '^imgclf[-_]model-v(\d+)\.pt$') {
        $n = [int]$Matches[1]
        if ($n -gt $script:lastLaunched) { $script:lastLaunched = $n }
        $txtSave.Text = Get-NextName
    }
})

$btnBench.Add_Click({
    Start-Console "& '$py' scripts/train_imgclf.py --benchmark --img-size $($nudSize.Value) --per-class $($nudPer.Value)"
})

$btnClose.Add_Click({ $form.Close() })

$cbBackbone.SelectedIndex = 0
$cbPreset.SelectedIndex = 1
$txtSave.Text = Get-NextName
Update-Estimate
[void]$form.ShowDialog()