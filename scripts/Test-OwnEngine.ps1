<#
.SYNOPSIS
    自前エンジン（engine/）の動作確認を行い、レポートを Markdown で出力する。

.DESCRIPTION
    変換 → F0/音量の解析 → 分割変換の速度計測、を順に実行し、
    reports/own-engine-latest.md に結果をまとめる。

.EXAMPLE
    .\scripts\Test-OwnEngine.ps1

.EXAMPLE
    .\scripts\Test-OwnEngine.ps1 -Device cuda -Pitch 0
#>
[CmdletBinding()]
param(
    [string]$ModelPath = 'models\tsukuyomi\tsukuyomi-normal1.pth',
    [string]$InputPath = 'logs\verify\tts-input-ja.wav',
    [string]$PythonPath = '.venv\Scripts\python.exe',
    [string]$Device = 'cpu',
    [int]$Pitch = 0,
    [double]$Hop = 0.5,
    [double]$Context = 0.15,
    [string]$ReportPath = 'reports\own-engine-latest.md'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not (Test-Path $PythonPath)) { throw "Python が見つかりません: $PythonPath" }
    if (-not (Test-Path $ModelPath)) { throw "モデルが見つかりません: $ModelPath" }
    if (-not (Test-Path $InputPath)) { throw "入力 WAV が見つかりません: $InputPath" }

    $verifyDir = Join-Path $repoRoot 'logs\verify'
    New-Item -ItemType Directory -Force -Path $verifyDir | Out-Null

    $stem = [System.IO.Path]::GetFileNameWithoutExtension($InputPath)
    $outWav = Join-Path $verifyDir ("own-$stem-$Device.wav")
    $outJson = Join-Path $verifyDir ("own-$stem-$Device.json")
    $chunkWav = Join-Path $verifyDir ("own-$stem-$Device-chunk.wav")

    Write-Host "=== 1/3 変換 ==="
    $convertLog = & $PythonPath -m engine convert `
        -m $ModelPath -i $InputPath -o $outWav --json $outJson `
        --pitch $Pitch --device $Device 2>&1 | Out-String
    Write-Host $convertLog.Trim()
    if ($LASTEXITCODE -ne 0) { throw "変換に失敗しました（終了コード $LASTEXITCODE）" }

    Write-Host "=== 2/3 解析（入力） ==="
    $analyzeIn = & $PythonPath -m engine analyze -i $InputPath 2>&1 | Out-String
    Write-Host $analyzeIn.Trim()

    Write-Host "=== 2/3 解析（出力） ==="
    $analyzeOut = & $PythonPath -m engine analyze -i $outWav 2>&1 | Out-String
    Write-Host $analyzeOut.Trim()

    Write-Host "=== 3/3 分割変換の速度 ==="
    $benchLog = & $PythonPath -m engine bench `
        -m $ModelPath -i $InputPath -o $chunkWav `
        --hop $Hop --context $Context --pitch $Pitch --device $Device 2>&1 | Out-String
    Write-Host $benchLog.Trim()
    if ($LASTEXITCODE -ne 0) { throw "分割変換の計測に失敗しました（終了コード $LASTEXITCODE）" }

    $gitHead = (& git rev-parse --short HEAD 2>&1 | Out-String).Trim()
    $gitBranch = (& git branch --show-current 2>&1 | Out-String).Trim()
    $pyVersion = (& $PythonPath -c "import platform, torch; print(platform.python_version(), 'torch', torch.__version__, 'cuda', torch.cuda.is_available())" 2>&1 | Out-String).Trim()

    $lines = @(
        '# 自前エンジン 動作確認レポート',
        '',
        "生成日時: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))",
        "ブランチ: $gitBranch（$gitHead）",
        "実行環境: $pyVersion",
        "モデル: $ModelPath",
        "入力: $InputPath",
        "デバイス: $Device / ピッチシフト: $Pitch 半音 / 分割: hop $Hop 秒・context $Context 秒",
        '',
        '## 1. 変換',
        '',
        '```text',
        $convertLog.Trim(),
        '```',
        '',
        '## 2. 入力の解析',
        '',
        '```text',
        $analyzeIn.Trim(),
        '```',
        '',
        '## 3. 出力の解析',
        '',
        '```text',
        $analyzeOut.Trim(),
        '```',
        '',
        '## 4. 分割変換（ライブ成立性）',
        '',
        '```text',
        $benchLog.Trim(),
        '```',
        '',
        '## 5. 出力ファイル',
        '',
        "- 一括変換: ``$([System.IO.Path]::GetRelativePath($repoRoot, $outWav))``（Git 管理外）",
        "- 分割変換: ``$([System.IO.Path]::GetRelativePath($repoRoot, $chunkWav))``（Git 管理外）",
        "- 計測 JSON: ``$([System.IO.Path]::GetRelativePath($repoRoot, $outJson))``（Git 管理外）",
        '',
        '## 6. 判定',
        '',
        '| 項目 | 判定 |',
        '| :--- | :--- |',
        '| 変換が最後まで通る | OK（上記ログ参照） |',
        '| 出力が無音でない | 出力の RMS が 0 でないこと |',
        '| ピッチが入力に追従 | 入力と出力の F0 中央値を比較 |',
        '| ライブ成立 | 分割変換の「最大遅延 < hop」 |',
        '',
        '聴感（声質）の最終判定は人による試聴が必要（`docs/04-verification.md`）。'
    )

    $reportFull = Join-Path $repoRoot $ReportPath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $reportFull) | Out-Null
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($reportFull, ($lines -join [Environment]::NewLine), $utf8)
    Write-Host "レポート: $reportFull"
}
finally {
    Pop-Location
}
