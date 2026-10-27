<#
.SYNOPSIS
    Phase 0: 機材・実行環境を確認し、レポートを生成する。

.DESCRIPTION
    設計書 6 章 Phase 0「環境確認」に対応する。
    OS / CPU / メモリ / GPU / 空きディスク / 音声入出力デバイス / VC Client の配置状況を収集し、
    Markdown レポートと JSON を reports/ に出力する。

    レポートは docs/00-phase0-environment.md の機材台帳へ転記して使う。

.PARAMETER VcClientRoot
    VC Client を解凍したフォルダ。省略時は config/settings.json の vcClient.root を使う。

.PARAMETER OutputDir
    レポート出力先。省略時は reports/。

.EXAMPLE
    .\scripts\Check-Environment.ps1

.EXAMPLE
    .\scripts\Check-Environment.ps1 -VcClientRoot 'D:\Tools\vcclient_win_cuda'
#>
[CmdletBinding()]
param(
    [string]$VcClientRoot = '',
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\Common.ps1')

$repoRoot = Get-RepoRoot
$timestamp = Get-Date
$stamp = $timestamp.ToString('yyyyMMdd-HHmmss')

if ([string]::IsNullOrWhiteSpace($OutputDir)) { $OutputDir = Join-Path $repoRoot 'reports' }
if (-not (Test-Path -LiteralPath $OutputDir)) { New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null }

# ---------------------------------------------------------------- 設定読み込み
$configPath = Join-Path $repoRoot 'config\settings.json'
$config = $null
if (Test-Path -LiteralPath $configPath) {
    try {
        $config = Read-JsonFile -Path $configPath
    } catch {
        Write-Err $_.Exception.Message
        Write-Info 'config/settings.json の書式（JSON）を確認してください。設定なしで続行します。'
    }
    if ($null -ne $config -and [string]::IsNullOrWhiteSpace($VcClientRoot)) {
        $VcClientRoot = [string](Get-SettingValue -Object $config -Path 'vcClient.root' -Default '')
    }
}

$result = [ordered]@{
    generatedAt = $timestamp.ToString('o')
    host        = $env:COMPUTERNAME
    vcClient    = $null
    findings    = @()
}

# ---------------------------------------------------------------- OS
Write-Step 'OS / ハードウェア'
$os = $null
try { $os = Get-CimInstance -ClassName Win32_OperatingSystem } catch { }
$osName = if ($null -ne $os) { $os.Caption } else { 'unknown' }
$osVersion = if ($null -ne $os) { "$($os.Version) (Build $($os.BuildNumber))" } else { 'unknown' }
Write-Info "OS: $osName / $osVersion"

$cpu = $null
try { $cpu = Get-CimInstance -ClassName Win32_Processor | Select-Object -First 1 } catch { }
$cpuName = if ($null -ne $cpu) { ($cpu.Name -replace '\s+', ' ').Trim() } else { 'unknown' }
$cpuCores = if ($null -ne $cpu) { $cpu.NumberOfCores } else { 0 }
$cpuThreads = if ($null -ne $cpu) { $cpu.NumberOfLogicalProcessors } else { 0 }
Write-Info "CPU: $cpuName ($cpuCores コア / $cpuThreads スレッド)"

$cs = $null
try { $cs = Get-CimInstance -ClassName Win32_ComputerSystem } catch { }
$ramGB = if ($null -ne $cs) { [math]::Round($cs.TotalPhysicalMemory / 1GB, 1) } else { 0 }
Write-Info "メモリ: $ramGB GB"

$gpus = @(Get-GpuInfo)
foreach ($gpu in $gpus) {
    $vramText = if ($null -ne $gpu.VramTotalMB) { "$([math]::Round($gpu.VramTotalMB / 1024, 1)) GB" } else { '不明' }
    Write-Info "GPU: $($gpu.Name) / VRAM $vramText / driver $($gpu.Driver)"
}

$freeGB = Get-FreeSpaceGB -Path $repoRoot
Write-Info "空きディスク ($repoRoot): $freeGB GB"

# ---------------------------------------------------------------- 音声デバイス
Write-Step '音声入出力デバイス'
$devices = @(Get-AudioDevices)
$inputDevices = @($devices | Where-Object { $_.DeviceType -eq 'input' })
$outputDevices = @($devices | Where-Object { $_.DeviceType -eq 'output' })
if ($devices.Count -eq 0) {
    Write-Warn '音声デバイスを列挙できませんでした（Get-PnpDevice が利用できない可能性）'
    $result.findings += [pscustomobject]@{ level = 'warn'; message = '音声デバイスを列挙できませんでした' }
} else {
    foreach ($d in $inputDevices) { Write-Info "入力: $($d.Name)" }
    foreach ($d in $outputDevices) { Write-Info "出力: $($d.Name)" }
    if ($inputDevices.Count -eq 0) {
        Write-Warn '入力デバイスが見つかりません。マイク接続を確認してください。'
        $result.findings += [pscustomobject]@{ level = 'warn'; message = '入力デバイスが 0 件' }
    }
    if ($outputDevices.Count -eq 0) {
        Write-Warn '出力デバイスが見つかりません。ヘッドホン等の接続を確認してください。'
        $result.findings += [pscustomobject]@{ level = 'warn'; message = '出力デバイスが 0 件' }
    }
}

# ---------------------------------------------------------------- VC Client
Write-Step 'VC Client の配置'
$vcInfo = [ordered]@{
    root       = $VcClientRoot
    exists     = $false
    launcher   = $null
    port       = 18000
    portOpen   = $false
    edition    = ''
    version    = ''
    hasCuda    = $false
    hasOnnx    = $false
    modelCount = 0
}
if ($null -ne $config) {
    $vcInfo.port = [int](Get-SettingValue -Object $config -Path 'vcClient.port' -Default 18000)
    if ([string]::IsNullOrWhiteSpace($VcClientRoot)) {
        $vcInfo.root = [string](Get-SettingValue -Object $config -Path 'vcClient.root' -Default '')
    }
}

if ([string]::IsNullOrWhiteSpace($vcInfo.root)) {
    Write-Warn 'vcClient.root が未設定です。config/settings.json を編集してください。'
    $result.findings += [pscustomobject]@{ level = 'warn'; message = 'vcClient.root が未設定' }
} elseif (-not (Test-Path -LiteralPath $vcInfo.root)) {
    Write-Err "VC Client フォルダが見つかりません: $($vcInfo.root)"
    $result.findings += [pscustomobject]@{ level = 'error'; message = 'VC Client フォルダが存在しない' }
} else {
    $vcInfo.exists = $true
    Write-Ok "VC Client フォルダ: $($vcInfo.root)"

    $launcher = Get-ChildItem -LiteralPath $vcInfo.root -Filter 'start_http.bat' -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -ne $launcher) {
        $vcInfo.launcher = $launcher.FullName
        Write-Ok 'start_http.bat を確認'
    } else {
        Write-Warn 'start_http.bat が見つかりません。解凍フォルダが正しいか確認してください。'
        $result.findings += [pscustomobject]@{ level = 'warn'; message = 'start_http.bat が見つからない' }
    }

    # エディション / バージョン: web_front\assets\gui_settings の定義ファイルが一次情報
    $guiSettings = Join-Path $vcInfo.root 'web_front\assets\gui_settings'
    $editionFile = Join-Path $guiSettings 'edition.txt'
    $versionFile = Join-Path $guiSettings 'version.txt'
    $rootLeaf = Split-Path -Leaf $vcInfo.root

    if (Test-Path -LiteralPath $editionFile) {
        $vcInfo.edition = ([string](Get-Content -LiteralPath $editionFile -Raw -ErrorAction SilentlyContinue)).Trim()
    }
    if (Test-Path -LiteralPath $versionFile) {
        $vcInfo.version = ([string](Get-Content -LiteralPath $versionFile -Raw -ErrorAction SilentlyContinue)).Trim()
    }
    # 定義ファイルが無い場合のみフォルダ名から推定する
    if ([string]::IsNullOrWhiteSpace($vcInfo.edition)) {
        if ($rootLeaf -match '(?i)cuda') { $vcInfo.edition = 'cuda' }
        elseif ($rootLeaf -match '(?i)dml|directml') { $vcInfo.edition = 'dml' }
        elseif ($rootLeaf -match '(?i)onnx') { $vcInfo.edition = 'onnx' }
    }
    if ([string]::IsNullOrWhiteSpace($vcInfo.version) -and $rootLeaf -match '(\d+\.\d+\.\d+[-\w.]*)') {
        $vcInfo.version = $Matches[1]
    }

    $vcInfo.hasCuda = ($vcInfo.edition -match '(?i)cuda')
    $vcInfo.hasOnnx = ($vcInfo.edition -match '(?i)onnx|dml')

    if ($vcInfo.hasCuda -or $vcInfo.hasOnnx) {
        Write-Ok "エディション: $($vcInfo.edition) / バージョン: $($vcInfo.version) — RVC 対応"
    } elseif ($vcInfo.edition -match '(?i)std') {
        Write-Err "エディション '$($vcInfo.edition)' は RVC 非対応です。cuda 版か dml 版を導入してください。"
        $result.findings += [pscustomobject]@{ level = 'error'; message = "RVC 非対応エディション: $($vcInfo.edition)" }
    } elseif ([string]::IsNullOrWhiteSpace($vcInfo.edition)) {
        Write-Warn 'エディションを判定できませんでした。cuda 版か dml 版かを確認してください。'
        $result.findings += [pscustomobject]@{ level = 'warn'; message = 'エディション判定不可（RVC 対応版か要確認）' }
    } else {
        Write-Warn "エディション: $($vcInfo.edition)"
    }

    $pth = @(Get-ChildItem -LiteralPath $vcInfo.root -Filter '*.pth' -Recurse -ErrorAction SilentlyContinue)
    $vcInfo.modelCount = $pth.Count
    if ($pth.Count -gt 0) { Write-Info "VC Client 配下の .pth: $($pth.Count) 件" }
}

$vcInfo.portOpen = Test-PortOpen -Port $vcInfo.port -TimeoutMilliseconds 1000
if ($vcInfo.portOpen) {
    Write-Ok "ポート $($vcInfo.port) は応答あり（VC Client 起動中）"

    # 起動中なら API と必須モジュールの状態も確認する
    try {
        $hello = Invoke-RestMethod -Uri "http://localhost:$($vcInfo.port)/api/hello" -TimeoutSec 8
        Write-Ok "API 応答: $($hello.message)"
        $vcInfo.apiOk = $true
    } catch {
        Write-Warn "API が応答しません: $($_.Exception.Message)"
        $vcInfo.apiOk = $false
    }
    try {
        $mods = Invoke-RestMethod -Uri "http://localhost:$($vcInfo.port)/api/module-manager/modules" -TimeoutSec 8
        $mods = @($mods)
        $validCount = @($mods | Where-Object { $_.valid }).Count
        $vcInfo.moduleTotal = $mods.Count
        $vcInfo.moduleValid = $validCount
        if ($mods.Count -gt 0 -and $validCount -lt $mods.Count) {
            Write-Warn "必須モジュール: $validCount/$($mods.Count) 件のみ有効（RVC は動作しません）"
            Write-Info 'ダウンロードが途中で止まっています。docs/03-troubleshooting.md の手順で再取得してください。'
            $result.findings += [pscustomobject]@{ level = 'warn'; message = "必須モジュール未完了 ($validCount/$($mods.Count))" }
        } elseif ($mods.Count -gt 0) {
            Write-Ok "必須モジュール: $validCount/$($mods.Count) 件すべて有効"
        }
    } catch {
        Write-Info 'モジュール状態を取得できませんでした（未対応のバージョンの可能性）'
    }
} else {
    Write-Info "ポート $($vcInfo.port) は閉じています（未起動）"
}

# ---------------------------------------------------------------- 判定
Write-Step 'Phase 0 判定'
$hasNvidia = @($gpus | Where-Object { $_.Name -match '(?i)nvidia' }).Count -gt 0
$vramGB = $null
$nvidia = $gpus | Where-Object { $_.Name -match '(?i)nvidia' } | Select-Object -First 1
if ($null -ne $nvidia -and $null -ne $nvidia.VramTotalMB) { $vramGB = [math]::Round($nvidia.VramTotalMB / 1024, 1) }

if ($hasNvidia) { Write-Ok 'NVIDIA GPU を検出（cuda 版が利用可能）' }
else {
    Write-Warn 'NVIDIA GPU を検出できませんでした。onnx(directml) 版または CPU 動作を検討してください。'
    $result.findings += [pscustomobject]@{ level = 'warn'; message = 'NVIDIA GPU 未検出' }
}

if ($null -ne $vramGB -and $vramGB -lt 4) {
    Write-Warn "VRAM が $vramGB GB です。RVC のリアルタイム変換には余裕がない可能性があります。"
    $result.findings += [pscustomobject]@{ level = 'warn'; message = "VRAM 不足の可能性 ($vramGB GB)" }
}

if ($ramGB -lt 8) {
    Write-Warn "メモリが $ramGB GB です。8 GB 以上を推奨します。"
} else {
    Write-Ok "メモリ $ramGB GB"
}

Write-Info 'マイク・ヘッドホンの実際の入力可否は、VC Client 起動後に GUI の input / output で確認してください。'

# ---------------------------------------------------------------- レポート出力
$result.vcClient = $vcInfo

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add('# 環境確認レポート')
$lines.Add('')
$lines.Add("- 生成日時: $($timestamp.ToString('yyyy-MM-dd HH:mm:ss'))")
$lines.Add("- ホスト名: ``$env:COMPUTERNAME``")
$lines.Add('')
$lines.Add('## OS / ハードウェア')
$lines.Add('')
$lines.Add('| 項目 | 値 |')
$lines.Add('| :--- | :--- |')
$lines.Add("| OS | $osName |")
$lines.Add("| OS バージョン | $osVersion |")
$lines.Add("| CPU | $cpuName |")
$lines.Add("| コア / スレッド | $cpuCores / $cpuThreads |")
$lines.Add("| メモリ | $ramGB GB |")
$lines.Add("| 空きディスク | $freeGB GB |")
$lines.Add('')
$lines.Add('## GPU')
$lines.Add('')
if ($gpus.Count -eq 0) {
    $lines.Add('GPU を検出できませんでした。')
} else {
    $lines.Add('| # | 名前 | VRAM | ドライバ | 取得元 |')
    $lines.Add('| :--- | :--- | :--- | :--- | :--- |')
    foreach ($gpu in $gpus) {
        $vramText = if ($null -ne $gpu.VramTotalMB) { "$([math]::Round($gpu.VramTotalMB / 1024, 1)) GB" } else { '不明' }
        $lines.Add("| $(if ($null -ne $gpu.Index) { $gpu.Index } else { '-' }) | $($gpu.Name) | $vramText | $($gpu.Driver) | $($gpu.Source) |")
    }
    $lines.Add('')
    $lines.Add('> `Win32_VideoController` の VRAM は 32 bit 制限で 4 GB 超が正確に出ない場合がある。`nvidia-smi` の値を優先する。')
}
$lines.Add('')
$lines.Add('## 音声デバイス')
$lines.Add('')
if ($devices.Count -eq 0) {
    $lines.Add('デバイスを列挙できませんでした。Windows の「サウンド設定」で確認してください。')
} else {
    $lines.Add('### 入力（録音）')
    $lines.Add('')
    if ($inputDevices.Count -eq 0) { $lines.Add('- なし') }
    foreach ($d in $inputDevices) { $lines.Add("- $($d.Name)") }
    $lines.Add('')
    $lines.Add('### 出力（再生）')
    $lines.Add('')
    if ($outputDevices.Count -eq 0) { $lines.Add('- なし') }
    foreach ($d in $outputDevices) { $lines.Add("- $($d.Name)") }
}
$lines.Add('')
$lines.Add('## VC Client')
$lines.Add('')
$lines.Add('| 項目 | 値 |')
$lines.Add('| :--- | :--- |')
$lines.Add("| 配置フォルダ | ``$($vcInfo.root)`` |")
$lines.Add("| フォルダ存在 | $($vcInfo.exists) |")
$lines.Add("| 起動スクリプト | $(if ($vcInfo.launcher) { 'start_http.bat あり' } else { '未検出' }) |")
$lines.Add("| エディション（推定） | $(if ($vcInfo.edition) { $vcInfo.edition } else { '不明' }) |")
$lines.Add("| バージョン（推定） | $(if ($vcInfo.version) { $vcInfo.version } else { '不明' }) |")
$lines.Add("| ポート $($vcInfo.port) | $(if ($vcInfo.portOpen) { '応答あり' } else { '閉' }) |")
$lines.Add("| 配下の .pth | $($vcInfo.modelCount) 件 |")
$lines.Add('')
$lines.Add('> エディションとバージョンはフォルダ名からの推定。**実際の配布名は GUI の Server Info で確認し、確定値を docs/00-phase0-environment.md に記録する。**')
$lines.Add('')
$lines.Add('## 検出した注意点')
$lines.Add('')
$findingList = @($result.findings)
if ($findingList.Count -eq 0) {
    $lines.Add('- なし')
} else {
    foreach ($f in $findingList) { $lines.Add("- [$($f.level)] $($f.message)") }
}
$lines.Add('')
$lines.Add('## 次の作業（Phase 0 完了条件）')
$lines.Add('')
$lines.Add('- [ ] 機材台帳（マイク / オーディオ IF / ヘッドホン / PC）を確定した')
$lines.Add('- [ ] VC Client の採用エディション・バージョンを確定し、ダウンロード元を記録した')
$lines.Add('- [ ] 使用する RVC モデル 1 種類を確定し、利用規約を記録した（docs/05-models.md）')
$lines.Add('- [ ] 上記を docs/00-phase0-environment.md に転記した')

$reportPath = Join-Path $OutputDir ("environment-$stamp.md")
$latestPath = Join-Path $OutputDir 'environment-latest.md'
$jsonPath = Join-Path $OutputDir ("environment-$stamp.json")
$lines -join "`r`n" | Set-Content -LiteralPath $reportPath -Encoding UTF8
$lines -join "`r`n" | Set-Content -LiteralPath $latestPath -Encoding UTF8
$result | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding UTF8

Write-Step '出力'
Write-Ok "レポート: $reportPath"
Write-Ok "最新版  : $latestPath"
Write-Ok "JSON    : $jsonPath"
Write-Host ''
Write-Host '次は docs/00-phase0-environment.md の台帳に転記してください。' -ForegroundColor Cyan
