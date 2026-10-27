<#
.SYNOPSIS
    VC Client をプロファイルに沿って起動し、GUI で設定する項目を表示する。

.DESCRIPTION
    設計書 6 章 Phase 1〜2 の「起動」を補助する。
    VC Client v2 系は変換設定を外部から API で流し込む手段が公開されていないため、
    このスクリプトは以下を担当する。

      1. プロファイル（config/profiles/*.json）の妥当性検証
      2. モデル / index ファイルの存在確認
      3. プロファイルに記録した入出力デバイスが現在も存在するか確認
      4. VC Client の起動と、GUI が応答するまでの待機
      5. GUI で設定する値のチェックリスト表示

    実際の設定反映は GUI で行い、結果をプロファイルに記録する運用とする。

.PARAMETER Profile
    プロファイル名（character-a）またはファイルパス。省略時は character-a。

.PARAMETER ConfigPath
    config/settings.json のパス。省略時はリポジトリ既定。

.PARAMETER DryRun
    起動せず、検証とチェックリスト表示のみ行う。

.PARAMETER TimeoutSeconds
    GUI の応答待ち時間。省略時は settings.json の vcClient.startupTimeoutSeconds。

.EXAMPLE
    .\scripts\Start-VoiceChanger.ps1 -Profile character-a

.EXAMPLE
    .\scripts\Start-VoiceChanger.ps1 -Profile character-b -DryRun
#>
[CmdletBinding()]
param(
    [string]$Profile = 'character-a',
    [string]$ConfigPath = '',
    [switch]$DryRun,
    [int]$TimeoutSeconds = 0
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\Common.ps1')

$repoRoot = Get-RepoRoot
if ([string]::IsNullOrWhiteSpace($ConfigPath)) { $ConfigPath = Join-Path $repoRoot 'config\settings.json' }
try {
    $config = Read-JsonFile -Path $ConfigPath
} catch {
    Write-Err $_.Exception.Message
    Write-Info 'config/settings.json の書式（JSON）を確認してください。'
    exit 1
}

$vcRoot = Resolve-RepoPath -Path ([string](Get-SettingValue -Object $config -Path 'vcClient.root' -Default ''))
$launcherName = [string](Get-SettingValue -Object $config -Path 'vcClient.launcher' -Default 'start_http.bat')
$port = [int](Get-SettingValue -Object $config -Path 'vcClient.port' -Default 18888)
if ($TimeoutSeconds -le 0) {
    $TimeoutSeconds = [int](Get-SettingValue -Object $config -Path 'vcClient.startupTimeoutSeconds' -Default 180)
}
$profilesDir = Resolve-RepoPath -Path ([string](Get-SettingValue -Object $config -Path 'paths.profiles' -Default 'config/profiles'))

# ---------------------------------------------------------------- プロファイル解決
$profilePath = $Profile
if (-not (Test-Path -LiteralPath $profilePath)) {
    if (-not $profilePath.EndsWith('.json')) { $profilePath = "$Profile.json" }
    $profilePath = Join-Path $profilesDir $profilePath
}
if (-not (Test-Path -LiteralPath $profilePath)) {
    Write-Err "プロファイルが見つかりません: $Profile"
    Write-Info "探した場所: $(Join-Path $profilesDir $Profile)"
    Write-Info "利用可能なプロファイル:"
    Get-ChildItem -LiteralPath $profilesDir -Filter '*.json' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike '_*' } |
        ForEach-Object { Write-Host "    - $($_.BaseName)" -ForegroundColor Gray }
    exit 1
}

$prof = Read-JsonFile -Path $profilePath
$profId = [string](Get-SettingValue -Object $prof -Path 'id' -Default ([System.IO.Path]::GetFileNameWithoutExtension($profilePath)))
$profName = [string](Get-SettingValue -Object $prof -Path 'displayName' -Default $profId)

Write-Host ''
Write-Host ' VL_VoiceChanger / 起動補助' -ForegroundColor White
Write-Host " プロファイル: $profName ($profId)" -ForegroundColor Gray
Write-Host " ファイル    : $profilePath" -ForegroundColor DarkGray

# ---------------------------------------------------------------- 検証
Write-Step '1. 構成の検証'
$errors = New-Object System.Collections.Generic.List[string]
$warnings = New-Object System.Collections.Generic.List[string]

if ([string]::IsNullOrWhiteSpace($vcRoot)) {
    $errors.Add('vcClient.root が未設定です。config/settings.json を編集してください。')
} elseif (-not (Test-Path -LiteralPath $vcRoot)) {
    $errors.Add("VC Client フォルダが存在しません: $vcRoot")
} else {
    Write-Ok "VC Client: $vcRoot"
    $launcher = Join-Path $vcRoot $launcherName
    if (-not (Test-Path -LiteralPath $launcher)) {
        $errors.Add("起動スクリプトが見つかりません: $launcher")
    } else {
        Write-Ok "起動スクリプト: $launcherName"
    }
}

$modelPth = [string](Get-SettingValue -Object $prof -Path 'model.pth' -Default '')
$modelIndex = [string](Get-SettingValue -Object $prof -Path 'model.index' -Default '')
if ([string]::IsNullOrWhiteSpace($modelPth)) {
    $warnings.Add('model.pth が未設定です（モデル未選定）。')
} else {
    $pthPath = Resolve-RepoPath -Path $modelPth
    if (Test-Path -LiteralPath $pthPath) {
        Write-Ok "モデル: $modelPth"
    } else {
        $warnings.Add("モデルファイルが見つかりません: $modelPth （GUI でモデルスロットから登録する運用でも可）")
    }
}
if (-not [string]::IsNullOrWhiteSpace($modelIndex)) {
    $idxPath = Resolve-RepoPath -Path $modelIndex
    if (Test-Path -LiteralPath $idxPath) { Write-Ok "index: $modelIndex" }
    else { $warnings.Add("index ファイルが見つかりません: $modelIndex") }
}

# 入出力デバイスの突き合わせ
$audioDevices = @(Get-AudioDevices)
$knownNames = @($audioDevices | ForEach-Object { $_.Name })
$inDev = [string](Get-SettingValue -Object $prof -Path 'audio.inputDevice' -Default '')
$outDev = [string](Get-SettingValue -Object $prof -Path 'audio.outputDevice' -Default '')
$monDev = [string](Get-SettingValue -Object $prof -Path 'audio.monitorDevice' -Default '')

foreach ($pair in @(
        @{ Label = '入力'; Value = $inDev },
        @{ Label = '出力'; Value = $outDev },
        @{ Label = 'モニター'; Value = $monDev })) {
    if ([string]::IsNullOrWhiteSpace([string]$pair.Value)) { continue }
    $hit = $knownNames | Where-Object { $_ -eq $pair.Value } | Select-Object -First 1
    if ($null -eq $hit) {
        $near = $knownNames | Where-Object { $_ -like "*$($pair.Value)*" } | Select-Object -First 1
        if ($null -ne $near) {
            $warnings.Add("$($pair.Label)デバイス名が完全一致しません: '$($pair.Value)' → 候補 '$near'")
        } else {
            $warnings.Add("$($pair.Label)デバイスが見つかりません: '$($pair.Value)'")
        }
    } else {
        Write-Ok "$($pair.Label)デバイス: $($pair.Value)"
    }
}

foreach ($w in $warnings) { Write-Warn $w }
foreach ($e in $errors) { Write-Err $e }
if ($errors.Count -gt 0) {
    Write-Host ''
    Write-Host ' 起動条件を満たしていません。上記を解消してください。' -ForegroundColor Red
    exit 1
}

# ---------------------------------------------------------------- 起動
Write-Step '2. VC Client 起動'
$alreadyRunning = Test-PortOpen -Port $port -TimeoutMilliseconds 800
if ($alreadyRunning) {
    Write-Ok "ポート $port は既に応答しています（起動済みと判断し、二重起動しません）"
} elseif ($DryRun) {
    Write-Info 'DryRun のため起動しません。'
    Write-Info "実行時のコマンド: `"$launcher`" (作業ディレクトリ: $vcRoot)"
} else {
    Write-Info "起動: $launcherName"
    Start-Process -FilePath (Join-Path $vcRoot $launcherName) -WorkingDirectory $vcRoot -WindowStyle Minimized
    Write-Info "GUI の応答を待機中（最大 $TimeoutSeconds 秒）。初回起動はモデル等のダウンロードで時間がかかります。"
    if (Wait-Port -Port $port -TimeoutSeconds $TimeoutSeconds) {
        Write-Ok "GUI が応答しました: http://localhost:$port/"
    } else {
        Write-Err "時間内にポート $port が開きませんでした。"
        Write-Info 'docs/03-troubleshooting.md の「GUI が開かない」を確認してください。'
        exit 1
    }
}

# ---------------------------------------------------------------- 設定チェックリスト
$mode = [string](Get-SettingValue -Object $prof -Path 'audio.mode' -Default 'server')
$pitch = Get-SettingValue -Object $prof -Path 'convert.pitch' -Default 0
$indexRatio = Get-SettingValue -Object $prof -Path 'convert.indexRatio' -Default 0
$f0 = [string](Get-SettingValue -Object $prof -Path 'convert.f0Detector' -Default 'rmvpe')
$chunk = Get-SettingValue -Object $prof -Path 'convert.chunk' -Default 128
$extra = Get-SettingValue -Object $prof -Path 'convert.extra' -Default 8192
$noiseGate = Get-SettingValue -Object $prof -Path 'convert.noiseGateDb' -Default -60
$gainIn = Get-SettingValue -Object $prof -Path 'convert.gainIn' -Default 0
$gainOut = Get-SettingValue -Object $prof -Path 'convert.gainOut' -Default 0
$volume = Get-SettingValue -Object $prof -Path 'convert.volume' -Default 1.0
$gpuId = Get-SettingValue -Object $prof -Path 'gpu.deviceId' -Default 0

Write-Step '3. GUI で設定する値'
Write-Host ''
Write-Host '  --- モデル ---' -ForegroundColor DarkCyan
Write-Host ("   モデル選択エリア        : {0}" -f $profName)
Write-Host ("   モデルファイル (RVC)    : {0}" -f $(if ($modelPth) { $modelPth } else { '（未設定）' }))
Write-Host ("   index                   : {0}" -f $(if ($modelIndex) { $modelIndex } else { '（なし）' }))
Write-Host ''
Write-Host '  --- 音声入出力 (詳細設定エリア) ---' -ForegroundColor DarkCyan
Write-Host ("   AUDIO                   : {0}{1}" -f $mode, $(if ($mode -eq 'server') { '  ※VC Client がデバイスを直接使用' } else { '  ※ブラウザ経由' }))
Write-Host ("   input                   : {0}" -f $(if ($inDev) { $inDev } else { '（未設定）' }))
Write-Host ("   output                  : {0}" -f $(if ($outDev) { $outDev } else { '（未設定）' }))
if (-not [string]::IsNullOrWhiteSpace($monDev)) {
    Write-Host ("   monitor                 : {0}" -f $monDev)
}
Write-Host ''
Write-Host '  --- 変換設定 ---' -ForegroundColor DarkCyan
Write-Host ("   TUNE (ピッチ)           : {0}" -f $pitch)
Write-Host ("   INDEX (比率)            : {0}" -f $indexRatio)
Write-Host ("   F0 Det                  : {0}" -f $f0)
Write-Host ("   CHUNK                   : {0}" -f $chunk)
Write-Host ("   EXTRA                   : {0}" -f $extra)
Write-Host ("   S. Thresh (ノイズゲート): {0}" -f $noiseGate)
Write-Host ("   GAIN in / out           : {0} / {1}" -f $gainIn, $gainOut)
Write-Host ("   vol                     : {0}" -f $volume)
Write-Host ("   GPU                     : gpu{0}" -f $gpuId)
Write-Host ''
Write-Host '  --- 起動後の確認 ---' -ForegroundColor DarkCyan
Write-Host '   1. [start] を押し、マイクに向かって発話する'
Write-Host '   2. 変換後の声が output デバイスから聞こえることを確認する'
Write-Host '   3. buf > res になっているか確認する（buf が短いと音切れしやすい）'
Write-Host '   4. 二重に聞こえる場合はダイレクトモニターを切る（docs/02-operation.md）'
Write-Host '   5. 調整した値は config/profiles の JSON に反映する'
Write-Host ''
if ($DryRun) {
    Write-Host ' [DryRun] ここまで。実際の起動は -DryRun を外して実行してください。' -ForegroundColor Yellow
} else {
    Write-Host " GUI: http://localhost:$port/" -ForegroundColor Green
}
