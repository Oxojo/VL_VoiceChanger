<#
.SYNOPSIS
  プロファイル（config/profiles/*.json）の設定を VC Client に適用する。

.DESCRIPTION
  VC Client 2.1.4-alpha の REST API（PUT /api/configuration-manager/configuration）を使い、
  以下を一括で反映する。

    - 使用モデル（スロット）
    - 入力モード（server / client）
    - 入出力デバイス（名前の部分一致で解決）
    - GPU（device_id）
    - ノイズゲート / EXTRA（extra_frame_sec）

  スロット固有の TUNE（pitch_shift）・INDEX（index_ratio）・CHUNK・F0 Detector は
  このバージョンの REST API では変更できない（PUT /api/slot-manager/slots/{index} は
  ボディを受け付けず、再読込のみ）。これらは GUI で設定する。

.PARAMETER Profile
  プロファイル名（character-a）または JSON のパス。

.PARAMETER DryRun
  適用せず、変更予定の内容だけを表示する。

.EXAMPLE
  .\scripts\Set-VoiceProfile.ps1 character-a -DryRun
  .\scripts\Set-VoiceProfile.ps1 character-a
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$Profile,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\Common.ps1')

$repoRoot = Get-RepoRoot
$config = Read-JsonFile -Path (Join-Path $repoRoot 'config\settings.json')
$port = [int](Get-SettingValue -Object $config -Path 'vcClient.port' -Default 18000)
$base = "http://localhost:$port"

Write-Host ''
Write-Host ' VL_VoiceChanger / プロファイル適用' -ForegroundColor White

# ---------------------------------------------------------------- プロファイル
Write-Step '1. プロファイルの読み込み'
if (Test-Path -LiteralPath $Profile) {
    $profilePath = (Resolve-Path -LiteralPath $Profile).Path
} else {
    $candidate = Join-Path $repoRoot ("config\profiles\{0}.json" -f $Profile)
    if (-not (Test-Path -LiteralPath $candidate)) {
        Write-Err "プロファイルが見つかりません: $Profile"
        exit 1
    }
    $profilePath = $candidate
}
$prof = Read-JsonFile -Path $profilePath
Write-Ok "$($prof.displayName) ($($prof.id))"
Write-Info $profilePath

if (-not (Test-PortOpen -Port $port -TimeoutMilliseconds 1000)) {
    Write-Err "ポート $port が開いていません。先に scripts\Start-VoiceChanger.ps1 を実行してください。"
    exit 1
}

# ---------------------------------------------------------------- 現在値
Write-Step '2. 現在の設定を取得'
$current = Invoke-VcApi -Uri "$base/api/configuration-manager/configuration" -TimeoutSec 20
$slots = Invoke-VcApi -Uri "$base/api/slot-manager/slots" -TimeoutSec 30
$slots = @($slots)
$inputs = Invoke-VcApi -Uri "$base/api/audio-device-manager/input_devices" -TimeoutSec 20
$inputs = @($inputs)
$outputs = Invoke-VcApi -Uri "$base/api/audio-device-manager/output_devices" -TimeoutSec 20
$outputs = @($outputs)
Write-Info "current_slot_index = $($current.current_slot_index) / mode = $($current.voice_changer_input_mode)"

function Resolve-DeviceIndex {
    param($Devices, [string]$Query, [string]$Label)
    if ([string]::IsNullOrWhiteSpace($Query)) { return -1 }   # -1 = 変更しない
    $hits = @($Devices | Where-Object { $_.name -like "*$Query*" })
    if ($hits.Count -eq 0) {
        Write-Warn "$Label に一致するデバイスがありません: '$Query'（変更しません）"
        return -1
    }
    if ($hits.Count -gt 1) {
        Write-Warn "$Label の候補が $($hits.Count) 件あります。先頭を使用します（ホスト API を名前に含めると絞れます）:"
        $hits | Select-Object -First 5 | ForEach-Object {
            Write-Host ("       index {0,3} [{1}] {2}" -f $_.index, $_.host_api, $_.name) -ForegroundColor DarkYellow
        }
    }
    return [int]$hits[0].index
}

# ---------------------------------------------------------------- 適用内容
Write-Step '3. 適用内容'
$changes = [ordered]@{}

# モデル（スロット）
$modelName = ''
if ($null -ne $prof.model -and -not [string]::IsNullOrWhiteSpace($prof.model.pth)) {
    $modelName = [System.IO.Path]::GetFileName([string]$prof.model.pth)
}
if (-not [string]::IsNullOrWhiteSpace($modelName)) {
    $match = @($slots | Where-Object { (Get-ObjectProperty -Object $_ -Name 'model_file' -Default '') -eq $modelName })
    if ($match.Count -eq 0) {
        Write-Warn "$modelName は未登録です。scripts\Register-Model.ps1 で登録してください（スロットは変更しません）。"
    } elseif ($match.Count -gt 1) {
        Write-Warn "$modelName が複数スロットに登録されています。先頭を使用します。"
        $changes['current_slot_index'] = [int](Get-ObjectProperty -Object $match[0] -Name 'slot_index' -Default 0)
    } else {
        $changes['current_slot_index'] = [int](Get-ObjectProperty -Object $match[0] -Name 'slot_index' -Default 0)
    }
}

# 入力モード
$mode = [string](Get-SettingValue -Object $prof -Path 'audio.mode' -Default '')
if ($mode -in @('server', 'client')) {
    $changes['voice_changer_input_mode'] = $mode
}

# デバイス（-1 は「変更しない」）
$inIdx = Resolve-DeviceIndex -Devices $inputs -Query ([string](Get-SettingValue -Object $prof -Path 'audio.inputDevice' -Default '')) -Label '入力'
$outIdx = Resolve-DeviceIndex -Devices $outputs -Query ([string](Get-SettingValue -Object $prof -Path 'audio.outputDevice' -Default '')) -Label '出力'
if ($inIdx -ge 0) { $changes['audio_input_device_index'] = $inIdx }
if ($outIdx -ge 0) { $changes['audio_output_device_index'] = $outIdx }

# GPU
$gpuId = Get-SettingValue -Object $prof -Path 'gpu.deviceId' -Default $null
if ($null -ne $gpuId) { $changes['gpu_device_id_int'] = [int]$gpuId }

# 変換パラメータ（REST で変更できるものだけ）
$noiseGate = Get-SettingValue -Object $prof -Path 'convert.noiseGateDb' -Default $null
if ($null -ne $noiseGate) { $changes['noise_gate'] = [double]$noiseGate }
$extra = Get-SettingValue -Object $prof -Path 'convert.extra' -Default $null
if ($null -ne $extra -and [double]$extra -gt 0) {
    $changes['extra_frame_sec'] = [math]::Round(([double]$extra / 48000.0), 4)
}

if ($changes.Count -eq 0) {
    Write-Warn '適用できる項目がありません。プロファイルの audio.inputDevice / outputDevice などを確認してください。'
    exit 0
}

foreach ($key in $changes.Keys) {
    Write-Host ("   {0,-30} {1} -> {2}" -f $key, $current.$key, $changes[$key]) -ForegroundColor Gray
}

# REST で変更できない項目の案内
$manual = @()
if ($null -ne $prof.convert) {
    if ($null -ne $prof.convert.pitch) { $manual += "TUNE（pitch_shift）= $($prof.convert.pitch)" }
    if ($null -ne $prof.convert.indexRatio) { $manual += "INDEX（index_ratio）= $($prof.convert.indexRatio)" }
    if ($null -ne $prof.convert.f0Detector) { $manual += "F0 Det = $($prof.convert.f0Detector)" }
    if ($null -ne $prof.convert.chunk) { $manual += "CHUNK = $($prof.convert.chunk)" }
}
if ($manual.Count -gt 0) {
    Write-Host ''
    Write-Warn '次の値は REST API では変更できません（GUI で設定してください）:'
    $manual | ForEach-Object { Write-Host "       - $_" -ForegroundColor DarkYellow }
}

if ($DryRun) {
    Write-Host ''
    Write-Info 'DryRun のため適用していません。'
    exit 0
}

# ---------------------------------------------------------------- 適用
Write-Step '4. 適用'
$payload = $current
foreach ($key in $changes.Keys) {
    $payload | Add-Member -NotePropertyName $key -NotePropertyValue $changes[$key] -Force
}
try {
    Invoke-VcApi -Uri "$base/api/configuration-manager/configuration" -Method Put -Body $payload -TimeoutSec 120 | Out-Null
} catch {
    Write-Err "適用に失敗しました: $($_.Exception.Message)"
    if ($null -ne $_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        Write-Host $reader.ReadToEnd() -ForegroundColor DarkGray
    }
    exit 1
}

Start-Sleep -Milliseconds 500
$after = Invoke-VcApi -Uri "$base/api/configuration-manager/configuration" -TimeoutSec 20
Write-Step '5. 適用結果'
foreach ($key in $changes.Keys) {
    $ok = ($after.$key -eq $changes[$key])
    if ($ok) { Write-Ok "$key = $($after.$key)" } else { Write-Err "$key = $($after.$key)（期待 $($changes[$key])）" }
}

Write-Host ''
Write-Info 'GUI で [start] を押すと変換が始まります（REST では開始できません）。'
