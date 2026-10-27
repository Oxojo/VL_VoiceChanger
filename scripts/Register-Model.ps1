<#
.SYNOPSIS
  VC Client に RVC モデル（.pth / .onnx）を登録する。

.DESCRIPTION
  VC Client 2.1.4-alpha の REST API を使い、GUI の「モデルをアップロード」と同じ処理を行う。
  実体は vcClient.root\upload_dir にコピーされ、サーバーが model_dir\<slot_index>\ へ移して
  params.json を生成する（inferencer_type / sample_rate / embedder は .pth から自動判定）。

  モデルの拡張子は必須。サーバーは拡張子で RVC / ONNX を判別するため、
  アップロード用の一時ファイル名に拡張子を残す必要がある。

.PARAMETER ModelPath
  登録するモデルファイル。リポジトリからの相対パス（models/...）か絶対パス。

.PARAMETER Name
  VC Client のモデル一覧に表示する名前。省略時はファイル名（拡張子なし）。

.PARAMETER SlotIndex
  登録先のスロット番号。省略時は空きスロットの最小番号を使う。

.PARAMETER Force
  同じファイル名のモデルが既に登録されていても、二重登録する。

.EXAMPLE
  .\scripts\Register-Model.ps1 models/tsukuyomi/tsukuyomi-normal1.pth -Name 'つくよみちゃん（通常1）'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)][string]$ModelPath,
    [string]$Name = '',
    [int]$SlotIndex = -1,
    [string]$IndexFile = '',
    [string]$TermsOfUseUrl = '',
    [ValidateSet('', 'hubert_base_l9fp', 'hubert_base_l12', 'contentvec', 'hubert_base_japanese_l9fp', 'hubert_base_japanese_l12', 'whisper', 'applio_japanese_hubert_base_l12', 'applio_chinese_hubert_base_l12', 'applio_korean_hubert_base_l12')]
    [string]$Embedder = '',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\Common.ps1')

$repoRoot = Get-RepoRoot
$configPath = Join-Path $repoRoot 'config\settings.json'
$config = Read-JsonFile -Path $configPath
$port = [int](Get-SettingValue -Object $config -Path 'vcClient.port' -Default 18000)
$vcRoot = Resolve-RepoPath -Path ([string](Get-SettingValue -Object $config -Path 'vcClient.root' -Default ''))
$base = "http://localhost:$port"
$uploadDir = Join-Path $vcRoot 'upload_dir'

Write-Host ''
Write-Host ' VL_VoiceChanger / モデル登録' -ForegroundColor White

# ---------------------------------------------------------------- 事前確認
Write-Step '1. VC Client の稼働確認'
if (-not (Test-PortOpen -Port $port -TimeoutMilliseconds 1000)) {
    Write-Err "ポート $port が開いていません。先に scripts\Start-VoiceChanger.ps1 を実行してください。"
    exit 1
}
try {
    $hello = Invoke-VcApi -Uri "$base/api/hello" -TimeoutSec 10
    Write-Ok "API 応答: $($hello.message)"
} catch {
    Write-Err "API が応答しません: $($_.Exception.Message)"
    exit 1
}
if (-not (Test-Path -LiteralPath $vcRoot)) {
    Write-Err "vcClient.root が見つかりません: $vcRoot"
    exit 1
}

# ---------------------------------------------------------------- モデルファイル
Write-Step '2. モデルファイルの確認'
$resolved = Resolve-RepoPath -Path $ModelPath
if (-not (Test-Path -LiteralPath $resolved)) {
    Write-Err "モデルが見つかりません: $resolved"
    exit 1
}
$file = Get-Item -LiteralPath $resolved
$extension = $file.Extension.ToLowerInvariant()
if ($extension -notin @('.pth', '.onnx')) {
    Write-Err "拡張子が .pth / .onnx ではありません: $($file.Name)"
    Write-Info 'VC Client は拡張子でモデル種別を判定するため、拡張子は必須です。'
    exit 1
}
if ([string]::IsNullOrWhiteSpace($Name)) { $Name = $file.BaseName }
Write-Ok "$($file.Name) ($(Format-FileSize -Bytes $file.Length))"

# ---------------------------------------------------------------- 重複確認
Write-Step '3. 登録済みスロットの確認'
$slots = Invoke-VcApi -Uri "$base/api/slot-manager/slots" -TimeoutSec 30
$slots = @($slots)
Write-Info "スロット数: $($slots.Count)"

$same = @($slots | Where-Object { (Get-ObjectProperty -Object $_ -Name 'model_file' -Default '') -eq $file.Name })
if ($same.Count -gt 0 -and -not $Force) {
    Write-Warn "同じモデルが既に登録されています（-Force で二重登録できます）。"
    $same | ForEach-Object {
        Write-Host ("       slot {0}: {1} / {2} / {3} Hz" -f (Get-ObjectProperty $_ 'slot_index'), (Get-ObjectProperty $_ 'name'), (Get-ObjectProperty $_ 'inferencer_type'), (Get-ObjectProperty $_ 'sample_rate')) -ForegroundColor DarkYellow
    }
    exit 0
}

if ($SlotIndex -lt 0) {
    $free = @($slots | Where-Object { [string]::IsNullOrWhiteSpace([string](Get-ObjectProperty -Object $_ -Name 'voice_changer_type' -Default '')) } | Sort-Object { [int](Get-ObjectProperty -Object $_ -Name 'slot_index' -Default -1) })
    if ($free.Count -eq 0) {
        $SlotIndex = ([int]($slots | ForEach-Object { [int](Get-ObjectProperty -Object $_ -Name 'slot_index' -Default -1) } | Measure-Object -Maximum).Maximum) + 1
    } else {
        $SlotIndex = [int](Get-ObjectProperty -Object $free[0] -Name 'slot_index' -Default 0)
    }
    Write-Info "空きスロットを自動選択: $SlotIndex"
} else {
    Write-Info "登録先スロット: $SlotIndex（指定）"
}

# ---------------------------------------------------------------- 取り込み
Write-Step '4. モデルの取り込み'
if (-not (Test-Path -LiteralPath $uploadDir)) {
    New-Item -ItemType Directory -Path $uploadDir -Force | Out-Null
}
$staged = Join-Path $uploadDir $file.Name
Copy-Item -LiteralPath $file.FullName -Destination $staged -Force
Write-Info "upload_dir へコピー: $($file.Name)"

$payload = [ordered]@{
    voice_changer_type = 'RVC'
    name               = $Name
    terms_of_use_url   = $TermsOfUseUrl
    slot_index         = $SlotIndex
    model_file         = $file.Name          # サーバーが upload_dir を前置するためファイル名のみ
    index_file         = $(if ([string]::IsNullOrWhiteSpace($IndexFile)) { $null } else { $IndexFile })
    icon_file          = $null
    embedder           = $(if ([string]::IsNullOrWhiteSpace($Embedder)) { $null } else { $Embedder })
}
$json = $payload | ConvertTo-Json -Depth 5

$sw = [System.Diagnostics.Stopwatch]::StartNew()
try {
    Invoke-VcApi -Uri "$base/api/slot-manager/slots" -Method Post -Body $json -TimeoutSec 600 | Out-Null
} catch {
    Write-Err "取り込みに失敗しました: $($_.Exception.Message)"
    if ($null -ne $_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        Write-Host $reader.ReadToEnd() -ForegroundColor DarkGray
    }
    exit 1
}
Write-Ok "取り込み完了（$([math]::Round($sw.Elapsed.TotalSeconds, 1)) 秒）"

# ---------------------------------------------------------------- 確認
Write-Step '5. 登録結果'
$after = Invoke-VcApi -Uri "$base/api/slot-manager/slots" -TimeoutSec 30
$after = @($after)
$registered = @($after | Where-Object { (Get-ObjectProperty -Object $_ -Name 'slot_index' -Default -1) -eq $SlotIndex -and (Get-ObjectProperty -Object $_ -Name 'model_file' -Default '') -eq $file.Name })
if ($registered.Count -eq 0) {
    Write-Err '登録後のスロットが見つかりません。vcclient.log を確認してください。'
    exit 1
}
$slot = $registered[0]
Write-Host ''
Write-Host "   slot_index    : $(Get-ObjectProperty $slot 'slot_index')" -ForegroundColor Gray
Write-Host "   name          : $(Get-ObjectProperty $slot 'name')" -ForegroundColor Gray
Write-Host "   model_file    : $(Get-ObjectProperty $slot 'model_file')" -ForegroundColor Gray
Write-Host ("   inferencer    : {0} / {1} Hz / f0={2}" -f (Get-ObjectProperty $slot 'inferencer_type'), (Get-ObjectProperty $slot 'sample_rate'), (Get-ObjectProperty $slot 'is_f0')) -ForegroundColor Gray
Write-Host "   embedder      : $(Get-ObjectProperty $slot 'embedder')" -ForegroundColor Gray
Write-Host "   pitch_estimator: $(Get-ObjectProperty $slot 'pitch_estimator')" -ForegroundColor Gray
Write-Ok "スロット $($SlotIndex) に登録されました"

Write-Host ''
Write-Info ("GUI（http://localhost:{0}/）でこのモデルを選択し、[start] で変換を開始してください。" -f $port)
Write-Info 'REST API では変換スレッドを開始できません（GUI の [start] が必要。docs/02-operation.md）。'
