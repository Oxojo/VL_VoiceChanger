<#
.SYNOPSIS
    10 分間の連続使用テストを記録し、MVP 判定用のレポートを生成する。

.DESCRIPTION
    設計書 6 章 Phase 3「安定性検証」に対応する。
    指定時間、VC Client のプロセスと GUI ポート・GPU 使用率を一定間隔で記録し、
    CSV（生ログ）と Markdown（判定用レポート）を出力する。

    実行中に次のキーを押すと、その時刻の事象を記録できる。
      D … 音切れがあった
      S … 変換が止まった / フリーズした
      N … 声が崩れた
      M … 自由記述のメモ
      Q … テストを途中終了する

    なお「音が明瞭に聞こえるか」は機械では判定できない。
    評価担当 2 名による聴取結果は docs/04-verification.md の評価シートに記入する。

.PARAMETER DurationMinutes
    連続使用時間（分）。既定は settings.json の stabilityTest.durationMinutes（10）。

.PARAMETER SampleSeconds
    サンプリング間隔（秒）。既定 5。

.PARAMETER Profile
    レポートに記録するプロファイル名。

.PARAMETER AllowNotRunning
    VC Client が起動していなくてもテストを続行する（動作確認用）。

.EXAMPLE
    .\scripts\Test-Stability.ps1 -Profile character-a

.EXAMPLE
    .\scripts\Test-Stability.ps1 -DurationMinutes 1 -AllowNotRunning
#>
[CmdletBinding()]
param(
    [int]$DurationMinutes = 0,
    [int]$SampleSeconds = 0,
    [string]$Profile = '',
    [string]$ProcessName = '',
    [string]$Url = '',
    [string]$ConfigPath = '',
    [int]$TimeoutSeconds = 5,
    [switch]$AllowNotRunning
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'lib\Common.ps1')

$repoRoot = Get-RepoRoot
if ([string]::IsNullOrWhiteSpace($ConfigPath)) { $ConfigPath = Join-Path $repoRoot 'config\settings.json' }
$config = $null
if (Test-Path -LiteralPath $ConfigPath) {
    try {
        $config = Read-JsonFile -Path $ConfigPath
    } catch {
        Write-Err $_.Exception.Message
        Write-Info 'config/settings.json の書式（JSON）を確認してください。既定値で続行します。'
    }
}

if ($DurationMinutes -le 0) { $DurationMinutes = [int](Get-SettingValue -Object $config -Path 'stabilityTest.durationMinutes' -Default 10) }
if ($SampleSeconds -le 0) { $SampleSeconds = [int](Get-SettingValue -Object $config -Path 'stabilityTest.sampleSeconds' -Default 5) }
if ([string]::IsNullOrWhiteSpace($ProcessName)) {
    $ProcessName = [string](Get-SettingValue -Object $config -Path 'stabilityTest.processName' -Default 'MMVC_Client*,MMVCServerSIO*,vcclient*')
}
$vcRoot = Resolve-RepoPath -Path ([string](Get-SettingValue -Object $config -Path 'vcClient.root' -Default ''))
$port = [int](Get-SettingValue -Object $config -Path 'vcClient.port' -Default 18888)
if ([string]::IsNullOrWhiteSpace($Url)) { $Url = "http://localhost:$port/" }

$reportDir = Resolve-RepoPath -Path ([string](Get-SettingValue -Object $config -Path 'paths.reports' -Default 'reports'))
$logDir = Resolve-RepoPath -Path ([string](Get-SettingValue -Object $config -Path 'paths.logs' -Default 'logs'))
foreach ($d in @($reportDir, $logDir)) {
    if (-not (Test-Path -LiteralPath $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null }
}

function Get-Stats {
    param($Values)
    $v = @($Values | Where-Object { $null -ne $_ })
    if ($v.Count -eq 0) { return $null }
    $m = $v | Measure-Object -Average -Minimum -Maximum
    return [pscustomobject]@{
        Avg = [math]::Round($m.Average, 1)
        Min = [math]::Round($m.Minimum, 1)
        Max = [math]::Round($m.Maximum, 1)
        N   = $v.Count
    }
}

function Get-GpuSample {
    $smi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -eq $smi) { return $null }
    try {
        $line = & $smi.Source --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits 2>$null | Select-Object -First 1
        if ([string]::IsNullOrWhiteSpace($line)) { return $null }
        $f = $line.Split(',').Trim()
        return [pscustomobject]@{ Util = [double]$f[0]; MemMB = [double]$f[1]; TempC = [double]$f[2] }
    } catch {
        return $null
    }
}

# ---------------------------------------------------------------- 事前確認
Write-Host ''
Write-Host ' VL_VoiceChanger / 安定性検証' -ForegroundColor White
Write-Host " 予定時間: $DurationMinutes 分 / サンプリング: $SampleSeconds 秒" -ForegroundColor Gray
Write-Host " 監視対象: VC Client プロセス（自動判定） / $Url" -ForegroundColor DarkGray

$initialProcs = @(Find-VcClientProcesses -Root $vcRoot -Names $ProcessName)
$initialPort = Test-PortOpen -Port $port -TimeoutMilliseconds 1000
if ($initialProcs.Count -eq 0 -and -not $initialPort) {
    Write-Host ''
    Write-Err "VC Client が起動していません（プロセスもポート $port も応答なし）。"
    Write-Info '先に scripts\Start-VoiceChanger.ps1 を実行し、[start] を押して変換を開始してください。'
    if (-not $AllowNotRunning) {
        Write-Info '監視だけ行いたい場合は -AllowNotRunning を付けて実行してください。'
        exit 1
    }
    Write-Warn 'AllowNotRunning のため続行します。全サンプルが異常記録になります。'
}
if ($initialProcs.Count -gt 1) {
    Write-Warn "同名プロセスが $($initialProcs.Count) 個あります。多重起動の可能性があります。"
}

# ---------------------------------------------------------------- ループ
Write-Step "計測開始（$(Get-Date -Format 'HH:mm:ss')）"
Write-Host ' 操作: [D] 音切れ  [S] 停止  [N] 声の崩れ  [M] メモ  [Q] 終了' -ForegroundColor DarkCyan
Write-Host ''

$samples = New-Object System.Collections.Generic.List[object]
$incidents = New-Object System.Collections.Generic.List[object]
$sw = [System.Diagnostics.Stopwatch]::StartNew()
$totalSeconds = $DurationMinutes * 60
$prevCpuMs = $null
$prevAt = $null
$lastProcOk = $null
$lastPortOk = $null
$lastHttpOk = $null
$keyInputAvailable = $true
$index = 0

function Add-Incident {
    param(
        [Parameter(Mandatory = $true)][string]$Kind,
        [Parameter(Mandatory = $true)][string]$Detail,
        [Parameter(Mandatory = $true)][double]$ElapsedSeconds
    )
    $entry = [pscustomobject]@{
        ElapsedSeconds = [math]::Round($ElapsedSeconds, 1)
        Clock          = (Get-Date -Format 'HH:mm:ss')
        Kind           = $Kind
        Detail         = $Detail
    }
    $incidents.Add($entry)
    Write-Host ''
    Write-Host ("  !! [{0}] {1} — {2}" -f $entry.Clock, $Kind, $Detail) -ForegroundColor Red
}

while ($sw.Elapsed.TotalSeconds -lt $totalSeconds) {
    $index++
    $now = Get-Date
    $elapsed = $sw.Elapsed.TotalSeconds

    # --- プロセス
    $procs = @(Find-VcClientProcesses -Root $vcRoot -Names $ProcessName)
    $proc = $procs | Sort-Object -Property WorkingSet64 -Descending | Select-Object -First 1
    $procOk = $null -ne $proc
    $cpuPct = $null
    $memMB = $null
    $threads = $null
    $handles = $null

    if ($procOk) {
        $memMB = [math]::Round($proc.WorkingSet64 / 1MB, 1)
        $threads = $proc.Threads.Count
        $handles = $proc.HandleCount
        if ($null -ne $prevCpuMs -and $null -ne $prevAt) {
            $deltaMs = ($now - $prevAt).TotalMilliseconds
            if ($deltaMs -gt 0) {
                $cpuDelta = ($proc.TotalProcessorTime.TotalMilliseconds) - $prevCpuMs
                $cpuPct = [math]::Round(($cpuDelta / $deltaMs / [Environment]::ProcessorCount) * 100, 1)
            }
        }
        $prevCpuMs = $proc.TotalProcessorTime.TotalMilliseconds
        $prevAt = $now
    } else {
        $prevCpuMs = $null
        $prevAt = $null
    }

    # --- ポート / HTTP
    $portOk = Test-PortOpen -Port $port -TimeoutMilliseconds 800
    $httpOk = $false
    $httpStatus = ''
    if ($portOk) {
        try {
            $resp = Invoke-WebRequest -Uri $Url -UseBasicParsing -TimeoutSec $TimeoutSeconds -ErrorAction Stop
            $httpOk = $true
            $httpStatus = [string]$resp.StatusCode
        } catch {
            $httpStatus = 'error: ' + $_.Exception.Message
        }
    }

    # --- GPU
    $gpu = Get-GpuSample

    # --- 状態遷移の記録
    if ($null -ne $lastProcOk -and $lastProcOk -ne $procOk) {
        if ($procOk) { Add-Incident -Kind 'プロセス復帰' -Detail 'VC Client のプロセスを再検出' -ElapsedSeconds $elapsed }
        else { Add-Incident -Kind 'プロセス消失' -Detail 'VC Client のプロセスがすべて終了した' -ElapsedSeconds $elapsed }
    }
    if ($null -ne $lastPortOk -and $lastPortOk -ne $portOk) {
        if ($portOk) { Add-Incident -Kind 'GUI 復帰' -Detail "ポート $port が応答を再開" -ElapsedSeconds $elapsed }
        else { Add-Incident -Kind 'GUI 停止' -Detail "ポート $port が応答しなくなった" -ElapsedSeconds $elapsed }
    }
    if ($null -ne $lastHttpOk -and $lastHttpOk -ne $httpOk) {
        if ($httpOk) { Add-Incident -Kind 'HTTP 復帰' -Detail "$Url が応答を再開" -ElapsedSeconds $elapsed }
        else { Add-Incident -Kind 'HTTP 失敗' -Detail "$Url の取得に失敗" -ElapsedSeconds $elapsed }
    }
    $lastProcOk = $procOk
    $lastPortOk = $portOk
    $lastHttpOk = $httpOk

    $row = [pscustomobject]@{
        Index        = $index
        ElapsedSec   = [math]::Round($elapsed, 1)
        Clock        = $now.ToString('HH:mm:ss')
        ProcessOk    = $procOk
        CpuPercent   = $cpuPct
        MemoryMB     = $memMB
        Threads      = $threads
        Handles      = $handles
        GpuUtilPct   = if ($null -ne $gpu) { $gpu.Util } else { $null }
        GpuMemMB     = if ($null -ne $gpu) { $gpu.MemMB } else { $null }
        GpuTempC     = if ($null -ne $gpu) { $gpu.TempC } else { $null }
        PortOpen     = $portOk
        HttpOk       = $httpOk
        HttpStatus   = $httpStatus
        Event        = ''
    }
    $samples.Add($row)

    $cpuText = if ($null -ne $cpuPct) { '{0,5:N1}%' -f $cpuPct } else { '   -  ' }
    $gpuText = if ($null -ne $gpu) { '{0,3:N0}% {1,5:N0}MB' -f $gpu.Util, $gpu.MemMB } else { '   -       ' }
    $stateText = if ($procOk -and $httpOk) { 'OK  ' } elseif ($portOk) { 'WARN' } else { 'NG  ' }
    $stateColor = if ($procOk -and $httpOk) { 'Green' } elseif ($portOk) { 'Yellow' } else { 'Red' }
    $remain = [math]::Max(0, [int]($totalSeconds - $sw.Elapsed.TotalSeconds))
    Write-Host ("  [{0,3}] {1} {2}  CPU {3}  GPU {4}  MEM {5,7} MB  残 {6,3}s" -f `
            $index, $now.ToString('HH:mm:ss'), $stateText, $cpuText, $gpuText, `
            $(if ($null -ne $memMB) { $memMB } else { 0 }), $remain) -ForegroundColor $stateColor

    # --- 次のサンプルまで待機しつつキー入力を受け付ける
    $waitUntil = $now.AddSeconds($SampleSeconds)
    while ((Get-Date) -lt $waitUntil) {
        if ($keyInputAvailable) {
            try {
                if ([Console]::KeyAvailable) {
                    $key = [Console]::ReadKey($true)
                    $row.Event = switch ($key.Key) {
                        'D' { Add-Incident -Kind '音切れ' -Detail '操作者が音切れを検知' -ElapsedSeconds $sw.Elapsed.TotalSeconds; '音切れ' }
                        'S' { Add-Incident -Kind '変換停止' -Detail '操作者が停止を検知' -ElapsedSeconds $sw.Elapsed.TotalSeconds; '停止' }
                        'N' { Add-Incident -Kind '声の崩れ' -Detail '操作者が声質の崩れを検知' -ElapsedSeconds $sw.Elapsed.TotalSeconds; '声の崩れ' }
                        'M' {
                            $memo = Read-Host '  メモ'
                            Add-Incident -Kind 'メモ' -Detail $memo -ElapsedSeconds $sw.Elapsed.TotalSeconds
                            "メモ: $memo"
                        }
                        'Q' { Add-Incident -Kind '手動終了' -Detail '操作者がテストを終了' -ElapsedSeconds $sw.Elapsed.TotalSeconds; '終了' }
                        default { $row.Event }
                    }
                    if ($key.Key -eq 'Q') { $sw.Stop(); break }
                }
            } catch {
                $keyInputAvailable = $false
                Write-Info '対話コンソールが無いためキー入力の記録は無効です。'
            }
        }
        Start-Sleep -Milliseconds 150
    }
    if ($sw.Elapsed.TotalSeconds -ge $totalSeconds) { break }
    if ($incidents.Count -gt 0 -and $incidents[$incidents.Count - 1].Kind -eq '手動終了') { break }
}
$sw.Stop()
$actualSeconds = [math]::Round($sw.Elapsed.TotalSeconds, 1)

# ---------------------------------------------------------------- 集計
Write-Step '集計'

$procStats = Get-Stats -Values ($samples | ForEach-Object { $_.CpuPercent })
$gpuStats = Get-Stats -Values ($samples | ForEach-Object { $_.GpuUtilPct })
$memStats = Get-Stats -Values ($samples | ForEach-Object { $_.MemoryMB })
$tempStats = Get-Stats -Values ($samples | ForEach-Object { $_.GpuTempC })

$sampleCount = $samples.Count
$procDown = @($samples | Where-Object { -not $_.ProcessOk }).Count
$portDown = @($samples | Where-Object { -not $_.PortOpen }).Count
$httpFail = @($samples | Where-Object { $_.PortOpen -and -not $_.HttpOk }).Count
$operatorEvents = @($incidents | Where-Object { $_.Kind -in @('音切れ', '停止', '変換停止', '声の崩れ') }).Count

$procUptimePct = if ($sampleCount -gt 0) { [math]::Round((($sampleCount - $procDown) / $sampleCount) * 100, 1) } else { 0 }
$httpUptimePct = if ($sampleCount -gt 0) { [math]::Round((($sampleCount - $portDown - $httpFail) / $sampleCount) * 100, 1) } else { 0 }

Write-Info "サンプル数: $sampleCount / 実測時間: $actualSeconds 秒"
Write-Info "プロセス稼働率: $procUptimePct %"
Write-Info "GUI 応答率: $httpUptimePct %"
if ($null -ne $procStats) { Write-Info "CPU 平均/最大: $($procStats.Avg) % / $($procStats.Max) %" }
if ($null -ne $gpuStats) { Write-Info "GPU 平均/最大: $($gpuStats.Avg) % / $($gpuStats.Max) %" }
Write-Info "操作者が記録した事象: $operatorEvents 件"

# ---------------------------------------------------------------- 出力
$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$csvPath = Join-Path $logDir ("stability-$stamp.csv")
$samples | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding UTF8

$L = New-Object System.Collections.Generic.List[string]
$L.Add('# 安定性検証レポート')
$L.Add('')
$L.Add("- 実施日時: $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))")
$L.Add("- プロファイル: $(if ($Profile) { $Profile } else { '（未指定）' })")
$L.Add("- 目標時間: $DurationMinutes 分 / 実測: $actualSeconds 秒")
$L.Add("- サンプリング間隔: $SampleSeconds 秒（$sampleCount サンプル）")
$L.Add("- 監視対象: VC Client プロセス（自動判定: ``$ProcessName``）/ ``$Url``")
$L.Add("- 生ログ: ``$csvPath``")
$L.Add('')
$L.Add('## 自動計測の結果')
$L.Add('')
$L.Add('| 指標 | 値 |')
$L.Add('| :--- | :--- |')
$L.Add("| プロセス稼働率 | $procUptimePct % |")
$L.Add("| GUI 応答率 | $httpUptimePct % |")
$L.Add("| プロセス消失サンプル数 | $procDown |")
$L.Add("| ポート無応答サンプル数 | $portDown |")
$L.Add("| HTTP 失敗サンプル数 | $httpFail |")
if ($null -ne $procStats) {
    $L.Add("| CPU 使用率（平均 / 最大） | $($procStats.Avg) % / $($procStats.Max) % |")
} else {
    $L.Add('| CPU 使用率 | 取得できず |')
}
if ($null -ne $gpuStats) {
    $L.Add("| GPU 使用率（平均 / 最大） | $($gpuStats.Avg) % / $($gpuStats.Max) % |")
} else {
    $L.Add('| GPU 使用率 | 取得できず（nvidia-smi なし） |')
}
if ($null -ne $tempStats) {
    $L.Add("| GPU 温度（最大） | $($tempStats.Max) ℃ |")
}
if ($null -ne $memStats) {
    $L.Add("| メモリ（平均 / 最大） | $($memStats.Avg) MB / $($memStats.Max) MB |")
}
$L.Add('')
$L.Add('## 記録された事象')
$L.Add('')
if ($incidents.Count -eq 0) {
    $L.Add('自動検出・操作者記録ともに事象なし。')
} else {
    $L.Add('| 経過 | 時刻 | 種別 | 内容 |')
    $L.Add('| :--- | :--- | :--- | :--- |')
    foreach ($i in $incidents) {
        $L.Add("| $($i.ElapsedSeconds) 秒 | $($i.Clock) | $($i.Kind) | $($i.Detail) |")
    }
}
$L.Add('')
$L.Add('## 聴取評価（評価担当が記入）')
$L.Add('')
$L.Add('機械では判定できないため、評価担当 2 名程度が次を記入する。詳細は docs/04-verification.md の評価シートを使う。')
$L.Add('')
$L.Add('| 発話パターン | 評価担当1 | 評価担当2 | コメント |')
$L.Add('| :--- | :--- | :--- | :--- |')
$L.Add('| 通常発話 |  |  |  |')
$L.Add('| 早口 |  |  |  |')
$L.Add('| 感情を込めた発話 |  |  |  |')
$L.Add('')
$L.Add('## MVP 判定（設計書 2.3）')
$L.Add('')
$L.Add('- [ ] 10 分間の連続発話中に明らかな停止がなかった')
$L.Add('- [ ] 頻繁な音切れがなかった')
$L.Add('- [ ] 著しい声の崩れがなかった')
$L.Add('- [ ] 評価担当 2 名が通常発話・早口・感情発話を聞き取れた')
$L.Add('- [ ] 会話のやり取りが可能な遅延だった（体感評価）')
$L.Add('')
$L.Add('判定: **未判定**（上記を記入して確定させる）')
$L.Add('')
$L.Add('## 所見・次アクション')
$L.Add('')
$L.Add('- ')
$L.Add('')
$L.Add('> 自動計測は「ソフトが生きているか」しか見ていない。音切れ・声質は必ず人の耳で判定すること。')

$mdPath = Join-Path $reportDir ("stability-$stamp.md")
$L -join "`r`n" | Set-Content -LiteralPath $mdPath -Encoding UTF8

Write-Step '出力'
Write-Ok "レポート: $mdPath"
Write-Ok "生ログ  : $csvPath"
Write-Host ''
Write-Host ' 次: レポートの「聴取評価」「MVP 判定」を埋め、docs/04-verification.md に転記してください。' -ForegroundColor Cyan
