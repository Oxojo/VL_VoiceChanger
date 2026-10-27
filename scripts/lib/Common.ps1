# VL_VoiceChanger 共通ライブラリ
# scripts/ 配下の各スクリプトから dot-source して使う。PowerShell 5.1 で動作すること。

Set-StrictMode -Version Latest

function Get-RepoRoot {
    # scripts/lib/Common.ps1 から2階層上がリポジトリルート
    return (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
}

function Resolve-RepoPath {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [string]$BasePath = (Get-RepoRoot)
    )
    if ([string]::IsNullOrWhiteSpace($Path)) { return '' }
    if ([System.IO.Path]::IsPathRooted($Path)) { return [System.IO.Path]::GetFullPath($Path) }
    return [System.IO.Path]::GetFullPath((Join-Path $BasePath $Path))
}

function Read-JsonFile {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "設定ファイルが見つかりません: $Path"
    }
    $raw = Get-Content -LiteralPath $Path -Raw -Encoding UTF8
    # コメント用の "$comment" キーは読み込み時に落とす
    return ($raw | ConvertFrom-Json)
}

function Get-SettingValue {
    param(
        $Object,
        [Parameter(Mandatory = $true)][string]$Path,
        $Default = $null
    )
    $current = $Object
    foreach ($segment in $Path.Split('.')) {
        if ($null -eq $current) { return $Default }
        $prop = $current.PSObject.Properties[$segment]
        if ($null -eq $prop) { return $Default }
        $current = $prop.Value
    }
    if ($null -eq $current -or ($current -is [string] -and [string]::IsNullOrWhiteSpace($current))) {
        return $Default
    }
    return $current
}

function Invoke-VcApi {
    # PowerShell 5.1 の Invoke-RestMethod は JSON の文字コードを誤認して日本語が化ける。
    # 生バイト列を UTF-8 としてデコードしてから ConvertFrom-Json する。
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [ValidateSet('GET', 'POST', 'PUT', 'DELETE')][string]$Method = 'GET',
        $Body = $null,
        [int]$TimeoutSec = 60
    )
    $params = @{
        Uri         = $Uri
        Method      = $Method
        UseBasicParsing = $true
        TimeoutSec  = $TimeoutSec
    }
    if ($null -ne $Body) {
        $json = if ($Body -is [string]) { $Body } else { $Body | ConvertTo-Json -Depth 10 }
        $params['Body'] = [System.Text.Encoding]::UTF8.GetBytes($json)
        $params['ContentType'] = 'application/json; charset=utf-8'
    }
    $response = Invoke-WebRequest @params
    $text = [System.Text.Encoding]::UTF8.GetString($response.RawContentStream.ToArray())
    if ([string]::IsNullOrWhiteSpace($text)) { return $null }
    return ($text | ConvertFrom-Json)
}

function Get-ObjectProperty {
    # Set-StrictMode 下では存在しないプロパティの参照が例外になる。
    # API のレスポンスはスロットの状態によってキーが欠けることがあるため、明示的に安全に読む。
    param(
        $Object,
        [Parameter(Mandatory = $true)][string]$Name,
        $Default = $null
    )
    if ($null -eq $Object) { return $Default }
    $prop = $Object.PSObject.Properties[$Name]
    if ($null -eq $prop) { return $Default }
    return $prop.Value
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host ''
    Write-Host "== $Message" -ForegroundColor Cyan
}

function Write-Ok {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "  [OK]   $Message" -ForegroundColor Green
}

function Write-Info {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "  [INFO] $Message" -ForegroundColor Gray
}

function Write-Warn {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "  [WARN] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host "  [NG]   $Message" -ForegroundColor Red
}

function Format-FileSize {
    param([double]$Bytes)
    if ($Bytes -ge 1TB) { return ('{0:N1} TB' -f ($Bytes / 1TB)) }
    if ($Bytes -ge 1GB) { return ('{0:N1} GB' -f ($Bytes / 1GB)) }
    if ($Bytes -ge 1MB) { return ('{0:N0} MB' -f ($Bytes / 1MB)) }
    if ($Bytes -ge 1KB) { return ('{0:N0} KB' -f ($Bytes / 1KB)) }
    return ('{0:N0} B' -f $Bytes)
}

function Get-FreeSpaceGB {
    param([Parameter(Mandatory = $true)][string]$Path)
    try {
        $drive = (Get-Item -LiteralPath $Path).PSDrive
        if ($null -eq $drive) { return $null }
        return [math]::Round($drive.Free / 1GB, 1)
    } catch {
        return $null
    }
}

function Test-PortOpen {
    param(
        [string]$ComputerName = 'localhost',
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$TimeoutMilliseconds = 1000
    )
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($ComputerName, $Port, $null, $null)
        if (-not $async.AsyncWaitHandle.WaitOne($TimeoutMilliseconds)) { return $false }
        $client.EndConnect($async)
        return $true
    } catch {
        return $false
    } finally {
        $client.Close()
    }
}

function Wait-Port {
    param(
        [Parameter(Mandatory = $true)][int]$Port,
        [int]$TimeoutSeconds = 120,
        [string]$Label = 'localhost'
    )
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    while ($sw.Elapsed.TotalSeconds -lt $TimeoutSeconds) {
        if (Test-PortOpen -Port $Port -TimeoutMilliseconds 500) { return $true }
        Start-Sleep -Milliseconds 500
    }
    return $false
}

function Get-AudioDevices {
    # 録音/再生エンドポイントを取得する。取得できない環境では空配列を返す。
    # MMDevAPI の InstanceId は {0.0.0.x} が再生（output）、{0.0.1.x} が録音（input）。
    $result = @()
    try {
        $pnp = Get-PnpDevice -Class AudioEndpoint -Status 'OK' -ErrorAction Stop
        foreach ($d in $pnp) {
            $result += [pscustomobject]@{
                Name       = $d.FriendlyName
                DeviceType = if ("$($d.InstanceId)" -like '*{0.0.1.00000000}*') { 'input' } else { 'output' }
                Status     = $d.Status
            }
        }
    } catch {
        # Get-PnpDevice が使えない場合は Win32_SoundDevice にフォールバック
        try {
            $snd = Get-CimInstance -ClassName Win32_SoundDevice -ErrorAction Stop
            foreach ($d in $snd) {
                $result += [pscustomobject]@{
                    Name       = $d.Name
                    DeviceType = 'unknown'
                    Status     = $d.Status
                }
            }
        } catch { }
    }
    return @($result)
}

function Get-GpuInfo {
    # nvidia-smi があれば実 VRAM と使用率を取得。無ければ CIM で名前だけ取得。
    $gpus = @()
    $smi = Get-Command 'nvidia-smi.exe' -ErrorAction SilentlyContinue
    if ($null -ne $smi) {
        try {
            $out = & $smi.Source --query-gpu=index,name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits 2>$null
            foreach ($line in @($out)) {
                if ([string]::IsNullOrWhiteSpace($line)) { continue }
                $f = $line.Split(',').Trim()
                $gpus += [pscustomobject]@{
                    Index      = [int]$f[0]
                    Name       = $f[1]
                    Driver     = $f[2]
                    VramTotalMB = [double]$f[3]
                    VramUsedMB  = [double]$f[4]
                    UtilPct     = [double]$f[5]
                    TempC       = [double]$f[6]
                    Source      = 'nvidia-smi'
                }
            }
        } catch { }
    }
    if ($gpus.Count -eq 0) {
        try {
            $vcs = Get-CimInstance -ClassName Win32_VideoController -ErrorAction Stop
            foreach ($vc in $vcs) {
                # AdapterRAM は 32bit 符号付きで 4GB を超えると不正確。参考値として扱う。
                $vram = if ($null -ne $vc.AdapterRAM -and $vc.AdapterRAM -gt 0) { [math]::Round($vc.AdapterRAM / 1MB, 0) } else { $null }
                $gpus += [pscustomobject]@{
                    Index       = $null
                    Name        = $vc.Name
                    Driver      = $vc.DriverVersion
                    VramTotalMB = $vram
                    VramUsedMB  = $null
                    UtilPct     = $null
                    TempC       = $null
                    Source      = 'Win32_VideoController'
                }
            }
        } catch { }
    }
    return @($gpus)
}

function Find-VcClientProcesses {
    <#
        VC Client のプロセスを探す。
        exe 名は版によって変わる（MMVC_Client / MMVCServerSIO など）ため、
        まず vcClient.root 配下の実行ファイルから起動されたプロセスを Path で判定し、
        見つからなければ名前パターンでフォールバックする。
    #>
    param(
        [string]$Root = '',
        [string]$Names = 'main,MMVC_Client*,MMVCServerSIO*,vcclient*'
    )

    $byId = @{}

    if (-not [string]::IsNullOrWhiteSpace($Root) -and (Test-Path -LiteralPath $Root)) {
        $normRoot = $Root.Replace('\', '/').TrimEnd('/')
        foreach ($p in @(Get-Process -ErrorAction SilentlyContinue)) {
            try {
                $path = $p.Path
                if ($path -and $path.Replace('\', '/').StartsWith($normRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
                    $byId[$p.Id] = $p
                }
            } catch {
                # 他ユーザーのプロセスは Path を読めない。無視する。
            }
        }
    }

    if ($byId.Count -eq 0 -and -not [string]::IsNullOrWhiteSpace($Names)) {
        foreach ($pattern in $Names.Split(',')) {
            $pattern = $pattern.Trim()
            if ([string]::IsNullOrWhiteSpace($pattern)) { continue }
            foreach ($p in @(Get-Process -Name $pattern -ErrorAction SilentlyContinue)) {
                $byId[$p.Id] = $p
            }
        }
    }

    return @($byId.Values | Sort-Object -Property WorkingSet64 -Descending)
}
