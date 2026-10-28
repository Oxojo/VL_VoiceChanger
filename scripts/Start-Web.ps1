<#
.SYNOPSIS
    自前エンジンの Web UI（ブラウザ画面）を起動する。

.DESCRIPTION
    Python の FastAPI サーバーを起動し、既定のブラウザで http://127.0.0.1:<Port> を開く。
    変換はこの PC 上で実行され、音声は外部に送信されない。
    停止は Ctrl+C（このウィンドウを閉じてもよい）。

.EXAMPLE
    .\scripts\Start-Web.ps1

.EXAMPLE
    .\scripts\Start-Web.ps1 -Port 9000 -NoBrowser
#>
[CmdletBinding()]
param(
    [string]$PythonPath = '.venv\Scripts\python.exe',
    [int]$Port = 8765,
    [string]$HostName = '127.0.0.1',
    [switch]$NoBrowser
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not (Test-Path $PythonPath)) {
        throw "Python が見つかりません: $PythonPath（先に .venv を作成してください）"
    }

    # Web UI の依存関係を確認し、無ければ入れる（初回のみ）
    & $PythonPath -c "import fastapi, uvicorn, multipart" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Web UI の依存関係をインストールします（初回のみ）...'
        & $PythonPath -m pip install fastapi uvicorn python-multipart
        if ($LASTEXITCODE -ne 0) { throw "依存関係のインストールに失敗しました（終了コード $LASTEXITCODE）" }
    }

    if (-not (Test-Path 'models')) {
        Write-Warning 'models/ がありません。RVC v2 の .pth を置くまで変換できません。'
    }

    $url = "http://${HostName}:$Port"
    if (-not $NoBrowser) {
        # サーバーが待ち受けを始めるまで少し待ってから開く
        Start-Job -ScriptBlock {
            param($u)
            Start-Sleep -Seconds 3
            Start-Process $u
        } -ArgumentList $url | Out-Null
    }

    Write-Host "ブラウザ UI: $url"
    Write-Host '停止するには Ctrl+C を押してください。'
    & $PythonPath -m engine serve --host $HostName --port $Port
}
finally {
    Pop-Location
}
