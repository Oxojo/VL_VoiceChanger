<#
.SYNOPSIS
    自前エンジン（engine/）で WAV を音声変換する。

.DESCRIPTION
    Python 仮想環境 .venv の python で `python -m engine convert` を呼び出す。
    追加の引数はそのまま CLI へ渡す（--pitch / --speaker / --rms-mix / --device など）。

.EXAMPLE
    .\scripts\Convert-Audio.ps1 -InputPath logs\verify\tts-input-ja.wav -OutputPath logs\verify\own-ja.wav

.EXAMPLE
    .\scripts\Convert-Audio.ps1 -InputPath in.wav -OutputPath out.wav -ExtraArgs '--pitch','12','--device','cuda'
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [string]$ModelPath = 'models\tsukuyomi\tsukuyomi-normal1.pth',

    [string]$PythonPath = '.venv\Scripts\python.exe',

    [string]$JsonPath,

    [string[]]$ExtraArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    if (-not (Test-Path $PythonPath)) {
        throw "Python が見つかりません: $PythonPath（py -3.12 -m venv .venv で作成してください）"
    }
    if (-not (Test-Path $ModelPath)) {
        throw "モデルが見つかりません: $ModelPath"
    }
    if (-not (Test-Path $InputPath)) {
        throw "入力 WAV が見つかりません: $InputPath"
    }

    $arguments = @('-m', 'engine', 'convert', '-m', $ModelPath, '-i', $InputPath, '-o', $OutputPath)
    if ($JsonPath) { $arguments += @('--json', $JsonPath) }
    if ($ExtraArgs.Count -gt 0) { $arguments += $ExtraArgs }

    & $PythonPath @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "変換に失敗しました（終了コード $LASTEXITCODE）"
    }
}
finally {
    Pop-Location
}
