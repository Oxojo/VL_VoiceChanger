# VL_VoiceChanger

バーチャルライブ向け **リアルタイム音声変換** の実証プロジェクト。
演者 1 名のマイク入力を RVC モデルで別キャラクターの声に変換し、指定した出力デバイスで確認できる状態を作る。

- 変換エンジン: [w-okada/voice-changer（VC Client）](https://github.com/w-okada/voice-changer) の **RVC 対応エディション（cuda / dml）**
- 自前エンジン: `engine/`（Python + PyTorch / ONNX Runtime）。VC Client に依存せず WAV → WAV 変換を実行する
- 採用モデル: **つくよみちゃん公式RVCモデル** ver 1.0.0（`models/tsukuyomi/`、Git 管理外）
- 対象: 演者 1 名 / モデル 1 種類（MVP）
- 本方針の原典: `virtuallive_音声変換特化_改訂版.md`（企画書）

## 成果物（企画書 8 章）

| # | 成果物 | このリポジトリでの実体 |
| :-- | :--- | :--- |
| 1 | 動作する音声変換環境 | `config/settings.json` + `config/profiles/*.json` + VC Client 本体 + `engine/` |
| 2 | 設定・導入手順書 | `docs/00`〜`docs/08` |
| 3 | 検証記録 | `reports/stability-*.md`、`reports/environment-*.md`、`docs/04-verification.md` |

## クイックスタート

```powershell
# 1. Phase 0: 機材・環境の棚卸し（レポートが reports/ に出る）
.\scripts\Check-Environment.ps1

# 2. Phase 1: モデルを VC Client に登録（初回のみ。登録済みなら何もしない）
.\scripts\Register-Model.ps1 models/tsukuyomi/tsukuyomi-normal1.pth -Name 'つくよみちゃん（通常1）'

# 3. Phase 1: プロファイルを確認して VC Client を起動
.\scripts\Start-VoiceChanger.ps1 -Profile character-a

# 4. Phase 1: プロファイルのうち REST で変更できる値を反映（-DryRun で差分確認）
.\scripts\Set-VoiceProfile.ps1 character-a -DryRun
.\scripts\Set-VoiceProfile.ps1 character-a

# 5. Phase 3: 10 分間の連続使用を計測
.\scripts\Test-Stability.ps1 -Profile character-a
```

`config/settings.json` の `vcClient.root` を、実際に VC Client を解凍したフォルダへ書き換えてから実行する。

**変換の開始（start）だけは GUI 操作**（`http://localhost:18000/`）。REST API では変換スレッドが起動しないことを実測で確認済み。

### 自前エンジン（VC Client を使わない変換）

```powershell
# 初回のみ: Python 3.12 の仮想環境と依存関係
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r engine\requirements.txt

# モデル構成の確認
.\.venv\Scripts\python.exe -m engine inspect -m models\tsukuyomi\tsukuyomi-normal1.pth

# 変換（--device cuda で GPU、--pitch 12 で 1 オクターブ上）
.\scripts\Convert-Audio.ps1 -InputPath logs\verify\tts-input-ja.wav -OutputPath logs\verify\own-ja.wav

# 出力の F0・音量の確認
.\.venv\Scripts\python.exe -m engine analyze -i logs\verify\own-ja.wav

# ノイズ除去（前処理）: 強さ 0〜1 / 電源ハム 50 or 60 Hz
.\.venv\Scripts\python.exe -m engine denoise -i logs\verify\input.wav -o logs\verify\input-clean.wav --strength 0.7 --dehum 50

# ブラウザ UI で変換（http://127.0.0.1:8765）
.\scripts\Start-Web.ps1
```

ブラウザ UI は音声のドロップ／マイク録音・設定・試聴・ダウンロードまでをブラウザで行える。
ノイズ除去・ハム除去も画面から指定できる。
詳細・API・実測値は [docs/08-web-ui.md](docs/08-web-ui.md)、エンジン自体は
[docs/07-own-engine.md](docs/07-own-engine.md)。

## 前提

- Windows 10/11 + PowerShell 5.1 以上
- VC Client の **RVC 対応ビルド**（`cuda` または `dml`）
  - 2026-10 時点で公開されている Windows 用は `vcclient_win_cuda_2.1.4-alpha.zip` / `vcclient_win_dml_2.1.4-alpha.zip`
  - **2.2.2-beta の Windows 版は Beatrice 専用で RVC 非対応**。版番号の大きさで選ばない
  - 詳細は [docs/01-setup.md](docs/01-setup.md) を参照
- モデルファイル（`.pth` / `.index`）は**リポジトリにコミットしない**（`models/` は除外済み）
  - 本リポジトリではつくよみちゃん公式RVCモデル（5 種類）を `models/tsukuyomi/` に配置済み
  - クレジットは「変換元の声」と「つくよみちゃん公式RVCモデル」の**両方**が必要 → [docs/05-models.md](docs/05-models.md)

## ディレクトリ構成

```text
VL_VoiceChanger/
├─ config/
│  ├─ settings.json          # VC Client の場所・ポート・既定値
│  └─ profiles/              # キャラクター別の設定（1 ファイル = 1 モデル）
├─ docs/                     # 導入手順・運用・復旧・検証・モデル規約・自前エンジン・Web UI
├─ engine/                   # 自前音声変換エンジン（Python / PyTorch / ONNX Runtime）
│  └─ web/                   # ブラウザ UI（FastAPI + 静的ファイル）
├─ models/                   # モデル配置（Git 管理外）
├─ reports/                  # 検証レポート（Markdown はコミット対象）
├─ logs/                     # 生ログ CSV（Git 管理外）
├─ tests/                    # 自前エンジンの最小テスト（unittest）
└─ scripts/
   ├─ Check-Environment.ps1  # Phase 0: 環境診断
   ├─ Start-VoiceChanger.ps1 # Phase 1: 起動と設定チェックリスト
   ├─ Register-Model.ps1     # Phase 1: モデル登録（REST API）
   ├─ Set-VoiceProfile.ps1   # Phase 1: プロファイル適用（REST API）
   ├─ Test-Stability.ps1     # Phase 3: 10 分連続計測
   ├─ Convert-Audio.ps1      # 自前エンジンでの WAV 変換
   ├─ Start-Web.ps1          # 自前エンジンのブラウザ UI を起動
   └─ lib/Common.ps1         # 共通関数
```

## フェーズ進行

| フェーズ | 作業 | 完了条件 | 状態 |
| :--- | :--- | :--- | :--- |
| Phase 0 | 環境確認・機材台帳 | 機材・ソフト・モデルを決定 | **進行中**（ソフト・モデルは決定済み。機材の現物確認が残り） |
| Phase 1 | 初回変換 | 変換後の声が確認できた | 未着手 |
| Phase 2 | 音質調整 | 評価担当 2 名が明瞭さを確認 | 未着手 |
| Phase 3 | 安定性検証 | MVP 条件を満たし手順を文書化 | 未着手 |
| Phase 4 | 追加検討 | 2 キャラ目の運用可否を判断 | 未着手 |

進捗は `docs/00-phase0-environment.md` の台帳と `docs/04-verification.md` の判定欄で管理する。

## 重要な注意

- **既定の経路は VC Client による変換**（設定・手順・検証記録を管理）。加えて `engine/` に**自前の変換実装**を持ち、VC Client に依存せず WAV → WAV 変換ができる。
- 自動化できる範囲は **起動・モデル登録・デバイス/GPU/ノイズゲート等の設定** まで（REST API）。
  - モデル登録: `POST /api/slot-manager/slots`（`Register-Model.ps1`）
  - 設定反映: `PUT /api/configuration-manager/configuration`（`Set-VoiceProfile.ps1`）
  - **変換の開始だけは GUI の start が必要**（REST 単独では `bulk conversion process is not started`）
  - スロット固有の TUNE / INDEX / CHUNK / F0 も GUI で設定する（`PUT /api/slot-manager/slots/{index}` はボディを受け付けない）
- モデルは利用規約が個別に異なる。「無料だから再配布自由」とは判断しない。採用前に `docs/05-models.md` へ条件を記録する。
- 本番 PA・OBS・DAW・映像同期は本リポジトリの対象外（企画書 2.2）。
