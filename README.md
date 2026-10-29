# VL_VoiceChanger

バーチャルライブ向け **リアルタイム音声変換** の実証プロジェクト。
演者 1 名のマイク入力を RVC モデルで別キャラクターの声に変換し、指定した出力デバイスで確認できる状態を作る。

- 変換エンジン: [w-okada/voice-changer（VC Client）](https://github.com/w-okada/voice-changer) の **RVC 対応エディション（cuda / dml）**
- 自前エンジン: `engine/`（Python + PyTorch / ONNX Runtime）。VC Client に依存せず WAV → WAV 変換を実行する
- CLI: `vlvc/`（Python 3.10+）。**Windows / macOS / Linux で同じコマンドが動く**（PowerShell 不要）
- 採用モデル: **つくよみちゃん公式RVCモデル** ver 1.0.0（`models/tsukuyomi/`、Git 管理外）
- 対象: 演者 1 名 / モデル 1 種類（MVP）
- 本方針の原典: `virtuallive_音声変換特化_改訂版.md`（企画書）

## 成果物（企画書 8 章）

| # | 成果物 | このリポジトリでの実体 |
| :-- | :--- | :--- |
| 1 | 動作する音声変換環境 | `config/settings.json` + `config/profiles/*.json` + VC Client 本体 + `engine/` |
| 2 | 設定・導入手順書 | `docs/00`〜`docs/09` |
| 3 | 検証記録 | `reports/stability-*.md`、`reports/environment-*.md`、`docs/04-verification.md` |

## クイックスタート

```bash
# 0. 初回のみ: 仮想環境と依存関係（.venv を作る。CUDA を使うなら --cuda）
python -m vlvc setup
# GPU を使う場合: python -m vlvc setup --cuda

# 1. Phase 0: 機材・環境の棚卸し（レポートが reports/ に出る）
python -m vlvc doctor

# 2. Phase 1: モデルを VC Client に登録（初回のみ。登録済みなら何もしない）
python -m vlvc register-model models/tsukuyomi/tsukuyomi-normal1.pth --name 'つくよみちゃん（通常1）'

# 3. Phase 1: プロファイルを確認して VC Client を起動
python -m vlvc launch character-a

# 4. Phase 1: プロファイルのうち REST で変更できる値を反映（--dry-run で差分確認）
python -m vlvc profile apply character-a --dry-run
python -m vlvc profile apply character-a

# 5. Phase 3: 10 分間の連続使用を計測
python -m vlvc stability --profile character-a
```

`pip install -e .` を実行すると `vlvc` コマンドとしても使える（以降の例は `python -m vlvc` で統一）。

`config/settings.json` の `vcClient.root` を、実際に VC Client を解凍したフォルダへ書き換えてから実行する。
VC Client 本体は Windows 向け配布のみ。**自前エンジン（`convert` / `serve` / `quality` / `selftest`）は macOS / Linux でも動く**。

**変換の開始（start）だけは GUI 操作**（`http://localhost:18000/`）。REST API では変換スレッドが起動しないことを実測で確認済み。

### 自前エンジン（VC Client を使わない変換）

```bash
# モデル構成の確認
python -m engine inspect -m models/tsukuyomi/tsukuyomi-normal1.pth

# 変換（--device cuda で GPU、--pitch 12 で 1 オクターブ上）
python -m vlvc convert -m models/tsukuyomi/tsukuyomi-normal1.pth \
    -i logs/verify/tts-input-ja.wav -o logs/verify/own-ja.wav --device cuda

# 出力の F0・音量の確認
python -m engine analyze -i logs/verify/own-ja.wav

# 変換前後の比較（F0 追従・有声率・調波性・クリップ）
python -m vlvc quality logs/verify/tts-input-ja.wav logs/verify/own-ja.wav

# ノイズ除去（前処理）: 強さ 0〜1 / 電源ハム 50 or 60 Hz
python -m engine denoise -i logs/verify/input.wav -o logs/verify/input-clean.wav --strength 0.7 --dehum 50

# 一括の動作確認（変換 → 解析 → 品質 → 分割変換）とレポート
python -m vlvc selftest --device cuda

# ブラウザ UI で変換（http://127.0.0.1:8765）
python -m vlvc serve
```

ブラウザ UI は音声のドロップ／マイク録音・設定・試聴・ダウンロードまでをブラウザで行える。
ノイズ除去・ハム除去も画面から指定できる。
詳細・API・実測値は [docs/08-web-ui.md](docs/08-web-ui.md)、エンジン自体は
[docs/07-own-engine.md](docs/07-own-engine.md)。

### NeoShowcase への配置

**Runtime / Dockerfile** を選び、コンテナの **8080 番ポート**へ転送する。
Static 配信では変換 API が動かない。Python 3.12・CPU 版 PyTorch の Dockerfile を同梱。
モデルの配置と公開前のアクセス制限については [docs/10-neoshowcase.md](docs/10-neoshowcase.md) を参照。

## 前提

- Python 3.10 以上（検証機は 3.12.10）
- VC Client を使う場合のみ Windows 10/11
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
├─ vlvc/                     # クロスプラットフォーム CLI（環境確認・起動・設定・計測）
├─ models/                   # モデル配置（Git 管理外）
├─ reports/                  # 検証レポート（Markdown はコミット対象）
├─ logs/                     # 生ログ CSV（Git 管理外）
├─ tests/                    # エンジンと CLI のテスト（unittest）
└─ pyproject.toml            # `pip install -e .` で `vlvc` コマンドを提供
```

### CLI のコマンド一覧

| コマンド | 旧 PowerShell（廃止） | 内容 |
| :--- | :--- | :--- |
| `vlvc setup` | （手順書のコマンド） | `.venv` 作成と依存関係の導入 |
| `vlvc doctor` | `Check-Environment.ps1` | 機材・環境の確認とレポート |
| `vlvc launch` | `Start-VoiceChanger.ps1` | 起動前検証・VC Client 起動・GUI チェックリスト |
| `vlvc profile list/check/apply` | `Set-VoiceProfile.ps1` | プロファイルの一覧・検証・REST 適用 |
| `vlvc register-model` | `Register-Model.ps1` | RVC モデルの登録 |
| `vlvc convert` | `Convert-Audio.ps1` | WAV 変換（engine へ委譲） |
| `vlvc serve` | `Start-Web.ps1` | ブラウザ UI の起動 |
| `vlvc quality` | `measure_quality.py` | 変換前後の数値比較 |
| `vlvc selftest` | `Test-OwnEngine.ps1` | 変換〜計測のまとめ実行 |
| `vlvc stability` | `Test-Stability.ps1` | 連続使用テスト（CSV + Markdown） |

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
  - モデル登録: `POST /api/slot-manager/slots`（`vlvc register-model`）
  - 設定反映: `PUT /api/configuration-manager/configuration`（`vlvc profile apply`）
  - **変換の開始だけは GUI の start が必要**（REST 単独では `bulk conversion process is not started`）
  - スロット固有の TUNE / INDEX / CHUNK / F0 も GUI で設定する（`PUT /api/slot-manager/slots/{index}` はボディを受け付けない）
- モデルは利用規約が個別に異なる。「無料だから再配布自由」とは判断しない。採用前に `docs/05-models.md` へ条件を記録する。
- 本番 PA・OBS・DAW・映像同期は本リポジトリの対象外（企画書 2.2）。
