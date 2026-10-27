# VL_VoiceChanger

バーチャルライブ向け **リアルタイム音声変換** の実証プロジェクト。
演者 1 名のマイク入力を RVC モデルで別キャラクターの声に変換し、指定した出力デバイスで確認できる状態を作る。

- 変換エンジン: [w-okada/voice-changer（VC Client）](https://github.com/w-okada/voice-changer) の **RVC 対応エディション（cuda または onnx）**
- 対象: 演者 1 名 / モデル 1 種類（MVP）
- 本方針の原典: `virtuallive_音声変換特化_改訂版.md`（企画書）

## 成果物（企画書 8 章）

| # | 成果物 | このリポジトリでの実体 |
| :-- | :--- | :--- |
| 1 | 動作する音声変換環境 | `config/settings.json` + `config/profiles/*.json` + VC Client 本体 |
| 2 | 設定・導入手順書 | `docs/00`〜`docs/06` |
| 3 | 検証記録 | `reports/stability-*.md`、`reports/environment-*.md`、`docs/04-verification.md` |

## クイックスタート

```powershell
# 1. Phase 0: 機材・環境の棚卸し（レポートが reports/ に出る）
.\scripts\Check-Environment.ps1

# 2. Phase 1: プロファイルを確認して VC Client を起動
.\scripts\Start-VoiceChanger.ps1 -Profile character-a

# 3. Phase 3: 10 分間の連続使用を計測
.\scripts\Test-Stability.ps1 -Profile character-a
```

`config/settings.json` の `vcClient.root` を、実際に VC Client を解凍したフォルダへ書き換えてから実行する。

## 前提

- Windows 10/11 + PowerShell 5.1 以上
- VC Client の **RVC 対応ビルド**（`cuda` または `dml`）
  - 2026-10 時点で公開されている Windows 用は `vcclient_win_cuda_2.1.4-alpha.zip` / `vcclient_win_dml_2.1.4-alpha.zip`
  - **2.2.2-beta の Windows 版は Beatrice 専用で RVC 非対応**。版番号の大きさで選ばない
  - 詳細は [docs/01-setup.md](docs/01-setup.md) を参照
- モデルファイル（`.pth` / `.index`）は各自で用意し、**リポジトリにはコミットしない**（`models/` は除外済み）

## ディレクトリ構成

```text
VL_VoiceChanger/
├─ config/
│  ├─ settings.json          # VC Client の場所・ポート・既定値
│  └─ profiles/              # キャラクター別の設定（1 ファイル = 1 モデル）
├─ docs/                     # 導入手順・運用・復旧・検証・モデル規約
├─ models/                   # モデル配置（Git 管理外）
├─ reports/                  # 検証レポート（Markdown はコミット対象）
├─ logs/                     # 生ログ CSV（Git 管理外）
└─ scripts/
   ├─ Check-Environment.ps1  # Phase 0: 環境診断
   ├─ Start-VoiceChanger.ps1 # Phase 1: 起動と設定チェックリスト
   ├─ Test-Stability.ps1     # Phase 3: 10 分連続計測
   └─ lib/Common.ps1         # 共通関数
```

## フェーズ進行

| フェーズ | 作業 | 完了条件 | 状態 |
| :--- | :--- | :--- | :--- |
| Phase 0 | 環境確認・機材台帳 | 機材・ソフト・モデルを決定 | 未着手 |
| Phase 1 | 初回変換 | 変換後の声が確認できた | 未着手 |
| Phase 2 | 音質調整 | 評価担当 2 名が明瞭さを確認 | 未着手 |
| Phase 3 | 安定性検証 | MVP 条件を満たし手順を文書化 | 未着手 |
| Phase 4 | 追加検討 | 2 キャラ目の運用可否を判断 | 未着手 |

進捗は `docs/00-phase0-environment.md` の台帳と `docs/04-verification.md` の判定欄で管理する。

## 重要な注意

- **このリポジトリは「変換そのもの」を実装しない。** 変換は VC Client が担当し、ここは設定・手順・検証記録を管理する。
- VC Client の変換設定は公開 API で一括投入できないため、設定値は GUI で適用し、その結果を `config/profiles/*.json` に記録する運用とする。
- モデルは利用規約が個別に異なる。「無料だから再配布自由」とは判断しない。採用前に `docs/05-models.md` へ条件を記録する。
- 本番 PA・OBS・DAW・映像同期は本リポジトリの対象外（企画書 2.2）。
