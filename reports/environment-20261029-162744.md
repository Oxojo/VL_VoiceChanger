# 環境確認レポート

- 生成日時: 2026-10-29T16:27:44+09:00
- ホスト名: `Oxojo_Main`
- Python: 3.12.10

## OS / ハードウェア

| 項目 | 値 |
| :--- | :--- |
| OS | Windows 11 |
| OS バージョン | 10.0.26200 |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700H |
| コア / スレッド | 14 / 20 |
| メモリ | 15.6 GB |
| 空きディスク | 21.0 GB |

## GPU

| # | 名前 | VRAM | ドライバ | 取得元 |
| :--- | :--- | :--- | :--- | :--- |
| 0 | NVIDIA GeForce RTX 3060 Laptop GPU | 6.0 GB | 591.86 | nvidia-smi |

> Windows の `Win32_VideoController` の VRAM は 32 bit 制限で 4 GB 超が不正確な場合がある。`nvidia-smi` の値を優先する。

## 音声デバイス

### 入力（録音）

- マイク (C-Media(R) Audio)
- Voicemeeter Out A2 (VB-Audio Voicemeeter VAIO)
- Voicemeeter Out A4 (VB-Audio Voicemeeter VAIO)
- ヘッドセット マイク (ATH-CKD7NC)
- Voicemeeter Out B2 (VB-Audio Voicemeeter VAIO)
- Voicemeeter Out A3 (VB-Audio Voicemeeter VAIO)
- Voicemeeter Out A5 (VB-Audio Voicemeeter VAIO)
- AI Noise-cancelling Input (ASUS Utility)
- Voicemeeter Out B3 (VB-Audio Voicemeeter VAIO)
- マイク (Virtual Desktop Audio)
- Voicemeeter Out A1 (VB-Audio Voicemeeter VAIO)
- Voicemeeter Out B1 (VB-Audio Voicemeeter VAIO)

### 出力（再生）

- Voicemeeter In 2 (VB-Audio Voicemeeter VAIO)
- AI Noise-cancelling Output (ASUS Utility)
- Voicemeeter VAIO3 Input (VB-Audio Voicemeeter VAIO)
- Voicemeeter In 5 (VB-Audio Voicemeeter VAIO)
- Voicemeeter In 1 (VB-Audio Voicemeeter VAIO)
- Voicemeeter In 4 (VB-Audio Voicemeeter VAIO)
- スピーカー (C-Media(R) Audio)
- Voicemeeter AUX Input (VB-Audio Voicemeeter VAIO)
- Voicemeeter Input (VB-Audio Voicemeeter VAIO)
- Voicemeeter In 3 (VB-Audio Voicemeeter VAIO)
- ヘッドホン (ATH-CKD7NC)

## VC Client

| 項目 | 値 |
| :--- | :--- |
| 配置フォルダ | `C:\Tools\vcclient_win_cuda` |
| フォルダ存在 | True |
| 起動スクリプト | あり |
| エディション（推定） | cuda |
| バージョン（推定） | 2.1.4-alpha |
| ポート 18000 | 閉 |
| 配下の .pth | 2 件 |

> エディションとバージョンは推定。実際の配布名は GUI の Server Info で確認し、docs/00-phase0-environment.md に記録する。

## 検出した注意点

- なし

## 次の作業（Phase 0 完了条件）

- [ ] 機材台帳（マイク / オーディオ IF / ヘッドホン / PC）を確定した
- [ ] VC Client の採用エディション・バージョンを確定し、ダウンロード元を記録した
- [ ] 使用する RVC モデル 1 種類を確定し、利用規約を記録した（docs/05-models.md）
- [ ] 上記を docs/00-phase0-environment.md に転記した
