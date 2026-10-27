# Phase 0: 環境確認と機材台帳

企画書 6 章 Phase 0 に対応。**「使用機材と採用ソフト・モデルを決定できた」** 状態をゴールとする。

このフェーズで決めた内容が、以降のすべての前提になる。曖昧なまま Phase 1 に進むと、後から「どの版で動かしたか分からない」「別の部員が再現できない」という問題になる。

---

## 1. 自動診断を実行する

```powershell
.\scripts\Check-Environment.ps1
```

`reports/environment-latest.md` に次が出力される。

- OS / CPU / メモリ / 空きディスク
- GPU と VRAM（`nvidia-smi` がある場合は実測値）
- 録音・再生デバイスの一覧
- VC Client の配置状況・エディション推定・ポート状態
- 自動判定した注意点

> エディションとバージョンは **フォルダ名からの推定**。確定値は VC Client 起動後の GUI「Server Info」で確認する。

## 2. 機材台帳を埋める

`config/settings.json` の `vcClient.root` を実際のパスに直したうえで、下表を記入する。

### PC

> 下記は `Check-Environment.ps1` を **2026-10-28 に検証機で実行した実測値**。別の PC を使う場合は上書きする。

| 項目 | 実測 / 型番 | 備考 |
| :--- | :--- | :--- |
| 機材の所有者 | （未記入） | 部員個人 / サークル共用 |
| 型番 | （未記入） | ノート PC の場合、型番も残す |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700H（14 コア / 20 スレッド） |  |
| メモリ | 15.6 GB |  |
| GPU / VRAM | NVIDIA GeForce RTX 3060 Laptop GPU / 6 GB（driver 591.86） | RVC は VRAM 4 GB 以上が望ましい |
| OS / ビルド | Windows 11 Home / 10.0.26200 (Build 26200) |  |
| 空きディスク | 32.5 GB | VC Client 3.5 GB ＋ 展開後 ＋ モデルで **10 GB 以上**は確保したい |
| 内蔵オーディオ | C-Media(R) Audio（マイク / スピーカー） | USB マイク・ヘッドホンも認識済み |

**この PC で検出された音声デバイスの注意点**

- **VB-Audio Voicemeeter がインストールされている**（`Voicemeeter In/Out *` が多数）。
  仮想デバイスが既定になっていると「音が出ない」「二重に聞こえる」の原因になる。VC Client の input / output は **物理デバイス名（マイク / スピーカー）を明示指定する**。
- 「マイク (Virtual Desktop Audio)」「AI Noise-cancelling Input/Output (ASUS Utility)」も存在する。**名前が似ているため、選び間違えに注意。**
- 詳細な一覧は `reports/environment-latest.md` を参照。

### 音声機器

| 機器 | 型番 | 接続 | 保有 | 備考 |
| :--- | :--- | :--- | :--- | :--- |
| マイク |  | XLR / USB | 有・無 |  |
| オーディオ IF |  |  | 有・無 | XLR マイク使用時のみ必須 |
| 確認用ヘッドホン |  | 有線 | 有・無 | 変換後の声を聴く担当が使用 |
| 予備ケーブル |  |  | 有・無 |  |

**不足品と調達方法**（企画書 5 章：追加費用 0 円は目標であり保証値ではない）

| 不足品 | 調達方法 | 期限 | 状態 |
| :--- | :--- | :--- | :--- |
|  |  |  |  |

## 3. ソフトウェア台帳

| 項目 | 記録内容 |
| :--- | :--- |
| ソフト名 | VC Client（w-okada/voice-changer） |
| 配布ファイル名 | 例: `vcclient_win_cuda_2.1.4-alpha.zip`（**ファイル名をそのまま記録する**） |
| エディション | cuda / dml（**std は RVC 非対応**） |
| バージョン | 例: 2.1.4-alpha |
| ダウンロード元 URL | https://huggingface.co/wok000/vcclient000 |
| 取得日 |  |
| 配置パス | 例: `C:\Tools\vcclient_win_cuda` |
| 起動方法 | `start_http.bat` |
| GUI の URL | http://localhost:18888/ |
| 備考 | パスに Unicode を含めない（ONNX Runtime が失敗する報告あり） |

## 4. モデルを 1 種類に絞る

候補を比較し、**1 種類だけ** 採用する。手順と規約記録は [05-models.md](05-models.md) を使う。

- [x] 利用規約 URL を確認した（2026-10-28、最終更新 2026-10-07 のページを確認）
- [x] 必要なクレジット表記を確認した（**変換元の声＋モデル名の 2 つ**）
- [x] 禁止用途を確認した
- [x] 部内共有・改変の可否を確認した
- [x] モデルファイルの取得元とハッシュを記録した（`models/tsukuyomi/`、SHA-256 は [05-models.md](05-models.md)）

## 5. PC 側の事前設定

| 項目 | 対応 | 済 |
| :--- | :--- | :--- |
| ウイルス対策の除外 | VC Client フォルダを除外（初回起動が極端に遅い / GUI が白くなる場合の原因） | [ ] |
| 電源プラン | 「高パフォーマンス」または「バランス」で、スリープを無効化 | [ ] |
| GPU ドライバ | 最新であることを確認 | [ ] |
| 不要アプリの停止 | ブラウザのタブ、ゲーム、配信ソフトを終了 | [ ] |
| Windows の音量 | マイクの「マイクブースト」を上げすぎない（割れの原因） | [ ] |
| 既定のデバイス | VC Client の input / output と取り違えないよう控える | [ ] |

## 6. Phase 0 完了条件

- [ ] 機材台帳を記入し、不足品と調達方法を決めた（PC 欄は記入済み。**残りは現物確認**）
- [x] VC Client のエディション・バージョン・ダウンロード元を確定した（`cuda` / 2.1.4-alpha / HuggingFace `wok000/vcclient000`）
- [ ] `config/settings.json` の `vcClient.root` を実パスに更新した
- [x] 採用モデル 1 種類とその利用条件を `docs/05-models.md` に記録した
- [x] `reports/environment-latest.md` を残した

すべて満たしたら [01-setup.md](01-setup.md) へ進む。
