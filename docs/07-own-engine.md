# 自前音声変換エンジン（engine/）

VC Client の内部 API に依存せず、**音声変換そのものを自前コードで実行する**ための実装です。
外部サービスや VC Client サーバーは使いません。推論は Python + PyTorch（RVC v2 合成器）と
ONNX Runtime（ContentVec / RMVPE）で行います。

- 対象ブランチ: `feature/self-engine`
- 追加日: 2026-10-28
- 実装: `engine/`（Python）、テスト: `tests/test_engine.py`

## 1. なぜ自前実装か

VC Client は変換までの経路が GUI 依存で、ファイル変換 API も本ビルドでは不具合が出ていました
（`docs/04-verification.md` の E10）。検証・本番のどちらでも「入力 WAV → 出力 WAV」を
コマンドだけで再現できる状態にするため、変換器そのものを自前実装しました。

| 項目 | VC Client（参照実装） | 自前エンジン |
| --- | --- | --- |
| 変換の起動 | GUI の [start] のみ | CLI 1 行 |
| ファイル変換 | 本ビルドでは 500 エラー | 対応 |
| パラメータの再現性 | GUI 状態に依存 | 引数・JSON に記録 |
| 処理の可視化 | 不可 | 中間値・所要時間を出力 |
| 学習済み重み | 利用 | 同じものを利用（`models/`） |

学習済みの重み（ContentVec / RMVPE / RVC モデル）はデータとして再利用し、
**推論コードと音声処理はすべて自前**です。

## 2. パイプライン

```
入力 WAV
  └─ 16 kHz モノラル化（resample_poly）
      └─ 48 Hz ハイパス（Butterworth 5 次・ゼロ位相）
          └─ 前後 80 サンプルを反射パディング
              ├─ RMVPE（ONNX）        → F0 [Hz] 100 fps
              │    └─ 無声区間を線形補間 → ピッチシフト → メル尺度で 1-255 に量子化
              └─ ContentVec（ONNX）    → 768 次元特徴 50 fps
                   └─ 2 倍に展開（100 fps）
                       └─ RVC v2 合成器（自前 PyTorch 実装）
                            enc_p（相対位置付き Transformer）
                            flow（ResidualCoupling × 8 + Flip）
                            dec（HiFi-GAN + NSF 音源）
                        → 40 kHz 波形
  └─ パディング除去 → 音量を入力に寄せる（RMS ミックス） → 出力 WAV
```

音源（NSF）は F0 から正弦波を生成するため、**ピッチは入力に追従**します。
話者性は ContentVec 特徴量と話者埋め込み（`emb_g`）で決まります。

### 実装ファイル

| ファイル | 役割 |
| --- | --- |
| `engine/audio.py` | WAV I/O、リサンプル、ハイパス、RMS エンベロープ |
| `engine/features/contentvec.py` | ContentVec ONNX ラッパ（layer9 / layer12） |
| `engine/features/rmvpe.py` | RMVPE ONNX ラッパ（F0 を直接出力） |
| `engine/model/commons.py` | マスク生成、自作 weight norm（`weight_g` / `weight_v`） |
| `engine/model/attentions.py` | 相対位置付き MultiHeadAttention / FFN / Encoder |
| `engine/model/modules.py` | WN、ResBlock1、ResidualCoupling、PosteriorEncoder |
| `engine/model/synthesizer.py` | TextEncoder / SineGen / GeneratorNSF / SynthesizerTrn |
| `engine/model/loader.py` | `.pth` の config 解釈と重み読み込み |
| `engine/convert.py` | 変換パイプライン本体 |
| `engine/cli.py` | `inspect` / `convert` / `analyze` |

## 3. セットアップ

```powershell
# Python 3.12（3.14 は PyTorch 非対応のため使わない）
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r engine\requirements.txt

# GPU（RTX 3060 等）を使う場合は CUDA 版 torch に入れ替える
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cu126 `
    --force-reinstall --no-deps "torch==2.14.1+cu126"
```

`.venv/` は Git 管理外です（`.gitignore`）。

ONNX モデルの場所は次の順で探します（`engine/config.py`）。

1. 環境変数 `VL_CONTENTVEC_ONNX` / `VL_RMVPE_ONNX`
2. `modules/contentvec/contentvec-f.onnx`、`modules/rmvpe/rmvpe_20231006.onnx`
3. `C:\Tools\vcclient_win_cuda\modules\...`（検証機の既定）

## 4. 使い方

```powershell
# モデルの構成を確認
.\.venv\Scripts\python.exe -m engine inspect -m models\tsukuyomi\tsukuyomi-normal1.pth

# 変換（既定: ピッチ 0、40 kHz 出力）
.\.venv\Scripts\python.exe -m engine convert `
    -m models\tsukuyomi\tsukuyomi-normal1.pth `
    -i logs\verify\tts-input-ja.wav `
    -o logs\verify\own-ja.wav `
    --json logs\verify\own-ja.json

# 出力の F0・音量を確認
.\.venv\Scripts\python.exe -m engine analyze -i logs\verify\own-ja.wav

# 分割変換の速度（ライブ成立するか）を計測
.\.venv\Scripts\python.exe -m engine bench `
    -m models\tsukuyomi\tsukuyomi-normal1.pth `
    -i logs\verify\tts-input-ja.wav `
    -o logs\verify\own-ja-chunk.wav `
    --hop 0.5 --context 0.15 --device cuda
```

`scripts/Convert-Audio.ps1` からも同じことができます（引数は CLI に素通し）。

### 主なオプション

| オプション | 既定 | 内容 |
| --- | --- | --- |
| `--pitch` | `0` | ピッチシフト（半音）。+12 で 1 オクターブ上 |
| `--speaker` | `0` | 話者 ID（多話者モデル用。つくよみちゃんは 0） |
| `--rms-mix` | `0.25` | 音量を入力に寄せる強さ（0 で入力と同一、1 で変換結果のまま） |
| `--seed` | `1234` | 乱数シード（`-1` で毎回ランダム） |
| `--output-sr` | モデル準拠 | 出力サンプリングレート |
| `--device` | `cpu` | `cuda` で GPU 推論 |

## 5. 実測結果（2026-10-28 / CPU）

入力: `logs/verify/tts-input-ja.wav`（22.05 kHz / 16 bit / 7.63 秒 / SAPI Haruka）
モデル: `tsukuyomi-normal1.pth`（RVC v2 / 40 kHz / 27.54 M パラメータ / 話者 109）

| 項目 | 結果 |
| --- | --- |
| 出力 | `logs/verify/own-ja.wav` 40 kHz / 7.61 秒 / 762 フレーム |
| 読み込み | 4.8 秒（初回・モデル + 2 つの ONNX） |
| F0 推定（RMVPE） | 0.40 秒 |
| 特徴量（ContentVec） | 1.00 秒 |
| 合成（torch / CPU） | 7.65 秒 |
| 合計 | 9.14 秒（入力 7.63 秒 → 実時間比 1.20 倍） |
| F0 追従（log F0 の相関） | **0.999**（中央値 244.6 Hz → 243.4 Hz） |
| 有声率 | 52.4 % → 52.6 % |
| エネルギー包絡の相関 | **0.971** |
| RMS | 0.1154 → 0.1155（`--rms-mix 0.25`） |
| ピーク | 0.648（クリップなし） |
| ピッチシフト +12 | 中央値 223.9 Hz → 447.9 Hz（ちょうど 2 倍） |
| スペクトル平坦度 | 入力 0.44 → 出力 0.25（より有声的） |
| 調波性（自己相関ピーク） | 入力 0.755 → 出力 0.720 |

「F0 が入力に追従し、エネルギー包絡も一致し、クリップせず有声的」であることを
数値で確認しました。声質（話者性）の最終判断は聴取で行います。

### CPU での位置づけ

実時間比 1.2 倍ではライブ用途には足りないため、GPU 実行を既定の選択肢にします
（`--device cuda`）。GPU 実測は `docs/04-verification.md` の E11 に追記します。

## 6. 制約と今後の課題

- 対応形式は **RVC v2（768 次元特徴）のみ**。v1（256 次元）は未対応。
- 検索インデックス（FAISS）による声質ブレンドと、それに付随する子音保護（protect）は未実装。
- `ResBlock2` 構成のモデルは未対応（配布モデルは `'1'`）。
- 現状は**ファイル一括変換**と**分割変換**（`engine/realtime.py` の `ChunkedConverter`）。
  マイク → 変換 → スピーカーのデバイス接続は未実装（同じ `ChunkedConverter` を呼ぶだけで載る）。
- 合成器の推論は CPU で実時間と同程度。GPU 併用を前提にする。
- 声質の評価は聴取が必要（自動評価は F0・包絡・調波性まで）。

## 7. 参照した仕様とライセンス

- RVC（Retrieval-based Voice Conversion WebUI, MIT）の**モデル構造と前処理仕様**を参照し、
  推論コードは本リポジトリ用に書き直しました。重みのキー名・テンソル形状は
  学習済みモデルとの互換性のために一致させています。
- ContentVec / HuBERT（MIT）、RMVPE（MIT）の ONNX をデータとして利用。
- つくよみちゃん RVC モデルの利用条件は `docs/05-models.md` を参照してください。
