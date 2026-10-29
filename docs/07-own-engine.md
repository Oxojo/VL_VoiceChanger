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
          └─ （任意）ノイズ除去: 電源ハムのノッチ → スペクトルサブトラクション
              └─ 前後 80 サンプルを反射パディング
                  ├─ RMVPE（ONNX）        → F0 [Hz] 100 fps
                  │    └─ 無声区間は 0 Hz を維持 → ピッチシフト → メル尺度で 1-255 に量子化
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
| `engine/denoise.py` | ノイズ除去（スペクトルサブトラクション）とハム除去 |
| `engine/cli.py` | `inspect` / `convert` / `analyze` / `bench` / `denoise` / `serve` |

## 3. セットアップ

どの OS でも同じコマンドでよい（Python 3.12 推奨。3.14 は PyTorch 非対応のため使わない）。

```bash
# 仮想環境 .venv と依存関係（GPU を使う場合は --cuda）
python -m vlvc setup
python -m vlvc setup --cuda    # CUDA 版 torch（cu126）に入れ替える
```

```text
従来の手動セットアップ:
  python -m venv .venv
  .venv/Scripts/python.exe -m pip install -r engine/requirements.txt   # Windows
  .venv/bin/python -m pip install -r engine/requirements.txt           # macOS / Linux
```

`.venv/` は Git 管理外です（`.gitignore`）。

ONNX モデルの場所は次の順で探します（`engine/config.py`）。

1. 環境変数 `VL_CONTENTVEC_ONNX` / `VL_RMVPE_ONNX`
2. `modules/contentvec/contentvec-f.onnx`、`modules/rmvpe/rmvpe_20231006.onnx`
3. `C:\Tools\vcclient_win_cuda\modules\...`（検証機の既定）

## 4. 使い方

```bash
# モデルの構成を確認
python -m engine inspect -m models/tsukuyomi/tsukuyomi-normal1.pth

# 変換（既定: ピッチ 0、40 kHz 出力）
python -m vlvc convert -m models/tsukuyomi/tsukuyomi-normal1.pth \
    -i logs/verify/tts-input-ja.wav -o logs/verify/own-ja.wav \
    --json logs/verify/own-ja.json

# 出力の F0・音量を確認
python -m engine analyze -i logs/verify/own-ja.wav

# 入力と出力を数値で比較（F0 追従・包絡・調波性・クリップ）
python -m vlvc quality logs/verify/tts-input-ja.wav logs/verify/own-ja.wav

# 分割変換の速度（ライブ成立するか）を計測
python -m engine bench -m models/tsukuyomi/tsukuyomi-normal1.pth \
    -i logs/verify/tts-input-ja.wav -o logs/verify/own-ja-chunk.wav \
    --hop 0.5 --context 0.15 --crossfade 0.05 --device cuda
```

`vlvc convert` は `python -m engine convert` に引数をそのまま渡します。
`vlvc selftest` で「変換 → 解析 → 品質 → 分割変換」をまとめて実行し、`reports/own-engine-latest.md` を更新できます。

### 主なオプション

| オプション | 既定 | 内容 |
| --- | --- | --- |
| `--pitch` | `0` | ピッチシフト（半音）。+12 で 1 オクターブ上 |
| `--speaker` | `0` | 話者 ID（多話者モデル用。つくよみちゃんは 0） |
| `--rms-mix` | `0.25` | 音量を入力に寄せる強さ（0 で入力と同一、1 で変換結果のまま） |
| `--seed` | `1234` | 乱数シード（`-1` で毎回ランダム） |
| `--output-sr` | モデル準拠 | 出力サンプリングレート |
| `--device` | `cpu` | `cuda` で GPU 推論 |
| `--hop` | `0.5` | 分割変換で 1 回に進める長さ（秒） |
| `--context` | `0.15` | 分割変換で前後に付ける文脈（秒） |
| `--crossfade` | `0.05` | 分割変換の継ぎ目のクロスフェード長（秒） |
| `--denoise` | `0` | ノイズ除去の強さ 0〜1（`0.5` 前後が無難） |
| `--dehum` | `0` | 電源ハム除去（`50` / `60` Hz、`0` で無効） |

## 5. 実測結果（2026-10-28 / CPU）

入力: `logs/verify/tts-input-ja.wav`（22.05 kHz / 16 bit / 7.63 秒 / SAPI Haruka）
モデル: `tsukuyomi-normal1.pth`（RVC v2 / 40 kHz / 27.54 M パラメータ / 話者 109）

| 項目 | 結果 |
| --- | --- |
| 出力 | `logs/verify/own-ja.wav` 40 kHz / 7.63 秒 / 763 フレーム |
| 読み込み | 4.8 秒（初回・モデル + 2 つの ONNX） |
| F0 推定（RMVPE） | 0.40 秒 |
| 特徴量（ContentVec） | 1.00 秒 |
| 合成（torch / CPU） | 7.65 秒 |
| 合計 | 9.14 秒（入力 7.63 秒 → 実時間比 1.20 倍） |
| F0 追従（log F0 の相関） | **0.999**（中央値 244.6 Hz → 243.8 Hz） |
| 有声率 | 52.4 % → 52.0 % |
| エネルギー包絡の相関 | **0.966** |
| RMS | 0.1176 → 0.1135（`--rms-mix 0.25`） |
| ピーク | 0.583（クリップなし） |
| 調波性（自己相関ピーク） | 0.960 → 0.968 |
| ピッチシフト +12 | 中央値 223.9 Hz → 447.9 Hz（ちょうど 2 倍） |

「F0 が入力に追従し、エネルギー包絡も一致し、クリップせず有声的」であることを
数値で確認しました。声質（話者性）の最終判断は聴取で行います。
この比較は `python -m vlvc quality` で再現できます。

### 分割変換（ライブ想定）

`--hop 0.5 --context 0.15 --crossfade 0.05` で 7.63 秒を 16 区間に分けた結果（CPU）:

| 項目 | 一括変換 | 分割変換（CPU） |
| --- | --- | --- |
| 全区間の処理時間 | 9.14 秒 | 69.5 秒 |
| RTF（処理 / 音声） | 1.20 | 9.11 |
| 区間あたり遅延 | — | 平均 4,346 ms / 最大 4,670 ms |
| F0 追従 | 0.999 | **0.999** |
| エネルギー包絡の相関 | 0.966 | **0.969** |
| 継ぎ目のサンプル跳び（平均） | — | 0.0050（全体平均 0.0158） |
| 継ぎ目のサンプル跳び（最大） | — | 0.0329 |

- 区間の切り出しを **10 ms フレーム単位**に量子化し、特徴量を `p_len` に揃えることで、
  出力長を `フレーム数 × サンプリングレート / 100` に一致させました。
  これがないと区間ごとに最大 20 ms ずれ、後半ほど時間ずれが累積します
  （修正前は F0 追従 0.795 / 包絡 0.748 まで低下）。
- 継ぎ目は 50 ms のクロスフェードでつなぎます。無い場合は継ぎ目の平均跳びが
  0.0587（全体平均の約 4 倍）まで出てクリックの原因になります。
- CPU では区間ごとの固定費（ONNX 呼び出し・合成器の起動）が支配的で RTF 9.1。
  ライブ成立には GPU が必須で、実測は `docs/04-verification.md` の E11 に追記します。

### GPU（CUDA 版 torch）での実測

`torch 2.14.1+cu126` を導入して同じ入力を `--device cuda` で変換した結果:

| 条件 | 処理時間 | RTF |
| :--- | ---: | ---: |
| 一括・初回（カーネル初期化を含む） | 7.12 秒 | 0.93 |
| 一括・2 回目以降（モデル常駐） | 2.07〜3.13 秒 | **0.27〜0.41** |
| 分割 16 区間（hop 0.5 / context 0.15 / crossfade 0.05） | 12.5〜15.3 秒 | 1.64〜2.08 |

- 一括変換は **GPU で実時間より速い**（7.63 秒の音声を 2〜3 秒で処理）。
- 分割変換は 1 区間あたり 0.8〜1.3 秒かかり、ライブ成立条件（最大遅延 < hop）は満たさない。
  区間ごとの固定費（ContentVec の ONNX が CPU 実行、合成器の起動）が支配的なため、
  ライブ化には ContentVec の GPU 実行（ONNX Runtime GPU / TensorRT）が必要。

### ブラウザから使う

`python -m engine serve`（または `python -m vlvc serve`）で Web UI が起動する。
音声のドロップ／マイク録音・設定・試聴・数値表示・WAV 保存までブラウザで完結する。
API と画面の説明は [08-web-ui.md](08-web-ui.md)。

## 6. ノイズ除去（入力の前処理）

`engine/denoise.py` に自前実装した前処理。学習済みモデルは使わず、**スペクトルサブトラクション**で
空調・ファン・環境音などの定常ノイズを抑える。電源ハム（50 / 60 Hz）はノッチフィルタで落とす。

```
16 kHz 波形 → STFT（Hann / 1024 点 / hop 256）
            → 周波数ごとに「時間方向で静かな方（15 パーセンタイル）」をノイズと推定
            → 過減算したマスク（強さ 0〜1）＋ 時間方向の平滑化（開く時は速く、閉じる時はゆっくり）
            → ISTFT
```

- 変換の前処理として使う: `--denoise 0.6 --dehum 50`
- ノイズ除去だけを試す: `python -m engine denoise -i 入力.wav -o 出力.wav --strength 0.7 --dehum 50`
- **入力音声にだけかけてから変換する**（Web UI の「ノイズ除去を試す」ボタン、
  または `POST /api/denoise`）。結果を試聴してから変換できる。適用後は二重がけを避けるため
  変換時の除去は 0 に戻る
- 分割変換では、区間ごとではなく**全体に 1 回**かける（区間ごとだとノイズ推定がばらつき、
  継ぎ目で音色が変わるため）。

### 効果の実測（2026-10-29）

`tts-input-ja.wav` に白色ノイズ（約 -22 dBFS）と 50 Hz ハムを加えた入力で比較。

| 項目 | 除去なし | 除去あり（0.7 + 50 Hz） |
| :--- | ---: | ---: |
| 入力のノイズフロア | -23.4 dBFS | **-39.0 dBFS** |
| 変換出力のノイズフロア | -59.8 dBFS | **-62.5 dBFS**（クリーン入力は -61.4） |
| 有声率（クリーン入力は 51.8 %） | 56.8 % | **54.7 %** |
| 包絡相関（クリーン入力の変換結果と比較） | 0.956 | **0.962** |
| クリーン入力の変換結果との差（RMS） | 0.154 | **0.131** |

処理時間は 7.63 秒の音声で 0.05〜0.06 秒（実時間の 1 % 未満）。

### 限界

- 定常ノイズ向け。**話している最中に重なる非定常ノイズ**（紙ずれ、笑い声、隣の会話）は落ちない。
- 強くかけすぎると声の倍音まで削れて**こもる**。まず `0.5`、物足りなければ `0.7` まで。
- 無音がデジタル無音（-∞ dB）の素材ではノイズ推定が効かないが、実害も無い。

## 7. 制約と今後の課題

- 対応形式は **RVC v2（768 次元特徴）のみ**。v1（256 次元）は未対応。
- 検索インデックス（FAISS）による声質ブレンドと、それに付随する子音保護（protect）は未実装。
- `ResBlock2` 構成のモデルは未対応（配布モデルは `'1'`）。
- 現状は**ファイル一括変換**と**分割変換**（`engine/realtime.py` の `ChunkedConverter`）。
  マイク → 変換 → スピーカーのデバイス接続は未実装（同じ `ChunkedConverter` を呼ぶだけで載る）。
- 分割変換は 10 ms フレーム単位の切り出しと 50 ms のクロスフェードで継ぎ目を処理済み。
  残る課題は速度で、GPU でも RTF 1.64〜2.08（区間遅延 最大 2.1 秒）。
  ContentVec の ONNX を GPU 実行にするのが次の一手。
- 合成器の推論は CPU で実時間と同程度。GPU 併用を前提にする。
- 声質の評価は聴取が必要（自動評価は F0・包絡・調波性まで）。

## 8. 参照した仕様とライセンス

- RVC（Retrieval-based Voice Conversion WebUI, MIT）の**モデル構造と前処理仕様**を参照し、
  推論コードは本リポジトリ用に書き直しました。重みのキー名・テンソル形状は
  学習済みモデルとの互換性のために一致させています。
- ContentVec / HuBERT（MIT）、RMVPE（MIT）の ONNX をデータとして利用。
- つくよみちゃん RVC モデルの利用条件は `docs/05-models.md` を参照してください。
