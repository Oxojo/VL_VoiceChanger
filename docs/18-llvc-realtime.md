# LLVCで単一ターゲットへの低遅延マイク変換を試す

Voice Mixing計画の段階1として、[KoeAI/LLVC](https://github.com/KoeAI/LLVC) の推論コードを組み込んだ。
これはStreamVCとは別のモデルで、LLVC配布checkpointの固定ターゲット声に変換する。
Voice Mixing・任意のRVCモデル・話者参照音声・ピッチ変更には対応していない。

**本環境ではモデル配布先がネットワーク制限で403となり、公式学習済み重みでの実推論は未検証。**
Web UIにマイク変換の経路を実装したが、音質・実モデルの速度・100ms以下の達成はまだ確認できていない。
未学習の試験用モデルで「声質変換できた」とするものではなく、製品コードには代替モデルがない。

## 試し方

```bash
git switch research/streamvc-stateful-probe
git pull --ff-only
npm run dev
```

1. トップ画面の「変換方式」で **LLVC** を選ぶ。
2. 初回のみ **LLVCモデルを取得** を押す。配布元は [KoeAI/llvc](https://huggingface.co/KoeAI/llvc)。
3. CPUを基準に、必要ならCUDAを選ぶ。**debugモードをオン**にする。
4. ヘッドホンを使い、**マイク変換を開始**。変換音声を逐次再生する。
5. 停止すると未処理の端数を変換・再生し、終了する。比較結果は診断JSONで保存できる。

`npm run dev`で通常どおりPython/Web依存を準備する。必要な`torch` / `onnx`はengine依存として自動導入する。
RVC用ONNX、SpeechBrain、fairseq、torchaudioはLLVC推論に使用しない。
CPU推論はセッション中1スレッドにし、終了・切断時に元の設定へ戻す。CUDAにはCUDA版PyTorchが必要。

ネットワーク取得に失敗する場合、[公式checkpoint](https://huggingface.co/KoeAI/llvc/resolve/main/models/checkpoints/llvc/G_500000.pth?download=true)
を `models/llvc/G_500000.pth` に置く方法も使える。違うモデルやRVCの`.pth`では動作しない。
配置検出はファイルの存在確認で、互換性は開始時の厳密なstate_dictロードとウォームアップで検証する。
モデルの利用条件は配布元を確認する。コードのMITライセンスと重みの利用条件は別に扱う。

## 何を接続したか

- `engine/llvc/model.py` / `cached_convnet.py`: 上流revision `1627c5d358cf9bb2b92b0ccc513d8b36807c923d` の推論コード。
  相対importへ変更し、SpeechBrainの位置符号化のみ互換の小さな実装へ置き換えた。MIT表示を同梱。
- `config.json`: 上流の `experiments/llvc/config.json`。16kHz、L=16、decoder chunk=13、内部208 samples（13ms）。
- `streaming.py`: 公式`infer.py`のstreaming入力規約（先頭Lシフト、過去2Lのcontext）を保持。
  任意のパケット境界を吸収し、encoder/decoder/prenet/outputの状態をセッションごとに保持する。
  未処理末尾だけをゼロ埋めし、返す音声は入力の正確なサンプル数へ切り詰める。
- モデルは`torch.load(weights_only=True)`と`load_state_dict(strict=True)`でロードし、`eval()` / `inference_mode()`で実行する。
  重みを取得できない/不正な場合は開始を失敗させる。
- 起動時にウォームアップし、状態と統計を初期化してからマイクを開く。
- `/api/stream`の`backend: "llvc"`で選択する。RVCは従来どおり既定値。
- マイク取得・16kHz化・20msパケット送信・逐次再生は既存の経路を利用する。
  再生開始時のクッションはLLVCだけ15ms（RVCは50ms）。ACKで入力の送信量を制御する。
  他の変換と排他、時間・待ち行列の上限も共通。
- 10msフレーム単位の診断値にはLLVCの13msを小数で表現する。
  サンプル数の比較は小数演算の誤差を許容し、実際の音声長の不一致は引き続きエラーにする。

## モデル取得の検証

`POST /api/llvc/install`は固定した配布リポジトリのmetadataからrevision・LFS SHA256・サイズを取得する。
そのrevisionのcheckpoint一つだけを取得し、サイズ/SHA256を照合してから置き換える。
上限512MiB、TLS検証有効、一時ファイルを削除し、検証失敗時は既存checkpointを保持する。
ダウンロードmanifestは `models/llvc/G_500000.json` に保存する。
ファイル変換・マイク変換・他のモデル取得と排他。別サイトのOriginからの取得開始は拒否する。

## 実モデルで速度を再現するコマンド

```bash
python -m engine.llvc --download
python -m engine.llvc --input input.wav --output logs/llvc-output.wav --device cpu
```

実checkpointを使い、20ms入力のファイル・ストリーミングを再現する。
入力/出力サンプル数、内部13ms単位の平均/p95推論時間、処理RTFを表示する。
ファイル試験は録音・ネットワーク・再生を含まないので、Web UIの診断と併用する。

## 100msの評価

目標はマイク入力から実際の出力までであり、内部13msだけで達成とはしない。
Web debugの推定値に加え、途切れ、送信/再生待ちの増加、p95、実デバイスの遅延を確認する。
継続的なRTF<1が必要。再生待ちが増えたり途切れたりする設定では、低遅延の合格としない。
物理的な100ms以下の検証には、既知の入力と出力を外部ループバック等で同時に記録して比較する必要がある。
ブラウザのcapture→scheduled playback推定には未測定のマイク/OS遅延がある。

## 検証と未完了項目

Python全144件は成功142・スキップ2・失敗0。スキップは既存RVCモデル依存と公式LLVC重み依存。
JavaScript14件成功。Chromiumでは既存15件とLLVC用1件が成功。
LLVC用ブラウザ試験は、セットアップ→マイク取得→WebSocket→公式アーキテクチャの推論→再生→debug JSON→停止/再開を通す。
ここでは**テストプロセス内限定の未学習重み**を使っており、声質変換・音質・速度の検証ではない。
分割入力・端数・公式逐次手順との一致・異常出力・モデル欠落・切断時解放・SHA256不一致もテストした。

現在の外部ブロッカー:

- Hugging FaceへのHTTPS CONNECTが403。公式checkpoint未取得、音質/日本語/GPU/実モデルの継続速度/100ms未検証。
- GitHub Issue作成も `https://api.github.com/graphql` の403で未完了。
  [Issue本文](issues/low-latency-voice-mixing.md)は保存済み。Issue作成済みとは扱わない。
- 環境設定の変更案へ、既存の`api.github.com`を残して `huggingface.co`、`cdn-lfs.huggingface.co`、
  `cdn-lfs-us-1.hf.co`、`cas-bridge.xethub.hf.co` を追加した。保存だけでは現在のネットワーク設定は変わらない。
  このクラウドで続きを検証するには、環境設定の変更案を確認・保存し、環境を公開した後、接続を再確認する。
  利用者の手元で通常のインターネット接続から試す場合、このクラウド設定の変更は必要ない。

## CPUで「入力に追いつかない」と表示される場合

CPU向けに[FP32 ONNX Runtimeでの高速化](19-llvc-cpu-runtime.md)を追加した。
開発サーバーをCtrl+Cで止め、`git pull --ff-only` → `npm run dev`で起動し直して画面を更新する。
既存のモデルを使うので、重みの再取得は不要。初回だけ、マイクを開く前にONNX生成と数値検証を行う。
開始後の実行方式が **LLVC: ONNX Runtime CPU / FP32** と表示されることを確認する。
処理区間13ms・入力送信20ms・再生クッション15ms・待ち行列上限は変えていない。
継続して停止する場合、debug JSONの`settings.runtime`、推論RTF・推論時間・ACK往復・入力待ちを確認する。

## GPUで処理が追いつかない場合

[提供ログの解析とCUDA Graphの試験対応](20-llvc-cuda-graph.md)を参照。
LLVCの「自動」はCPU ONNXを優先する。CUDA明示選択では、マイク開始前にgraphの数値検証と容量計測を行う。
このクラウドにはGPUがなく、実機での動作・速度は未検証。
