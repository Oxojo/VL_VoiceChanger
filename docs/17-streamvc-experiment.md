# StreamVC の状態保持・計算時間の試作

添付論文 [StreamVC: Real-Time Low-Latency Voice Conversion, v1](https://arxiv.org/abs/2401.03078)
を参考に、20 msごとに状態を引き継ぐニューラルコアを実装した。
**未学習の計算実験であり、声質変換できるStreamVCバックエンドではない。**
Web UIには計測専用画面を用意した。RVCの音声変換設定には接続していない。

## RVCとの違いと期待できる改善

現行RVCは、双方向の特徴抽出器・attentionを使い、前後の文脈を含む窓を繰り返し変換する。
過去の特徴量を一部保持しても、大きな入力窓・先読みは残る。
StreamVCは専用に学習した因果的content encoderとdecoderを持ち、畳み込みの過去状態だけを保持する。
入力は16 kHz / 320 samples（20 ms）、潜在特徴は50 Hz。話者参照音声は開始前に一度だけ符号化する。
RVC `.pth` / ContentVec / RMVPEの重みをそのまま移せる方式ではない。

論文§2.4の60 msは、YINの3フレーム窓と学習時の出力2フレーム遅延を含む構造上の遅延。
Pixel 7の単一CPUコア・XNNPACKでニューラル計算10.8 ms、合計70.8 msと報告されている。
本アプリでの実測値でも、マイク・WebSocket・ブラウザ再生を含む保証値でもない。
現行の700 ms入力蓄積を大きく減らせる設計ではあるが、専用モデルへの置換が必要になる。

## 実装した範囲

`engine/experimental/streamvc.py`:

- SoundStream型content encoder C=64/D=64、speaker encoder C=32/D=64、decoder C=40/D=64。
- ダウンサンプル比2×4×5×8=320、逆順アップサンプル、dilation 1/3/9のresidual unit。
- 話者のattention poolingと、decoder residual unit間のFiLM条件付け。
- 因果畳み込みは履歴とstride位相を保持する。転置畳み込みは末尾を次の入力へoverlap-addし、biasを一度だけ加える。
- 状態は重みと分離し、セッションごとに保持する。reset/flushで解放する。

サンプル単位の任意分割は畳み込みプリミティブで検証。
ニューラルコアの入力契約は完成した320サンプル単位で、WebSocketの分割吸収は未実装。
合成側には9次元のYIN特徴とenergyに相当する10チャネル、および話者embeddingを外部から与える。
実験ではside featuresを0にし、source/referenceには合成乱数を使用する。音声ファイルは出力しない。

**未実装**: YIN・running F0 whitening、学習ラベル生成・GAN学習、論文の学習/出力遅延アラインメント、
学習済み重みの形式/読み込み、録音・再生・音声変換としてのWeb UI接続。
従ってこのコードの出力の音質やレイテンシを、論文のStreamVCのものとして評価できない。

## 実行

通常のエンジン用Python環境（NumPy等とPyTorch、`pip install -e '.[engine]'`）から:

```bash
python scripts/streamvc_probe.py --frames 100 --threads 1 --json logs/streamvc-core.json
# CUDA版PyTorchとGPUがある場合
python scripts/streamvc_probe.py --device cuda --frames 100 --json logs/streamvc-core-cuda.json
python -m unittest discover -s tests -p 'test_streamvc_probe.py' -v
```

初期化・話者参照の符号化・warmupを計測区間から外す。CUDAでは開始/終了を同期する。
100個の20 msフレームを状態付きで処理し、独立した一括処理と数値比較する。
JSONの`neural_ms`はニューラル計算時間のみ。`frames_over_20ms`が多い場合、継続的なリアルタイム処理は難しい。
`real_time_factor < 1`でも音質やend-to-end遅延の合格を意味しない。

## この環境の結果

[生データ](benchmarks/streamvc-neural-core-cpu.json)。CPU: AMD EPYC 9V74、割当5 CPU、PyTorch 2.15.0+cpu、
単一スレッド、FP32、warmup 10回、計測100フレーム。共有クラウド環境なので実行時負荷で変動する。

| 指標 | 結果 |
| --- | ---: |
| ニューラル計算 平均 / p95 | 30.96 / 35.04 ms |
| real-time factor | 1.55 |
| 20 ms超過 | 100 / 100 |
| 保持状態 | 536,776 bytes（約524 KiB、一定） |
| 一括処理との最大絶対誤差 | 4.10e-8 |
| 出力長 | 入力32,000 samplesに対して32,000 samples |

**この実行形態では20 msごとの入力に追いつかない。** 論文のXNNPACK等への最適化が必要。
この数値に60 msを足して「遅延91 ms」と見積もることもできない。処理が入力より遅いので待ち行列が増え続ける。
GPUは利用できず、CUDA計測・音質・実際の変換遅延は未検証。

Python全124件: 成功123、スキップ1、失敗0。追加6件は、stride/転置畳み込みの境界、
一括/逐次一致、未来入力の非参照、状態上限、セッション分離、flush/reset、不正形状を検証した。
スキップは既存の実モデル依存テスト。既存RVC経路の挙動変更はない。

## 公開実装・重みの調査（2026-11-05）

- [公式デモ](https://google-research.github.io/seanet/stream_vc/)は音声例と論文へのリンク。
  確認できたページにコード/重みの配布リンクはなかった。
- [yuval-reshef/StreamVC](https://github.com/yuval-reshef/StreamVC/tree/a81ef600655d04976699ea6c4663ddc81d8692ba):
  READMEで学習済みcheckpointなし、論文どおりのstreaming inferenceは未実装と明記。
  本試作は同実装のコードを実行時に取得・importしない。
- [hrnoh24/stream-vc](https://github.com/hrnoh24/stream-vc): WIP。READMEは学習手順で、推論用重みの案内は確認できなかった。
- [SteamVC/StreamVC_model](https://github.com/SteamVC/StreamVC_model): 学習・checkpoint保存/再開の手順。
  配布済み学習モデルへのリンクは確認できなかった。
- [PSI-TAMU/streamVC](https://github.com/PSI-TAMU/streamVC): 重みの案内があるが、
  [別論文 2406.09277](https://arxiv.org/abs/2406.09277)の話者匿名化方式である。
  重み配布先Google Driveへの接続は、この環境ではHTTP CONNECT 403で失敗した。重み未取得・未実行。

「StreamVCの学習済みモデルが世の中に存在しない」という断定ではなく、今回確認した範囲での結果。

## 実際に声を変えるまで

まず、互換性・ライセンス・品質を確認できる学習済み重みの確保が必要。
入手できない場合は、論文のHuBERT擬似ラベルによるcontent encoder学習と、speaker encoder/decoderのGAN学習が必要になる。
論文は555.15時間・2,311話者のLibriTTS、batch 128・130万stepを使用しており、このCPU環境での小規模学習では再現できない。
実用化には日本語・短い参照音声・ピッチの評価も必要。

重み確保後、YIN/whiteningと学習アラインメントを揃え、一括/逐次の音質を比較する。
次に実行バックエンドを最適化してp95が20 msを下回るか計測する。
最後に専用backendとしてWeb UIへ接続し、debugのcapture→playback遅延と聴感を検証する。

## Web UIで試す

このブランチを取得し、通常どおり起動する:

```bash
git switch research/streamvc-stateful-probe
git pull --ff-only
npm run dev
```

トップ画面の「StreamVC 実験」リンク、または http://localhost:3000/streamvc を開く。
CPUまたはCUDAとCPUスレッド数を選び「処理速度を計測する」を押す。RVCの登録モデルは不要。
CUDA計測はCUDA版PyTorchとGPUが必要。ONNX Runtime CUDAは使わない。
これは**処理速度の計測画面**で、StreamVCによる声質変換・マイク試聴の画面ではない。

平均/p95・20 ms超過数・保持状態・一括処理との誤差を表示し、JSONを保存できる。
計測は別Pythonプロセスで実行して終了時に解放する。最大120秒、同時に一つだけで、RVCの変換とも排他にする。
画面を閉じても開始済み計測は終了またはタイムアウトまで継続する。

追加APIテスト5件を含むPython全129件は成功128・スキップ1・失敗0。
Chromiumからページ遷移→実際のCPU計測100フレーム→結果表示→JSON保存と、CUDA非対応時の表示・再試行を確認した。
GPU実機・音声変換品質は未検証。
