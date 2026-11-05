# LLVC GPUの処理遅延とCUDA Graphの試験対応

## 提供ログの所見

ユーザー提供の `vlvc-stream-debug (2).json` は、LLVC / CUDA / `PyTorch CUDA`、20ms入力パケット、45出力。

| 指標 | 結果 |
| --- | ---: |
| 推論時間 / 20ms入力パケット（平均） | 60.63 ms |
| 推論時間 / パケット（p95） | 86.88 ms |
| ワーカー実行待ち（平均） | 0.18 ms |
| ACK往復 − サーバー時間（平均） | 約1.26 ms |
| 終了直前のブラウザ内送信待ち | 1,919.7 ms |
| 停止時の待ち音声量 | 2,020 ms |
| 終了時推論RTF | 3.03 |

推論が所要時間の大半を占めている。20msの入力に対する処理が平均60.6msなので、
入力より約3倍遅く、待ち行列が増え続けて上限に到達した。
初回出力のソフトウェア推定は約76msだったが、継続中の中央値は約1.1秒、最大は約2.04秒に増加している。
GPUでは小さな演算を多数起動する負担が考えられるが、GPUプロファイラを実行していないので、
カーネル起動・GPUクロック・他のGPU負荷等の内訳は確定できない。

## 変更

- CUDA選択時は、固定13msの推論を **PyTorch CUDA Graph / FP32** としてcapture/replayする。
  細かい演算をPythonから一つずつ起動する負担を減らすことが目的。
- 学習済み重み・計算構造・内部13ms・入力20ms・再生クッション15ms・停止上限を維持する。
  新たなpip依存やONNX Runtime GPUは使わず、CUDA版PyTorchの機能を使う。
- マイク開始前に、ロードした重みで通常のCUDA推論と6回の逐次比較を行い、出力と全状態を検証する。
  許容誤差 `rtol=1e-3 / atol=1e-4`。検証失敗時に未検証の経路でマイクを開始しない。
- graphの入力/出力アドレスを固定し、入力・状態をGPU上でコピーしてreplayする。
  decoder/prenetの入力状態をcloneしてfunctional stepにし、前フレームの出力状態を次の入力へ渡す。
  graph出力のメモリは再利用するので、runnerはセッションごとに作り、セッション間で共有しない。
- ウォームアップ後に24パケット（480ms相当、端数のflushを含む）を処理して容量を見積もる。
  実時間/入力時間のRTFが1以上なら、マイクを開く前に開始を拒否し、CPU経路を案内する。
  測定後は音声状態・統計をresetし、診断には`settings.preflight`として残す。
- LLVCの「自動」はCPU ONNXを選ぶ。RVCの「自動」は従来どおりGPUがあれば使う。
  明示的なCUDA選択を、黙ってCPUへ切り替えることはしない。

## 試し方

開発サーバーをCtrl+Cで停止し、同じ `research/streamvc-stateful-probe` ブランチで:

```bash
git pull --ff-only
npm run dev
```

まずLLVC / CPUまたは自動を基準にする。開始後に **LLVC: ONNX Runtime CPU / FP32** と表示される。
GPUを試す場合はCUDAを明示的に選ぶ。成功時の実行方式は **LLVC: PyTorch CUDA Graph / FP32**。
開始前RTFも画面に表示し、debug JSONには`settings.runtime`、`settings.preflight`、数値比較の最大出力誤差を保存する。
初回のcapture・検証・事前計測は、マイクを開く前に行う。

capture非対応・メモリ不足・数値不一致の場合は、準備エラーとCPUへの切り替え案内を表示する。
事前RTFが良好でも、録音・ネットワーク・再生や実行中の負荷を含む100ms以下を保証するものではない。
事前計測はwarmup有効時に実施する。通常のWeb UIでは有効で、明示的にAPIで無効化した呼び出しは対象外。

## 検証と限界

このクラウドはCPU版PyTorchで、CUDAデバイスがない。
**CUDA Graphの実capture/replay・ユーザーGPUでの速度改善・学習済みモデルの音質は未検証。**
GPU未検証の状態で、GPU変換が直った・100msを達成したとは扱わない。

Python全155件: 成功151・スキップ4・失敗0。スキップはRVC実モデル・LLVC公式重み各1件と、CUDA Graph実機2件。
CPU上で、自動選択、事前計測の成功/拒否、状態・統計の初期化、CPUモデルを誤ってgraphへ渡すケースを確認した。
JavaScript14件成功。Chromium16件成功（CPU ONNX経路および既存RVC試験）。

実機でのGPU試験は次で実行できる。テストの重みは未学習で、演算・状態の検証である。

```bash
python -m unittest discover -s tests -p 'test_llvc_cuda.py' -v
```

CUDAがないと2件を明示的にスキップする。CUDAがある場合は、逐次出力・flush/reset・セッション分離を実GPUで比較する。
実際の変換はWeb UIで同じモデル・同じ入力を使ってCPU/GPUを比較し、RTF・入力/再生待ち・途切れと音質を確認する。
