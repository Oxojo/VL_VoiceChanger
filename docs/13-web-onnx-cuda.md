# Web UI の特徴抽出を CUDA 化する試験

## 今回試す施策

これまで Web UI の「デバイス → CUDA」は RVC 合成だけを GPU 化し、
RMVPE（ピッチ推定）と ContentVec（音声特徴抽出）は CPU で動かしていた。
今回はこの 2 つも **CUDA 優先の ONNX セッション**に切り替えられるようにする。

過去の記録では重複する特徴抽出が負荷要因の一つ。この部分の GPU 化で
区間ごとの推論時間と、推論に追いつかず増える送信待ちを短縮できる可能性がある。
速度改善率は GPU・モデル・ドライバー・区間長に依存するため、実測で確認する。
hop/context/crossfade と既定の入力蓄積 700 ms は変更しない。

## 導入

NVIDIA GPU と、ONNX Runtime GPU に対応する CUDA / cuDNN が必要。
PyTorch が CUDA を使えても、ONNX の対応ビルドや依存 DLL が揃っているとは限らない。
対応表: <https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements>。

通常の `npm run dev` の依存準備後、サーバーを停止し、**同じ Python 環境**で CPU 版を GPU 版へ入れ替える。

```bash
source .venv/bin/activate
# Windows PowerShell は .\.venv\Scripts\Activate.ps1
python -m pip uninstall -y onnxruntime
python -m pip install "onnxruntime-gpu>=1.20"
python -c "import torch; import onnxruntime as ort; print(ort.get_available_providers())"
npm run dev
```

CPU 版と GPU 版は同じ Python モジュールを提供するため、重ねてインストールしない。
CUDA / cuDNN の対応に合わせ、必要なら対応表にある版を指定する。
一覧への CUDA の登録は DLL 読み込み・モデル実行の成功までは保証しない。
`npm run setup` などの依存更新では既定依存の CPU 版が入り直す場合がある。
その場合は CPU 版と GPU 版を両方削除して GPU 版を再インストールする。
既存の CUDA 版 PyTorch は開発ランチャーで上書きしない。

CPU に戻すには、サーバーを止めて GPU 版を削除し CPU 版を導入する。

```bash
python -m pip uninstall -y onnxruntime-gpu
python -m pip install "onnxruntime>=1.20"
```

## 画面から比較

1. モデル、合成デバイス、hop/context/crossfade、出力レートを固定し、
   「特徴抽出の高速化 → CPU」で開始する。
2. 十数秒以上試し、「区間あたり平均」のピッチ推定・音声特徴・合成・後処理、
   推論 RTF、最大推論、送信待ちを記録する。
   マイク開始前の事前準備は別表示され、これらの統計には含まれない。
3. 停止して「GPU / CUDA（試験）」に切り替え、同じ条件で再実行する。
4. ContentVec / RMVPE が両方 **CUDA 優先**と表示されることを確認し、処理時間を比較する。

CPU/GPU のキャッシュは分離され、ロード済み CPU 変換器を GPU 指定で再利用しない。
CUDA 未導入や、読み込み時に CPU にフォールバックしていた場合は開始をエラーで止める。
一部ノードが CPU に配置されることはあり、「CUDA 優先」は全演算の GPU 実行を保証しない。
段階別時間にはデータ転送を含む。前処理などの一部は 4 段階に含まれないため、和は全推論時間と一致しない場合がある。
ファイル変換にも適用でき、multipart / WebSocket の両方で `ort_cuda: true` を受け付ける。

厳密な比較には同じ WAV を用意し、以下を複数回実行する。差は `--ort-cuda` の有無だけにする。
CLI は Web UI と違って事前推論を自動実行しないので、初回コストを含む点にも注意する。

```bash
python -m engine stream-bench -m models/tsukuyomi/tsukuyomi-normal1.pth \
  -i logs/verify/tts-input-ja.wav --device cuda --hop 0.5 --context 0.15 --crossfade 0.05
python -m engine stream-bench -m models/tsukuyomi/tsukuyomi-normal1.pth \
  -i logs/verify/tts-input-ja.wav --device cuda --hop 0.5 --context 0.15 --crossfade 0.05 --ort-cuda
```

## 検証範囲

Python 全 94 件: 成功 93 / スキップ 1 / 失敗 0。
Chromium 結合テスト 8 件: 成功 8 / 失敗 0（テスト専用の模擬変換器）。
キャッシュ分離、両セッションの CUDA 優先検証、CPU フォールバック拒否、
設定伝達・プロバイダー・段階別平均時間の表示、異常出力時の統計復元を確認した。

この環境には実モデル・ONNX 資産・GPU がないため、実 GPU 推論、速度短縮量、
GPU メモリ消費、変換音声の品質は未検証。
模擬変換器の処理時間を実測の改善値として扱わない。
