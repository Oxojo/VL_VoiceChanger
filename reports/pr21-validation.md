# PR #21 / Issue #6 検証結果

以下は初回 PR 検証時点（Web UI 実装・npm 起動入口追加・push 対応前）の記録。

対象: `fix/streaming-boundary-validation`、PR HEAD `c30f4c31c52cb330cb073b3acc57e224566e99d7`。
作業開始時は空の checkout だったため origin から取得し、PR HEAD と対象ブランチの一致を確認した。
変更は対象ブランチのローカル作業ツリーのみ。main と feature/streaming-foundation の変更、push、PR マージは実施していない。

## 実行環境と再現手順

Python 3.12.14、Linux、CPU。仮想環境: `/workspace/venvs/vlvc`。
PyTorch 2.15.0+cpu、NumPy 2.3.5、SciPy 1.17.0、ONNX Runtime 1.31.1、SoundFile 0.14.0、FastAPI 0.141.1。
`pyproject.toml` の依存関係と engine/web extras を editable install した。依存定義・lockfile は変更していない。

```bash
cd /workspace/VL_VoiceChanger
source /workspace/venvs/vlvc/bin/activate
python -m unittest discover -s tests -p 'test_streaming.py' -v
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v
```

全体テストでは CPU スレッド数のみ制限し、テスト選択・assertion は変更していない。

| 実行 | 総件数 | 成功 | 失敗/エラー | スキップ |
| --- | ---: | ---: | ---: | ---: |
| PR HEAD の streaming | 14 | 14 | 0 | 0 |
| PR HEAD の全体 | 72 | 71 | 0 | 1 |
| 回帰テスト追加後・修正前の streaming | 17 | 15 | 2 | 0 |
| 修正後の streaming | 17 | 17 | 0 | 0 |
| 修正後の全体 | 75 | 74 | 0 | 1 |

スキップは `test_model_path_accepts_listed_model`（models/ に .pth がないため）。
ログは Git 管理外の `logs/pr21/` に保存。修正後 streaming 0.046 秒、全体 0.760 秒。
ONNX Runtime の telemetry ID 保存警告と、既存の定数波形相関テストの NumPy RuntimeWarning が出るがテストは成功している。
`git diff --check` と `python -m engine stream-bench --help` も成功。

## コードレビュー

- **入力分割**: 16 kHz mono を float32 に変換し、160 サンプル単位のフレームを保持。1/7/160/320 サンプルや不均等分割でも同一出力。2 次元入力と flush 後の push を拒否する。
- **フレーム数**: 整数の絶対フレーム位置で管理し、出力は完成入力フレーム数 × output_sr/100。48 kHz の 480 サンプル単位も検証済み。最終 160 サンプル未満だけを捨てる。
- **クロスフェード**: フェードを hop-1 フレーム以内に制限。前チャンクの未確定 tail と次チャンクの先頭を一度だけ混合する。異なるチャンク波形で端点と offline assembly との一致を確認。既返却音声は変更しない。
- **flush/reset**: flush は残る完成フレームを処理し、再実行は空配列。短い区間は推論時だけ 100 ms にパディングし、出力を実入力フレーム数に戻す。reset は入力・端数・fade・位置・統計・closed を初期化する。実モデルにおける短区間の offline 実装との差は既存ドキュメントに記載されている。
- **異常なモデル出力長**: パディング後の推論入力長を期待値として、±1 出力フレームまで補正し、それを超える不足・超過を ValueError にする。追加テストは 16/48 kHz、1/30 入力フレーム、±1 フレームちょうどとその 1 サンプル外側を検証する。

## 発見・修正した問題

`push()` / `flush()` は返却配列を組み立てる前に複数チャンクの状態を進めていた。
後続チャンクが長さ検証で失敗すると、先行チャンクの音声も返却されない一方、出力位置と統計が進んでしまう。
例: 70 フレーム入力で第 3 回推論だけが不正な長さを返すと、先に成功した未返却の 20 フレームが消費される。

`_drain()` に処理をまとめ、例外時はその呼び出し開始時の出力位置・保持 context/fade・完了統計を復元するよう修正した。
入力は受信済みのまま保持されるため、backend を修正後に `push([])` / `flush()` で再試行できる。同じ入力を再送しない。
backend 自体の内部状態は復元しない。状態を持つ backend では reset が必要になり得る。

追加した 3 テスト:

1. 以前に返却済みの音声がある状態で、後続 push の途中の推論が失敗しても未返却音声・fade・統計を保持し、再試行で全音声を返せること。
2. flush の途中の推論が失敗しても入力・端数・統計・open 状態を保持し、再試行で全音声を返せること。
3. 出力長許容範囲をサンプル単位で検証すること（20 subtest 条件）。

1 と 2 は元実装で失敗し、修正後に成功した。再試行後の波形と元入力、出力長と統計も検証している。

## 未検証

実 RVC .pth、ContentVec/RMVPE ONNX、ベンチマーク用入力 WAV は checkout にない。
ONNX 環境変数も未設定で、資産 resolver は両 ONNX を解決できなかった。
NVIDIA device node と nvidia-smi はなく、torch.cuda.is_available() は False。
ONNX providers は AzureExecutionProvider / CPUExecutionProvider のみ。

このため実モデルの stream-bench は実行していない。CLI help の成功はモデル推論の検証を意味しない。
RVC/ONNX 互換性、CUDA provider、RTF、音質・境界ノイズ、マイク/再生を含む遅延は未検証。
これらは必要なモデル・入力音声・GPU がある環境で別途計測する必要がある。
