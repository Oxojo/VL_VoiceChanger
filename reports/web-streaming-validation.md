# Web UI ストリーミング実装・検証

対象ブランチ: `fix/streaming-boundary-validation`。元 HEAD は PR #21 の `c30f4c31c52cb330cb073b3acc57e224566e99d7`。
前回の PR 検証で追加した未コミット変更を保持して実装した。
main / feature/streaming-foundation は変更していない。
ユーザーの依頼により npm 起動入口と合わせて対象ブランチへ commit・push する。マージはしない。

## 実装

- Web UI に「マイク変換を開始」「停止して残りを再生」を追加。
- AudioWorklet で mono 音声を取得し、状態を保持するリサンプラーで 16 kHz 化。
  20 ms の float32 パケットで `/api/stream` に送信する。
- WebSocket サーバーは既存のモデルキャッシュと単一ワーカーを使い、
  `StreamingChunkedConverter.push()` / `flush()` を実行する。イベントループで推論しない。
- 出力サンプリングレートを開始時に通知し、ブラウザは変換済み区間を連続する時刻に予約再生する。
  生のマイク音声は再生しない。
- 1 パケットずつ ack を待ち、未送信＋処理待ちを 2 秒以内、再生待ちを 3 秒以内に制限。
  処理に追いつかない場合はエラーを表示して停止する。1 セッションは 180 秒まで。
- 停止時は送信待ちを処理して flush、再度ボタンを押せば残りを破棄して切断。
  マイク拒否・切断・エラー・ページ離脱ではトラック、接続、再生ソース、AudioContext を解放する。
- マイク変換同士とファイル変換の同時実行を排他制御。不正入力・設定、異常なモデル出力、
  資産不足、受信タイムアウト、別 origin のブラウザ接続を扱う。
- websockets を Web UI 依存関係と起動前チェックへ追加し、CLI / Docker 起動時の受信サイズ・キュー上限も設定。
- ユーザー向け操作、WebSocket プロトコル、テスト方法を `docs/08-web-ui.md` に記載。

## 実行済み検証

Python 3.12.14 / PyTorch 2.15.0+cpu / ONNX Runtime 1.31.1 / Chromium 154.0.8037.57。

| コマンド | 成功 | 失敗 | スキップ |
| --- | ---: | ---: | ---: |
| `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m unittest discover -s tests -v` | 84 | 0 | 1 |
| `node --test tests/test_stream_audio.mjs` | 6 | 0 | 0 |
| `OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/browser_streaming_smoke.py -v` | 7 | 0 | 0 |

Python 全 85 件のうち 9 件が新規 WebSocket 結合テスト、1 件が開発サーバー起動・停止テスト。入力分割・48 kHz 出力・短い末尾・再接続・
セッション排他・異常設定/音声・資産不足・推論エラー・期間上限・タイムアウト・origin・
推論中も HTTP イベントループが応答することを確認。
スキップ 1 件は既存の `test_model_path_accepts_listed_model`（.pth がないため）。

JavaScript テストは 16/44.1/48 kHz 入力の分割不変性、ゲイン、パケット端数、
transfer によるバッファ detach、連続再生時刻、途切れ回数・再生キュー上限を確認。

ブラウザテストではテスト専用サーバーにのみ模擬変換器を注入。
Chromium の模擬マイクに既知の 440 Hz 波形を入力し、実際の AudioWorklet →
WebSocket → ストリーミング処理 → 非ゼロの 48 kHz 再生バッファという経路を確認した。
停止・再開を 2 回行い、切断後の再接続、マイク拒否からの復帰、許可待ち中の停止、
推論エラー表示、遅い推論時の停止、従来の録音ボタンとの共存も検証した。
本番サーバーに模擬変換器を選ぶ機能や、実モデル不足時のフォールバックは追加していない。

実ブラウザ検証で見つかった、transfer 後の配列長 0 によって送信が最初のパケットで止まる問題を修正し、
detach を再現する回帰テストを追加した。再生待ちが残っている間は、後続区間を既存区間の直後に接続して余計な隙間を挿入しない。

通常の `python -m engine serve --port 8765` も起動。
`/api/health`、`/api/models`、`/` と 3 つの新しい JS ファイルの HTTP 200、
実サーバーで存在しないモデルを指定した WebSocket の error 応答を確認した。
`git diff --check`、Python コンパイル、JavaScript 構文確認も成功。

ログ: `logs/web-streaming-tests.log`、`logs/web-streaming-browser.log`（Git 管理外）。

## 未検証・制約

実 RVC .pth / ContentVec・RMVPE ONNX / 利用可能な GPU はこの環境にない。
実モデルの声の変換、音質、CUDA 動作、RTF、音響出力の聴取、実マイクからスピーカーまでの遅延は未検証。
ブラウザテストは再生ソースの予約と波形を検証しており、物理スピーカーからの音を録音・聴取したものではない。

既定設定は入力蓄積だけで 700 ms 必要。非因果 RVC の重複推論であり、低遅延や実時間性能は保証しない。
初回ウォームアップが長い場合も送信キュー上限に達して停止し得る。
切断後も実行中の推論スレッド自体は強制終了せず、1 区間の処理が終わるまで次の接続を待たせる場合がある。
ONNX の特徴抽出は CPU のまま。ネイティブ音声デバイス I/O や特徴量キャッシュは追加していない。

## 手元で試す

```bash
python -m pip install -e ".[engine,web]"
python -m engine serve
```

RVC モデルと 2 つの ONNX を用意して画面を開き、モデルとデバイスを選び、
ノイズ除去・ハム除去を 0 にしてヘッドホンを使用する。
「マイク変換を開始」で許可し、話して試す。localhost または HTTPS が必要。
開発時は Node.js 18+ / Python 3.10+ を用意し、`npm run dev` だけでも起動できる。
初回の `.venv` 作成・CPU 版 PyTorch/Web 依存導入、開発サーバーからのブラウザ表示、
HTTP / WebSocket 応答、Ctrl+C での停止とポート解放も検証した。
