# 10. NeoShowcase への配置

## 1. 今回のビルド失敗の原因

ログでは Buildpack が `requires-python = ">=3.10"` から Python 3.14.7 を選択した。
SciPy 1.15.3 はこの環境でソースビルドに進み、Fortran コンパイラーがなく失敗した。
Fortran の追加ではなく、Dockerfile で Python 3.12 とバイナリ依存を選ぶ。

また、このアプリは FastAPI サーバーが変換を行うため **Static では動かない**。
`engine/web/static` だけを公開しても `/api/*` が使えない。

## 2. NeoShowcase の設定

| 項目 | 設定 |
| :--- | :--- |
| 対象ブランチ | `feature/self-engine`（Dockerfile が入ったコミット以降） |
| Deploy Type | **Runtime** |
| Build Type | **Dockerfile** |
| Context | リポジトリのルート（指定が必要なら `.`） |
| Dockerfile のパス | `Dockerfile` |
| 公開先のコンテナポート | **8080 / HTTP** |
| 起動コマンド | Dockerfile の既定値を使う（上書き不要） |

Static 用の Artifact Path（`./showcase`）や SPA 設定は使わない。
保存後に再ビルドする。画面の項目名が異なる場合も、HTTP の転送先はコンテナの 8080 番とする。
HTTPS の公開 URL を使う（ブラウザのマイク利用に必要）。

## 3. モデルは別途配置する

Git と Docker イメージにはモデル・生音声を含めない。
画面の表示と入力ノイズ除去はモデルなしでも使えるが、声の変換には以下が必要。
NeoShowcase のストレージ機能を使えるか、管理者に確認して配置する。

| コンテナ内のパス | 内容 |
| :--- | :--- |
| `/app/models/<character>/*.pth` | 使用許諾を確認した RVC v2 モデル |
| `/app/modules/contentvec/contentvec-f.onnx` | ContentVec |
| `/app/modules/rmvpe/rmvpe_20231006.onnx` | RMVPE |
| `/app/logs` | アップロード・変換結果の保存先（書き込み必須） |

ONNX を別の場所に置く場合は、環境変数 `VL_CONTENTVEC_ONNX` / `VL_RMVPE_ONNX` に
**コンテナ内のファイルパス**を設定する。ローカル PC の `C:\Tools\...` は使えない。
実行ユーザーは UID 10001。マウントするフォルダはこのユーザーがモデルを読め、ログを書ける権限にする。
一時領域のみの場合、再配置時に音声ファイルは失われる。履歴はメモリ保持なので再起動で消える。

## 4. 起動確認

- 公開 URL の `/` で画面が出る。
- `/api/health` が応答する（CPU 版なので `cuda: false` が正常）。
- `/api/models` に配置したモデルが出る。空なら配置先を確認する。
- 短い音声でノイズ除去、続いて変換を試す。

Docker が使える環境なら、事前に次で起動を確認できる。

```bash
docker build -t vlvc .
docker run --rm -p 8080:8080 vlvc
# http://localhost:8080/ （モデルなしで画面・ノイズ除去の確認）
```

## 5. 公開時の注意

- 現状は認証なしで、履歴や音声 API を利用者間で分離していない。
  **機密音声を扱わず、まずは NeoShowcase 側のアクセス制限を有効にした限定公開にする**。
  アプリ独自の認証・利用制限なしでの一般公開は推奨しない。
- アップロードされた音声はローカル PC ではなく **NeoShowcase のサーバーへ送信される**。
  公開前に利用者への案内・保存期間・削除運用を決める。
- CPU 推論はライブ用途の速度を保証しない。実測の速さは配置先の CPU・メモリ制限に依存する。
  CPU 版でも PyTorch・ONNX・モデルで相応のメモリとイメージ容量が必要。
- モデル・ContentVec・RMVPE の利用規約と公開サービスでの利用可否を確認する。
- `/api/health` はプロセスの生存確認であり、モデルの配置完了や変換成功を保証しない。
