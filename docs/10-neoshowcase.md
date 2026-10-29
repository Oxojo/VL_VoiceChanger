# 10. NeoShowcase への公開（紹介ページ）

`showcase/` は、NeoShowcase でそのまま配信できるビルド不要の静的サイトです。
公開されるのはプロジェクト紹介ページであり、音声変換処理そのものではありません。

## 公開方式を選ぶ

- **紹介ページを公開する**: この手順の Static / Command を使います。
- **ブラウザから音声を変換する**: [11-neoshowcase-runtime.md](11-neoshowcase-runtime.md) の
  Runtime / Dockerfile を使います。別アプリとして登録してください。

変換 Web UI の通常利用はローカルを基本とします。サーバー公開では、次の準備が必要です。

- RVC モデルは容量・利用規約の都合で Git 管理外なので別途配置する
- PyTorch / ONNX Runtime の実行環境と、CPU / GPU ごとの速度を確認する
- 音声アップロードを受け付ける公開 API の認証・保存期間・同時利用を設計する

## NeoShowcase の設定

NeoShowcase の管理画面でリポジトリと `main` ブランチを登録し、次の値を設定します。
統合内容を main へマージする前に確認する場合は `feature/self-engine` を指定します。

| 項目 | 値 |
| :--- | :--- |
| Deploy Type | `Static` |
| Build Type | `Command` |
| Context | リポジトリのルート（指定が必要なら `.`） |
| Base Image | 空欄 |
| Build Command | 空欄 |
| Artifact Path | `showcase` |
| Is SPA | `No` |

**Buildpack は選びません。** リポジトリの Python 用 `pyproject.toml` が自動検出され、
紹介ページに不要な Python 依存のインストールで失敗する場合があります。
ルートの Dockerfile も紹介ページ用ではなく、変換サーバー専用です。

`showcase/index.html` がエントリーポイントです。CSS、JavaScript、画像はすべて
`showcase/` の中にあるため、ビルドコマンドや外部 CDN は不要です。

Website には利用する FQDN を設定します。公開後は次を確認します。

1. トップ、プロジェクト、仕組み、現在地、試すの各リンクへ移動できる
2. 実画面のスクリーンショットが表示される
3. GitHub とドキュメントへのリンクが開く
4. スマートフォン幅でメニューが開閉できる

## ローカル確認

```bash
python -m http.server 8080 --directory showcase
```

ブラウザで `http://127.0.0.1:8080/` を開きます。
紹介ページは音声を収集せず、コピーできるコマンドはローカル Web UI を起動するためのものです。
