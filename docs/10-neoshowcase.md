# 10. NeoShowcase への公開

`showcase/` は、NeoShowcase でそのまま配信できるビルド不要の静的サイトです。
公開されるのはプロジェクト紹介ページであり、音声変換処理そのものではありません。

既存の変換 Web UI（`engine/web/`）は、次の理由からローカル利用を前提とします。

- RVC モデルは容量・利用規約の都合で Git 管理外
- PyTorch / ONNX Runtime と、実用速度を得るための GPU が必要
- 音声アップロードを受け付ける公開 API には、認証・保存期間・同時実行数の設計が必要

## NeoShowcase の設定

NeoShowcase の管理画面でリポジトリと `main` ブランチを登録し、次の値を設定します。

| 項目 | 値 |
| :--- | :--- |
| Deploy Type | `Static` |
| Build Type | `Command` |
| Base Image | 空欄 |
| Build Command | 空欄 |
| Artifact Path | `showcase` |
| Is SPA | `No` |

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

## 変換 Web UI を将来公開する場合

現在の `engine/web/` を NeoShowcase 上で動かすには、静的サイトではなく Runtime としての構成が必要です。
少なくとも次を決めてから別アプリとして公開します。

1. 再配布可能なモデルと特徴量抽出モデルの提供方法
2. CPU / GPU の実行環境と、許容する処理時間
3. 認証、アップロード上限、音声の削除タイミング
4. 1 ワーカー直列処理を前提にした同時利用者数
5. Web サーバーを `0.0.0.0` と NeoShowcase の Website 設定ポートで待ち受ける起動方法
