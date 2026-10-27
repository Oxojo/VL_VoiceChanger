# Phase 1: 導入と初回変換

企画書 6 章 Phase 1 に対応。**「変換後の声が確認できた」** 状態をゴールとする。

---

## 1. エディションを選ぶ

VC Client は v2.2.1 以降、エディションごとに対応モデルが異なる。**RVC を使うなら `cuda` か `onnx` を選ぶ。**

| エディション | プラットフォーム | RVC 対応 | 選定基準 |
| :--- | :--- | :--- | :--- |
| `cuda` | Windows | **対応** | NVIDIA GPU がある場合の第一候補 |
| `onnx` | Windows / Mac(m1) | **対応** | NVIDIA が無い / DirectML を使う場合 |
| `std` | Windows / Mac / Linux | **非対応** | 本企画では使わない |

> 対応表は公式 README の記載に基づく。**導入時点の最新版で再確認する。**
> https://github.com/w-okada/voice-changer

## 2. ダウンロードと配置

1. https://huggingface.co/wok000/vcclient000 から目的のエディションの zip を取得する。
2. **英数字のみの浅いパス**へ解凍する。
   - 良い例: `C:\Tools\vcclient_win_cuda`
   - 避ける: `C:\Users\山田\OneDrive\...\ボイスチェンジャー\`（Unicode・同期フォルダ）
3. 旧バージョンが別フォルダにある場合は混在させない（1 フォルダ 1 バージョン）。
4. `config/settings.json` の `vcClient.root` を更新する（他のキーは残したまま）。

```json
{
  "vcClient": {
    "root": "C:/Tools/vcclient_win_cuda",
    "launcher": "start_http.bat",
    "port": 18888
  }
}
```

（`vcClient` 部分のみ抜粋。`paths` / `stabilityTest` は変更しない。）

パスはスラッシュ区切りでもバックスラッシュでもよい。`\` を使う場合は JSON 上で `\\` と二重に書く。

## 3. 初回起動

```powershell
.\scripts\Start-VoiceChanger.ps1 -Profile character-a
```

スクリプトは次を行う。

1. プロファイルとモデルファイルの存在確認
2. 記録済みの入出力デバイスが現在も存在するか確認
3. `start_http.bat` の起動
4. GUI ポートの応答待ち（最大 180 秒）
5. **GUI で設定する値のチェックリスト表示**

初回起動時は必要なデータをインターネットからダウンロードするため 1〜2 分かかる。

> GUI が開かない場合は [03-troubleshooting.md](03-troubleshooting.md) の「GUI が開かない」を参照。

## 4. モデルを登録する

1. GUI の **モデル選択エリア** で「編集」を押す。
2. 「アップロード」で **ボイスチェンジャタイプに `RVC` を選ぶ**（ここを間違えると読み込めない）。
3. `.pth` を指定する。index ファイル（`added_*.index`）があれば併せて指定する。
4. 戻ってモデルをクリックし、キャラクター画像が表示されれば読み込み成功。
5. モデル選択エリアの **設定保存** を押す（次回起動時に設定が復元される。一部項目は除く）。

モデルファイルは `models/character-a/` に置く運用にすると、プロファイルのパスと一致して管理しやすい。**`models/` は Git 管理外**。

## 5. 音声入出力を設定する

詳細設定エリアの **AUDIO** で方式を選ぶ。

| 方式 | 動作 | 利点 | 注意 |
| :--- | :--- | :--- | :--- |
| `client` | ブラウザ（Chrome）がマイク・スピーカーを扱う | ノイズ抑制が使える、設定が簡単 | ブラウザ経由のため遅延が増えやすい |
| `server` | VC Client がデバイスを直接扱う | **遅延を抑えやすい**（本番向き） | デバイスを占有する。他アプリと同時に使えない |

初期検証は `server` を第一候補とし、音切れが解消しない場合に `client` も試す。**採用した方式はプロファイルに記録する。**

続いて入力・出力を指定する。

- **input**: マイク（USB マイク、またはオーディオ IF の入力）
- **output**: 確認用ヘッドホン
- **monitor**: `server` モードのみ有効。演者に返す場合に使うが、**遅延した自分の声は話しづらくなる**ため初期検証では原則使わない（企画書 3 章）。

## 6. 初回変換の確認

1. **start** を押す。
2. マイクに向かって短く発話する。
3. 変換後の声が output のヘッドホンから聞こえることを確認する。
4. 変換されない場合は **pass through** を押して、生声が出るか切り分ける。
   - 生声も出ない → 入力または出力デバイスの指定ミス
   - 生声は出るが変換されない → モデル未ロード、または `S. Thresh`（ノイズゲート）が高すぎる

## 7. 記録する

`config/profiles/character-a.json` を実際の値に更新する。

| 項目 | 記録先 |
| :--- | :--- |
| モデルファイル・index | `model.pth` / `model.index` |
| AUDIO 方式 | `audio.mode` |
| 入力・出力・モニター | `audio.inputDevice` など |
| 変換パラメータ | `convert.*` |
| buf / res の実測表示 | `measured.bufMs` / `measured.resMs` |

## 8. Phase 1 完了条件

- [ ] `Start-VoiceChanger.ps1` で起動できた
- [ ] モデルが RVC として読み込まれた
- [ ] 変換後の声が output デバイスから聞こえた
- [ ] 採用エディション・バージョンを `docs/00-phase0-environment.md` に記録した
- [ ] プロファイル JSON を実測値で更新した
- [ ] （任意）`reports/phase1-first-conversion.md` にスクリーンショットや所感を残した

次は [06-tuning.md](06-tuning.md) で音質を詰める（Phase 2）。
