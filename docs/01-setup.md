# Phase 1: 導入と初回変換

企画書 6 章 Phase 1 に対応。**「変換後の声が確認できた」** 状態をゴールとする。

---

## 1. どの配布物を落とすか

**公式 README の「エディション対応表」と、実際に公開されている配布ファイルは別物。** 落とすファイル名で判断する。

2026-10 時点で `wok000/vcclient000` に公開されている Windows 用は次の 3 つ。

| 配布ファイル | 版 | サイズ目安 | RVC | 用途 |
| :--- | :--- | :--- | :--- | :--- |
| `vcclient_win_cuda_2.1.4-alpha.zip` | 2.1.4-alpha | 約 3.5 GB | **対応** | NVIDIA GPU がある場合の第一候補 |
| `vcclient_win_dml_2.1.4-alpha.zip` | 2.1.4-alpha | 約 2.8 GB | **対応** | NVIDIA 以外 / DirectML を使う場合 |
| `vcclient_std_win_2.2.2-beta_only_beatrice.zip` | 2.2.2-beta | 約 128 MB | **非対応** | Beatrice 専用。本企画では使わない |

> **注意**: 版番号が新しい 2.2.2-beta の Windows 版は **Beatrice 専用（RVC 非対応）**。番号の大きさで選ぶと RVC が使えない。
> 配布状況は変わるため、**導入時に https://huggingface.co/wok000/vcclient000 で必ず再確認する。**

選定の目安:

| 環境 | 選ぶファイル |
| :--- | :--- |
| NVIDIA GPU（VRAM 4 GB 以上） | `vcclient_win_cuda_*` |
| NVIDIA 以外 / CUDA が不安定 | `vcclient_win_dml_*` |
| GPU なし | `vcclient_win_dml_*` で CPU 動作、または本企画の対象外と判断 |

本リポジトリの検証機（RTX 3060 Laptop / VRAM 6 GB）では **`cuda` 版**を第一候補とする。

## 2. ダウンロードと配置

1. https://huggingface.co/wok000/vcclient000 から上表の zip を取得する（3 GB 前後あるため回線に注意）。
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

### 2.1 展開後のフォルダ構造（2.1.4-alpha）

zip の中は `dist\main\` に全ファイルが入っている。**`start_http.bat` がある階層が `vcClient.root`** になる。

```text
C:\Tools\vcclient_win_cuda\        ← vcClient.root はここ
├─ main.exe                       ← 実体（プロセス名は main）
├─ start_http.bat                 ← 中身は "main.exe start --https false"
├─ start_https.bat
├─ start_http_with_ngrok.bat
└─ _internal\                     ← 依存 DLL・モデル置き場
```

- **`dist\main` を残したまま `root` に指定しない。** 一段深いと `start_http.bat` が見つからず起動に失敗する。
- `start_http.bat` は `main.exe` を起動するだけの 1 行。**プロセス名は `main`**（旧版の `MMVC_Client*` ではない）。本リポジトリのスクリプトは `vcClient.root` 配下の実行ファイルから自動判定するため、名前を覚える必要はない。

## 3. 初回起動

```powershell
.\scripts\Start-VoiceChanger.ps1 -Profile character-a
```

スクリプトは次を行う。

1. プロファイルとモデルファイルの存在確認
2. 記録済みの入出力デバイスが現在も存在するか確認
3. `start_http.bat` の起動（**既にプロセスが動いている場合は二重起動しない**）
4. GUI ポートの応答待ち（既定 300 秒。ただし**モジュールのダウンロードが進んでいる間は延長**）
5. **GUI で設定する値のチェックリスト表示**

### 3.1 初回起動はとても長い（実測）

初回起動時、VC Client は RVC 用の必須モジュール **10 個・合計約 3.3 GB** を取得し、**それが終わるまで GUI を開かない**。

| 項目 | 実測（2026-10-28 / 下り約 3 Mbps 相当） |
| :--- | :--- |
| モジュール取得 | 10 個 / 約 3.3 GB |
| 所要時間 | **1 時間以上**（回線次第で 30 分〜数時間） |
| 進捗の見え方 | スクリプトが `必須モジュールのダウンロード: 3/10 件` のように表示 |
| 2 回目以降 | 取得済みのため数十秒で起動 |

> **初回は「失敗した」と判断して再起動を繰り返さないこと。** 途中まで取得したモジュールは再利用されるが、無駄に時間がかかる。
> 進捗は `C:\Tools\vcclient_win_cuda\vcclient.log` で確認できる（[03-troubleshooting.md](03-troubleshooting.md) に確認用コマンドあり）。

待ち時間を長めにしたい場合:

```powershell
.\scripts\Start-VoiceChanger.ps1 -Profile character-a -TimeoutSeconds 7200
```

> GUI が開かない場合は [03-troubleshooting.md](03-troubleshooting.md) の「GUI が開かない」を参照。

## 4. モデルを登録する

1. GUI の **モデル選択エリア** で「編集」を押す。
2. 「アップロード」で **ボイスチェンジャタイプに `RVC` を選ぶ**（ここを間違えると読み込めない）。
3. `.pth` を指定する。index ファイル（`added_*.index`）があれば併せて指定する。
4. 戻ってモデルをクリックし、キャラクター画像が表示されれば読み込み成功。
5. モデル選択エリアの **設定保存** を押す（次回起動時に設定が復元される。一部項目は除く）。

モデルファイルは `models/<character>/` に置く運用にすると、プロファイルのパスと一致して管理しやすい。**`models/` は Git 管理外**。

本リポジトリでは **つくよみちゃん公式RVCモデル** を `models/tsukuyomi/` に配置済み（ASCII 名にリネームしてある）。

| ファイル | 位置づけ |
| :--- | :--- |
| `tsukuyomi-normal1.pth` | **まずこれで検証する**（プロファイル `character-a` の既定値） |
| `tsukuyomi-normal2.pth` / `normal3.pth` | 声質の比較用 |
| `tsukuyomi-strong.pth` | 張りのある発話向け |
| `tsukuyomi-weak.pth` | 囁き・弱い発声向け |

> 5 種類を順に試して**演者の地声に最も近い 1 つ**を採用する。切り替えは GUI のモデル選択エリアで行い、決まったファイル名を `config/profiles/*.json` の `model.pth` に記録する。
> クレジットは「変換元の声」と「つくよみちゃん公式RVCモデル」の**両方**が必要（[05-models.md](05-models.md)）。

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
