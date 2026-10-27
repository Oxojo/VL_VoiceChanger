# 障害時の復旧手順とトラブルシュート

企画書 7 章の基本方針に沿う。

> 変換が止まった場合は無理に動作を継続せず、
> **ソフトウェア停止 → 入出力デバイス確認 → モデル再読み込み → 変換再開** の順で復旧する。

**原則: 一度に 1 つだけ変える。** 複数箇所を同時に変えると、何が効いたか分からなくなり記録が資産にならない。

---

## 1. 標準復旧フロー

```text
[1] 変換が止まった
      │
      ├─ stop を押す（無理に動かし続けない）
      │
[2] デバイスを確認
      ├─ マイク / オーディオ IF / ヘッドホンの接続
      ├─ Windows の既定デバイス
      └─ GUI の input / output の指定
      │
[3] モデルを再読み込み
      ├─ モデルをクリックし直す
      ├─ 必要なら GUI を再読み込み（F5）
      └─ それでも駄目なら VC Client を再起動
      │
[4] 変換を再開して発話確認
      │
[5] 復旧した条件を記録（下の記録欄）
```

再起動しても直らない場合は、**モデルの読み込み失敗**か **GPU のメモリ不足** を疑う。

## 2. 症状別の切り分け

### 音がまったく出ない

| 確認順 | 見るところ | 対処 |
| :--- | :--- | :--- |
| 1 | **start** が押されているか | 押す |
| 2 | output デバイスの指定 | ヘッドホンを選び直す |
| 3 | ヘッドホンの音量・接続 | 物理確認 |
| 4 | Windows の出力先 | VC Client の output と同じか確認 |
| 5 | **pass through** | 生声が出るなら変換側、出ないなら入出力側の問題 |

### 生声は出るが変換されない

| 原因 | 確認 | 対処 |
| :--- | :--- | :--- |
| モデル未ロード | キャラクター画像が出ているか | モデルをクリックし直す |
| `S. Thresh` が高すぎる | 小さい声で反応しない | 値を下げる（例: -60 → -50） |
| 入力レベル不足 | メーターがほとんど振れない | `GAIN in` を上げる / マイクゲインを上げる |
| モデル形式の取り違え | RVC 以外で読み込んでいないか | RVC として登録し直す |

### 音が途切れる・ブツブツする

**最も多い症状。** 1 項目ずつ試す。

| 順 | 変更 | 効果 | 副作用 |
| :-- | :--- | :--- | :--- |
| 1 | `CHUNK` を上げる（128 → 256 → 512 → 1024） | 音切れが減りやすい | 遅延（buf）が増える |
| 2 | `F0 Det` を `rmvpe`/`crepe` → `dio` に変える | CPU/GPU 負荷が下がる | ピッチ精度が落ちる |
| 3 | `EXTRA` を下げる（8192 → 4096 → 2048） | 変換時間（res）が短くなる | 声質が粗くなる |
| 4 | AUDIO を `server` に変える | 遅延が減る | デバイスを占有する |
| 5 | 他のアプリを終了する | 負荷が下がる | — |
| 6 | `NOISE` の Sup1/Sup2 を切る | 負荷が下がる | ノイズは増える |

**判定基準は `buf > res`。** buf が res より短いと、変換が間に合わず音が切れる。

### 声が高すぎる／低すぎる・不自然

| 原因 | 対処 |
| :--- | :--- |
| `TUNE` の値が合っていない | 演者とモデルに合わせて試聴で決める。**+9〜+14 固定にしない** |
| 演者の声域とモデルが合っていない | 別モデルを検討（Phase 4） |
| `INDEX` が高すぎる | 0.3〜0.5 程度から試す |

企画書 4.2 のとおり、ピッチは数値を変えれば必ず良くなるものではない。**必ず耳で決める。**

### 声が割れる・音量が大きすぎる

1. マイクのゲインを下げる（Windows の「マイクブースト」も確認）
2. `GAIN in` を下げる
3. `vol` / `GAIN out` を下げる
4. オーディオ IF の入力レベルつまみを下げる

大声を出したときに波形が振り切れていないかを確認する。

### 変換音と生声が二重に聞こえる

| 原因 | 対処 |
| :--- | :--- |
| オーディオ IF のダイレクトモニター | つまみを 0 にする / 機能を切る |
| Windows の「Listen」設定 | 「このデバイスを聴く」をオフ |
| `monitor` に output と同じデバイスを指定 | monitor を空にする |
| client モードでブラウザの音が二重 | AUDIO を `server` に統一する |

### 途中で落ちる・フリーズする

| 疑うもの | 確認 | 対処 |
| :--- | :--- | :--- |
| VRAM 不足 | `nvidia-smi` でメモリ使用量を確認 | `EXTRA` を下げる、他 GPU アプリを閉じる |
| 電源設定 | スリープ／GPU 省電力 | 高パフォーマンスに変更 |
| ウイルス対策 | 起動直後のスキャン | VC Client フォルダを除外 |
| 熱 | GPU 温度 | 冷却・負荷軽減 |

再発時は `Test-Stability.ps1` を回し、落ちた時刻を特定できるようにする。

### GUI が開かない／画面が白い

> **最初に疑うのは「初回起動のモジュールダウンロード中」。**
> 2.1.4-alpha は初回起動時に RVC 用の必須モジュール **10 個・合計約 3.3 GB** を
> `C:\Tools\vcclient_win_cuda\modules\` へダウンロードし、**すべて終わるまでポート 18888 を開かない**。
> 回線速度によっては **30〜90 分**かかる。

| 症状 | 対処 |
| :--- | :--- |
| **ポートが開かない（初回）** | 仕様。`vcclient.log` に `Start downloading:` / `Downloading completed:` が出ていれば正常に進行中。**中断・再起動しない**（最初からやり直しになる） |
| ポートが開かない（2 回目以降） | コンソールのエラーを確認。`VC Client` を再起動 |
| `ERR_CONNECTION_REFUSED` | 起動処理中。1〜2 分待つ。ウイルス対策の除外を追加 |
| 白い画面のまま | ブラウザで `http://localhost:18888/` を直接開く。F5 |
| ポート 18888 が使用中 | 旧プロセスが残っている。タスクマネージャで `main`（2.1.4-alpha の実行ファイル名）を終了 |
| 起動直後に閉じる | パスに Unicode が含まれていないか確認（ONNX Runtime が失敗する） |
| ダウンロードが途中で止まる | `vcclient.log` の最終行の時刻を確認。30 分以上進まなければ VC Client を終了して再起動（途中まで取得済みのモジュールは再利用される） |

```powershell
# 進捗の確認（初回起動時）
Get-Content 'C:\Tools\vcclient_win_cuda\vcclient.log' -Tail 5
(Select-String 'C:\Tools\vcclient_win_cuda\vcclient.log' -Pattern 'Start downloading:').Count
(Select-String 'C:\Tools\vcclient_win_cuda\vcclient.log' -Pattern 'Downloading completed:').Count
"取得済み: {0:N0} MB" -f ((Get-ChildItem 'C:\Tools\vcclient_win_cuda\modules' -Recurse -File |
    Measure-Object Length -Sum).Sum / 1MB)

# 使用中のポートを確認する
Get-NetTCPConnection -LocalPort 18888 -ErrorAction SilentlyContinue |
    Select-Object LocalAddress, LocalPort, State, OwningProcess
```

初回に取得されるモジュール（実測 2026-10-28）:

| モジュール | サイズ |
| :--- | ---: |
| `contentvec/hubert_base.pt` | 181 MB |
| `contentvec/contentvec-f.onnx` | 361 MB |
| `rinna_hubert/rinna_hubert_base-f.onnx` | 361 MB |
| `onnxcrepe/tiny.onnx` | 2 MB |
| `onnxcrepe/full.onnx` | 85 MB |
| `rmvpe/rmvpe_20231006.pt` | 173 MB |
| `rmvpe/rmvpe_20231006.onnx` | 345 MB |
| `applio/japanese_hubert_base.pt` | 361 MB |
| `applio/chinese_hubert_base.pt` | 1,084 MB |
| `applio/korean_hubert_base.pt` | 361 MB |
| **合計** | **約 3.3 GB** |

> 中国語・韓国語のモデルも**使わなくても取得される**。ディスクは VC Client 本体（約 8 GB）に加えて 3.3 GB 必要。

### モデルが読み込めない

- ボイスチェンジャタイプを **RVC** で登録したか
- `.pth` が壊れていないか（サイズが 0 でないか）
- index は任意。**index 無しでも動く**ので、まず index を外して切り分ける
- エディションが `std` になっていないか（**std は RVC 非対応**）

## 3. 復旧記録

復旧したら必ず残す。同じ症状の再発時に効く。

```markdown
### YYYY-MM-DD HH:MM 発生

- 症状:
- 直前の操作:
- 環境: エディション / バージョン / AUDIO 方式
- 設定値: CHUNK= / EXTRA= / F0= / TUNE= / INDEX=
- 試したこと（順番に）:
  1.
  2.
- 復旧した操作:
- 再発防止メモ:
```

## 4. 会場での代替手段

変換が復旧しない場合の代替 MC・BGM 等は **本企画の対象外**。ライブ運営計画側で定める（企画書 7 章）。
