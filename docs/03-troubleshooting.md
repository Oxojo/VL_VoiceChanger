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

再発時は `python -m vlvc stability --profile character-a` を回し、落ちた時刻を特定できるようにする。

### GUI が開かない／画面が白い

> **最初に疑うのは「初回起動のモジュールダウンロードが止まっている」。**
> 2.1.4-alpha は初回起動時に RVC 用の必須モジュール **10 個・合計約 3.3 GB** を
> `C:\Tools\vcclient_win_cuda\modules\` へダウンロードする。
> **GUI（ポート 18000）はダウンロードが全部終わる前に開くことがある**ため、
> 「画面は出るのに RVC が動かない」場合はモジュールの状態を確認する。

| 症状 | 対処 |
| :--- | :--- |
| **ポートが開かない（初回）** | 仕様。`vcclient.log` に `Start downloading:` / `Downloading completed:` が出ていれば正常に進行中。**中断・再起動しない** |
| ポートが開かない（2 回目以降） | コンソールのエラーを確認。`VC Client` を再起動 |
| `ERR_CONNECTION_REFUSED` | 起動処理中。1〜2 分待つ。ウイルス対策の除外を追加 |
| 白い画面のまま | ブラウザで `http://localhost:18000/` を直接開く。F5 |
| **画面は出るが RVC が動かない** | モジュールが未完了。下のコマンドで `valid=False` を確認し、VC Client を再起動して取得を完了させる |
| ポート 18000 が使用中 | 旧プロセスが残っている。タスクマネージャで `main`（2.1.4-alpha の実行ファイル名）を終了 |
| 起動直後に閉じる | パスに Unicode が含まれていないか確認（ONNX Runtime が失敗する） |
| ダウンロードが途中で止まる | `vcclient.log` の最終行の時刻を確認。30 分以上進まなければ VC Client を終了して再起動（**部分ファイルは `curl -C -` で再開できる**。下記参照） |

```bash
# モジュール状態（起動中ならこれが一番確実）
curl -s http://localhost:18000/api/module-manager/modules | python -m json.tool

# ログの末尾（パスは環境に合わせる）
python -c "from pathlib import Path; print(Path('C:/Tools/vcclient_win_cuda/vcclient.log').read_text(encoding='utf-8', errors='replace')[-3000:])"

# ポートが開いているか
python -c "import socket; s=socket.socket(); print('open' if s.connect_ex(('127.0.0.1',18000))==0 else 'closed')"
```

`python -m vlvc doctor` でも、起動中なら `/api/module-manager/modules` の有効件数を表示する。
（Windows の PowerShell では `curl` が別名のため `curl.exe` を使う）

**ダウンロードが止まったときの復旧**（2026-10-28 に実際に発生）:

1. VC Client を終了する（`main` と `voice-changer-native-client`）
2. 部分ファイルは残っているので、**同じ URL へ `curl -C -` で再開**する
3. 取得後に **SHA-256 を照合**する（ハッシュは `vcclient.log` に記録されている）
4. 一致したら VC Client を起動する（モジュールは検証済みとして扱われる）

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

`python -m vlvc register-model`（REST API）で失敗する場合は次を確認する。**エラー本文は `vcclient.log` にしか出ない**（API は 400 を返すだけ）。

| 症状・ログ | 原因 | 対処 |
| :--- | :--- | :--- |
| `Failed to estimate RVC model` | 一時ファイル名に拡張子が無い（`upload_dir\model_file` など） | **拡張子を保ったファイル名**で `upload_dir` に置く |
| `No such file or directory: 'upload_dir\\upload_dir\\...'` | `model_file` に `upload_dir\` まで含めた | **ファイル名だけ**を渡す（サーバーが前置する） |
| `POST /api/slot-manager/slots` が 400 | 上記のいずれか | `vcclient.log` の末尾で `Failed to set_new_slot` を確認（上のログ確認コマンド） |
| `POST .../slots/dummy/rvc` が 500 | **この URL はドキュメント用のダミー** | `POST /api/slot-manager/slots` を使う（`vlvc register-model` が行う） |

> **スロット番号は 0 始まり。** `slot_index=5` は GUI では「6.」と表示される。`model_dir\5\` が実体。

### 変換が始まらない（`bulk conversion process is not started`）

- **症状**: GUI も API も応答するが、音声が変換されない。`vcclient.log` に `bulk conversion process is not started` が繰り返し出る
- **原因**: 2.1.4-alpha では変換スレッドの開始が **GUI（Socket.IO）側の操作**にある。
  REST の `POST /api/local-voice-changer-interface/operation/start` は音声 I/O スレッドだけを起動し、変換スレッドは起動しない
- **対処**: `http://localhost:18000/` の GUI で **start** を押す。押すとログに `start_convert_chunk_bulk called` が出る

```bash
python -c "from pathlib import Path; t=Path('C:/Tools/vcclient_win_cuda/vcclient.log').read_text(encoding='utf-8', errors='replace'); print('\n'.join(l for l in t.splitlines() if 'start_convert_chunk_bulk' in l or 'bulk conversion process is not started' in l)[-2000:])"
```

### 録音した WAV が無音（`input.wav` が -90 dB 前後）

server モードの録音で `input.wav` が無音なら、**マイク入力が届いていない**。

| 確認 | コマンド / 見方 |
| :--- | :--- |
| デバイスの接続状態 | `python -m vlvc doctor` の「音声入出力デバイス」に、使用するマイクが入力として出るか |
| Windows のマイク許可 | 設定 > プライバシー > マイク（`ConsentStore\microphone` が `Allow`） |
| 入力レベル | GUI の入力メーターが発話で振れるか |
| 録音の実値 | `input.wav` の RMS が **-40 dB 以上**あるか |

実測（2026-10-28）では USB オーディオ IF `マイク (3- AT-UMX3)` が **未接続（PnP Status=Unknown）** で、
内蔵 `マイク (C-Media(R) Audio)` も実入力が 0 だった（`input.wav` = **-90.7 dB**）。
**デバイスを接続し直し、`input.wav` が -40 dB 以上になることを確認してから**変換の評価に入る。

### オフライン変換 API が 500 を返す（既知の不具合）

`POST /api/voice-changer/convert_file` は 2.1.4-alpha では**常に 500**になる。

```text
File "vcclient\voice_changer\voice_change_manager\voice_changer.py", line 733, in convert_file
TypeError: Timer.__init__() got an unexpected keyword argument 'enable'
```

`realtime_process_status_enabled` / `realtime_output_status_enabled` を `false` にしても回避できない（実測）。
**代替**: server モードの**録音**で `input.wav` / `output.wav` を取る（[02-operation.md](02-operation.md) §4.1）。


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
