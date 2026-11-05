# 既存RVCの声をLLVCへ追加学習する（試験）

既存RVCの`.pth`はLLVCへ直接読み込めない。
そのRVCで変換した音声を教師にしてLLVCを微調整し、声ごとのLLVCモデルを書き出す。
Web UIでは「LLVCの声」から選んで、既存の13ms推論・CPU ONNX / CUDA Graph経路で使う。

**実装したのは学習・登録の仕組みで、対象RVCの声を再現できたという結果ではない。**
このクラウドには対象RVC・学習データ・GPUがなく、本番学習、声質/日本語の明瞭性、物理100ms以下は未検証。
RVCの声を同一品質でコピーできる保証はない。学習結果は試聴・比較して採用を判断する。

## 必要なもの

- 既存エンジンのRVCモードで正常に変換できるRVC v2 `.pth`。
- RVC用ContentVec/RMVPE ONNX。既存RVCと同じ`resolve_asset()`設定を使用する。
- 追加学習の初期値になるLLVC配布checkpoint（Web UIの「LLVCモデルを取得」で準備）。
- 変換元の普通の発話音声。WAV / FLAC / OGGのファイルをフォルダーに用意する。
  **RVCを作った際の元の学習データは不要**で、RVCに入力して教師音声を生成できる発話を使う。
- CUDA版PyTorchとGPU。`npm run dev`が新規作成した環境はCPU版の場合があるので、
  既にCUDA動作を確認したPython環境からコマンドを実行する。CPU指定は小規模動作確認向け。

入力音声は独立した録音を最低2ファイル、各2秒以上必要。
これは形式上の下限であって品質に十分な量ではない。最初は代表的な日本語・普段のマイク条件の発話を
10〜30分程度用意して試すことが考えられるが、必要量は話者・声の差・内容によって変わる。
明瞭な発話、語尾、子音、長短の発話を含め、評価用の録音も別に用意する。
学習中はマイク変換を止め、元モデルや音声ファイルを変更しない。

## 1. RVCで教師音声を作る

リポジトリ直下で、パスとピッチ/話者IDを自分のRVC設定へ置き換える。以下はPowerShellでもそのまま使える一行コマンド。

```bash
python -m engine.llvc.finetune prepare --teacher "models/my-rvc.pth" --audio-dir "training-audio" --output "logs/llvc-pairs/my-voice" --device cuda --pitch 0 --speaker 0
```

`training-audio`には**未変換の発話**を置く。教師の設定は`--pitch`、`--speaker`、`--rms-mix`（既定0.25）。
教師側のピッチ設定も学習する変換に含まれる。後からWeb UIのRVCピッチを変更してもLLVCには反映しない。

既定では4秒単位で読み込み、16kHz mono化、10ms単位へ入力をそろえた後にRVC変換する。
モデル出力長が一致しない場合はエラーとし、出力を強引に切り詰めてペアにしない。
両音声を同じ倍率でピーク調整し、音量関係を保ってFLOAT WAVで保存する。
2秒未満の末尾やほぼ無音の区間は除外する。

録音ファイル単位でtrain/valへ分割する。同じファイルのクリップは混ぜず、同一バイト内容の複製も除く。
別エンコードや分割された同一録音・同一話者の分離までは保証しないため、入力の整理は必要。
manifestには教師SHA256、設定、入力・音声ペアのSHA256を残す。
出力先は入力音声フォルダーと重ならない場所にする。

**まず`train/*_converted.wav`を聴き、教師が希望する声になっているか確認する。**
中断後は同じコマンドに`--resume`を付ける。モデル・データ・設定や保存済みペアが変わっている場合は再開を拒否する。

## 2. LLVCを微調整する

```bash
python -m engine.llvc.finetune train --dataset "logs/llvc-pairs/my-voice" --run-dir "logs/llvc-training/my-voice" --device cuda --steps 10000
```

10,000は試験用の初期値で、品質が完成する回数を意味しない。収束・過学習・音質を見て判断する。
学習時間や必要VRAMはこの環境で未計測。
既定batch=1、中央8192 samplesのloss、その前に8192 samplesのcontextを与える。
content側のprenet / input conv / encoderは凍結し、ラベル埋め込み・decoder・出力側などを更新する。
`--unfreeze-encoder`で全体更新も選べるが、学習条件が変わるため別runで比較する。

採用したlossはMR-STFT + log-mel、least-squares GAN、discriminator feature matching。
上流LLVCのMITコードからdiscriminatorを同梱し、periodは2/3/5へ減らしている。
**HuBERT知覚lossは実装しておらず、上流の学習再現ではない。** fairseq等の追加依存は不要。
十分な内容保持が得られるか、音質・声質ともに実データでの検証が必要。

- `config.json`: 再開条件とdataset/baseモデルのSHA256。
- `latest.pth`: generator/discriminator、両optimizer、scaler、crop/Torch/CUDA乱数状態、step。
- `best.pth`: 検証spectral lossが最小だった学習済みgenerator。元のbaseを上回る保証はない。
- `baseline.json` / `previews/base/`: 学習前の比較用。
- `metrics.jsonl` / `previews/step-N/`: lossと、元音声・教師・studentの試聴WAV。
  試聴は最大約4秒のオフライン出力で、リアルタイムの速度確認とは別。

既定で100 stepごとに再開用を保存、250 stepごとに検証・試聴・best更新する。終了時にも保存する。
中断時は**最後に完了した保存**から再開し、途中の不完全な更新は保存しない。
未保存区間を再計算するため、追記ログには同じstepが複数回現れる場合がある。

```bash
python -m engine.llvc.finetune train --dataset "logs/llvc-pairs/my-voice" --run-dir "logs/llvc-training/my-voice" --device cuda --steps 10000 --resume
```

`--steps`は追加回数ではなく総step数。10,000完了後に続けるなら20,000などへ増やす。
再開は同じ学習条件・データ・baseが前提。VRAM不足などでbatch/context等を変える場合は、
別runにし、必要に応じて`--base "旧run/best.pth"`から新しく微調整する。
`--amp`はCUDAでの省メモリ学習用だが、このクラウドでは実GPU検証できていない。

## 3. 試聴してWeb UIへ登録する

lossだけで声質・明瞭性を合格にしない。教師とstudentを比較し、学習に使っていない発話も試す。
採用するcheckpointを書き出す:

```bash
python -m engine.llvc.finetune export --checkpoint "logs/llvc-training/my-voice/best.pth" --voice-id my-voice --name "自分のRVCから追加学習した声"
```

`models/llvc/voices/my-voice/`へ推論用の`model.pth`と`voice.json`を保存する。
optimizer/discriminatorは含めず、初期モデルがなくてもその声だけで推論できる。
既存IDは上書きしない。再学習版は`my-voice-v2`など別IDで書き出す。
出力は「追加学習・試験」と表示し、未検証の品質を保証したモデルとして扱わない。

`npm run dev`でWeb UIを開き、「変換方式 → LLVC」→「声一覧を更新」→「LLVCの声」で新しい声を選ぶ。
設定はマイク開始時に反映する。登録RVCモデルの選択欄とは分けている。
RVC専用設定は直接適用せず、学習時に使った教師設定が声のモデルに含まれる。
モデルのファイルパスで推論キャッシュを分離し、別の声と取り違えないようにした。

実際の逐次推論を同じWAVで確認するコマンド:

```bash
python -m engine.llvc --model llvc:my-voice --input "held-out.wav" --output "logs/my-voice-stream.wav" --device cpu
```

構造・パラメータ数・13ms区間は維持するので、低遅延の実行方法はそのまま使える。
ただし**100ms以下の達成は、学習したモデル・手元の環境で別途計測する**。
CPU ONNX / CUDA Graphの表示、RTF、送信/再生待ち、途切れ、debug推定と物理ループバックを比較する。

## 検証範囲

Python全163件: 成功158・スキップ5・失敗0。JavaScript14件成功、Chromium16件成功。
スキップはRVC実モデル・LLVC公式重み・CUDA Graph実機2件・GPU学習/AMP。

教師データ生成のRVC部分はテスト専用の変換器で形式・長さ・split・改変検出・再開を確認した。
学習はフルサイズLLVC generatorの実際の勾配更新、デフォルトcontextでの逆伝播、
再開後と連続実行の重み一致、保存/書き出し/CPU ONNX推論を小規模CPU試験で確認した。
学習ループ試験のdiscriminatorは小型テスト用で、同梱した本物のdiscriminatorの逆伝播は別試験で確認した。
ブラウザ試験は声IDの選択→送信→診断→再生→停止/再開で、テスト用の未学習ネットワークを使う。

実RVCでの教師生成、GPUでの本番学習、RVC声質の再現、日本語の内容保持、リアルタイムの音質・100ms以下は未検証。
現在は単一教師の一つの声の追加学習。複数RVCの固定混合教師・動的Voice Mixingは次の作業。
