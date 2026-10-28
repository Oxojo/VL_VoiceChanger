# 自前エンジン 動作確認レポート

生成日時: 2026-10-28 11:03:36
ブランチ: feature/self-engine（be72d6b）
実行環境: 3.12.10 torch 2.14.1+cu126 cuda True
モデル: models\tsukuyomi\tsukuyomi-normal1.pth
入力: logs\verify\tts-input-ja.wav
デバイス: cuda / ピッチシフト: 0 半音 / 分割: hop 0.5 秒・context 0.15 秒・crossfade 0.05 秒

## 1. 変換

```text
モデルと特徴量抽出器を読み込み中...
  モデル: 40000 Hz / 話者数 109 / 読み込み 3.66 秒
変換中: logs\verify\tts-input-ja.wav
出力: C:\Users\oxojo\VL_VoiceChanger\logs\verify\own-tts-input-ja-cuda.wav
  長さ 7.63 秒 → 7.63 秒 / 40000 Hz / 764 フレーム
  F0 中央値 223.9 Hz / 有声率 52.4 % / RMS 0.1154 → 0.1153
  所要時間: f0 0.69s, contentvec 0.85s, synth 1.06s, post 0.06s, total 2.68s
  JSON: C:\Users\oxojo\VL_VoiceChanger\logs\verify\own-tts-input-ja-cuda.json
```

## 2. 入力の解析

```text
ファイル      : C:\Users\oxojo\VL_VoiceChanger\logs\verify\tts-input-ja.wav
長さ          : 7.632 秒（16 kHz 換算）
ピーク        : 0.8956 / RMS 0.1074
F0（有声のみ）: 中央値 244.6 Hz / 5-95% 165.3-312.8 Hz / 有声率 52.4 %
F0（補間後）  : 中央値 223.5 Hz
```

## 3. 出力の解析

```text
ファイル      : C:\Users\oxojo\VL_VoiceChanger\logs\verify\own-tts-input-ja-cuda.wav
長さ          : 7.630 秒（16 kHz 換算）
ピーク        : 0.5985 / RMS 0.1024
F0（有声のみ）: 中央値 244.3 Hz / 5-95% 164.9-312.2 Hz / 有声率 51.8 %
F0（補間後）  : 中央値 225.2 Hz
```

## 4. 品質の数値比較（一括変換）

入力 : tts-input-ja.wav  7.63 秒 (16 kHz 換算)
出力 : own-tts-input-ja-cuda.wav  7.63 秒 / 40000 Hz

| 項目 | 入力 | 出力 |
| --- | --- | --- |
| F0 中央値 | 244.6 Hz | 244.3 Hz |
| 有声率 | 52.4 % | 51.8 % |
| RMS | 0.1176 | 0.1143 |
| ピーク | 0.896 | 0.598 |
| 調波性 | 0.960 | 0.969 |

F0 追従（log F0 の相関）    : 0.999
エネルギー包絡の相関        : 0.962
クリップ（|値| >= 1.0）     : なし

## 5. 分割変換（ライブ成立性）

```text
分割変換: hop 0.5s / context 0.15s / crossfade 0.05s / デバイス cuda
出力: C:\Users\oxojo\VL_VoiceChanger\logs\verify\own-tts-input-ja-cuda-chunk.wav
  区間数 16 / 音声 7.63 秒 / 処理 15.26 秒
  RTF 2.000（1.0 未満で実時間より速い）
  区間あたり遅延 平均 954 ms / 最大 2058 ms
  ライブ成立（最大遅延 < hop）: 不可
```

## 6. 分割変換の品質

入力 : tts-input-ja.wav  7.63 秒 (16 kHz 換算)
出力 : own-tts-input-ja-cuda-chunk.wav  7.63 秒 / 40000 Hz

| 項目 | 入力 | 出力 |
| --- | --- | --- |
| F0 中央値 | 244.6 Hz | 242.4 Hz |
| 有声率 | 52.4 % | 53.0 % |
| RMS | 0.1176 | 0.1211 |
| ピーク | 0.896 | 0.749 |
| 調波性 | 0.960 | 0.957 |

F0 追従（log F0 の相関）    : 0.999
エネルギー包絡の相関        : 0.969
クリップ（|値| >= 1.0）     : なし
継ぎ目の跳び（0.5s ごと）  : 平均 0.0126 / 最大 0.0552
全体の平均 |Δ|              : 0.0165

## 7. 出力ファイル

- 一括変換: `logs\verify\own-tts-input-ja-cuda.wav`（Git 管理外）
- 分割変換: `logs\verify\own-tts-input-ja-cuda-chunk.wav`（Git 管理外）
- 計測 JSON: `logs\verify\own-tts-input-ja-cuda.json`（Git 管理外）

## 8. 判定

| 項目 | 判定 |
| :--- | :--- |
| 変換が最後まで通る | OK（上記ログ参照） |
| 出力が無音でない | 出力の RMS が 0 でないこと |
| ピッチが入力に追従 | F0 追従（log F0 の相関）が 0.9 以上 |
| 継ぎ目が不連続でない | 継ぎ目の跳びが全体平均と同程度 |
| ライブ成立 | 分割変換の「最大遅延 < hop」 |

聴感（声質）の最終判定は人による試聴が必要（`docs/04-verification.md`）。