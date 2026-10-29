"""VL_VoiceChanger 自前音声変換エンジン。

外部の変換サービスを使わず、特徴量抽出（ContentVec）・F0 推定（RMVPE）・
RVC v2 系の音声合成器を自前のコードで実行するためのパッケージ。

構成:
    engine.audio       WAV 入出力・リサンプル・前処理
    engine.features    ContentVec / RMVPE の ONNX 推論ラッパ
    engine.model       RVC v2 合成器（SynthesizerTrnMs768NSFsid 相当）
    engine.convert     入力 WAV → 出力 WAV の変換パイプライン
    engine.cli         コマンドライン入口
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
