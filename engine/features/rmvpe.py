"""RMVPE による F0 推定。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class RMVPE:
    """rmvpe_20231006.onnx の推論ラッパ。

    この ONNX（VC Client 同梱版）はメル spectrogram の計算と
    cents 変換をグラフ内に含んでおり、波形を入れると F0 [Hz] が返る。

    入力: waveform [1, N]（16 kHz / float32）, threshold [1]
    出力: pitchf [1, T]（100 fps = hop 160 サンプル、無声区間は 0）
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
        threshold: float = 0.03,
        log_severity: int = 3,
    ):
        so = ort.SessionOptions()
        so.log_severity_level = log_severity
        self.session = ort.InferenceSession(
            str(model_path),
            so,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.inputs = {i.name: i for i in self.session.get_inputs()}
        self.threshold = float(threshold)

    @property
    def frame_hop(self) -> int:
        """1 フレームあたりの入力サンプル数（16 kHz 換算）。"""
        return 160

    def infer(self, wav16k: np.ndarray) -> np.ndarray:
        """[N] の波形から [T] の F0 [Hz] を返す。"""
        x = np.ascontiguousarray(wav16k, dtype=np.float32)[None, :]
        feed: dict[str, np.ndarray] = {"waveform": x}
        if "threshold" in self.inputs:
            feed["threshold"] = np.array([self.threshold], dtype=np.float32)

        out = self.session.run(None, feed)[0]
        f0 = np.asarray(out, dtype=np.float32).reshape(-1)
        return np.maximum(f0, 0.0)
