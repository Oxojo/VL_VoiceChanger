"""ContentVec（HuBERT 系）による音声特徴抽出。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

#: RVC v1 は layer9（256 次元）、v2 は layer12（768 次元）を使う
LAYER_OUTPUT = {
    9: "units9",
    12: "unit12",
}


class ContentVec:
    """contentvec-f.onnx の推論ラッパ。

    入力: 16 kHz / float32 / モノラル波形 [1, N]
    出力: 50 fps（hop 320 サンプル）の特徴量 [1, T, dim]
    """

    def __init__(self, model_path: str | Path, providers: list[str] | None = None, log_severity: int = 3):
        so = ort.SessionOptions()
        so.log_severity_level = log_severity
        self.session = ort.InferenceSession(
            str(model_path),
            so,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self._outputs = {o.name: o for o in self.session.get_outputs()}

    @property
    def frame_hop(self) -> int:
        """特徴量 1 フレームあたりの入力サンプル数（16 kHz 換算）。"""
        return 320

    def extract(self, wav16k: np.ndarray, layer: int = 12) -> np.ndarray:
        """[N] の波形から [1, T, dim] の特徴量を返す。"""
        if layer not in LAYER_OUTPUT:
            raise ValueError(f"未対応の layer です: {layer}（9 か 12）")
        name = LAYER_OUTPUT[layer]
        if name not in self._outputs:
            raise RuntimeError(f"ONNX に出力 {name} がありません: {list(self._outputs)}")

        x = np.ascontiguousarray(wav16k, dtype=np.float32)[None, :]
        out = self.session.run([name], {self.input_name: x})[0]
        return np.ascontiguousarray(out, dtype=np.float32)
