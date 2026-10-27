"""エンジンが使う資産（ONNX モデル等）の場所解決。

優先順位:
    1. 環境変数（VL_CONTENTVEC_ONNX / VL_RMVPE_ONNX）
    2. リポジトリ内 modules/ 配下
    3. VC Client のインストール先（既定の検証環境）
"""

from __future__ import annotations

import os
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = ENGINE_DIR.parent

#: 既定の探索先（この順に探す）
DEFAULT_CANDIDATES = {
    "contentvec": [
        Path(r"C:\Tools\vcclient_win_cuda\modules\contentvec\contentvec-f.onnx"),
        REPO_ROOT / "modules" / "contentvec" / "contentvec-f.onnx",
    ],
    "rmvpe": [
        Path(r"C:\Tools\vcclient_win_cuda\modules\rmvpe\rmvpe_20231006.onnx"),
        REPO_ROOT / "modules" / "rmvpe" / "rmvpe_20231006.onnx",
    ],
}

ENV_VARS = {
    "contentvec": "VL_CONTENTVEC_ONNX",
    "rmvpe": "VL_RMVPE_ONNX",
}


def resolve_asset(kind: str) -> Path:
    """ONNX 資産のパスを返す。見つからない場合は FileNotFoundError。"""
    if kind not in DEFAULT_CANDIDATES:
        raise KeyError(f"未知の資産種別: {kind}")

    env = os.environ.get(ENV_VARS[kind])
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
        raise FileNotFoundError(f"環境変数 {ENV_VARS[kind]} が指すファイルがありません: {p}")

    for p in DEFAULT_CANDIDATES[kind]:
        if p.is_file():
            return p

    tried = "\n".join(f"  - {p}" for p in DEFAULT_CANDIDATES[kind])
    raise FileNotFoundError(
        f"{kind} の ONNX が見つかりません。次のいずれかに置くか、"
        f"環境変数 {ENV_VARS[kind]} を設定してください:\n{tried}"
    )
