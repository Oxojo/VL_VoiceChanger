"""Python 実行環境の解決と、engine コマンドの起動。

仮想環境は OS ごとに実行ファイルの場所が違う（Windows: ``.venv/Scripts/python.exe``、
macOS / Linux: ``.venv/bin/python``）。ここで吸収する。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from . import ui
from .config import REPO_ROOT

IS_WINDOWS = os.name == "nt"


class PythonNotFound(RuntimeError):
    """Python 実行ファイルが見つからない。"""


def venv_dir(repo: Path = REPO_ROOT) -> Path:
    return repo / ".venv"


def venv_python(repo: Path = REPO_ROOT) -> Path | None:
    candidates = [
        venv_dir(repo) / "Scripts" / "python.exe",  # Windows
        venv_dir(repo) / "bin" / "python",  # macOS / Linux
        venv_dir(repo) / "bin" / "python3",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def python_exe(override: str | None = None, repo: Path = REPO_ROOT) -> Path:
    """使う Python を決める。``--python`` > .venv > 実行中の Python。"""
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = (repo / path).resolve()
        if not path.is_file():
            raise PythonNotFound(f"Python が見つかりません: {path}")
        return path

    found = venv_python(repo)
    if found:
        return found

    # PATH 上の python3 / python
    for name in ("python3", "python"):
        exe = shutil.which(name)
        if exe:
            return Path(exe)

    if sys.executable:
        return Path(sys.executable)
    raise PythonNotFound(
        "Python が見つかりません。先に `vlvc setup` を実行するか --python で指定してください。"
    )


def env_for_python(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = str(REPO_ROOT)
    if extra:
        env.update(extra)
    return env


def run(
    args: Iterable[str],
    cwd: Path = REPO_ROOT,
    capture: bool = True,
    check: bool = False,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    command = [str(a) for a in args]
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=check,
        env=env or env_for_python(),
    )


def engine(
    args: Iterable[str],
    python: str | None = None,
    capture: bool = True,
    check: bool = False,
    repo: Path = REPO_ROOT,
) -> subprocess.CompletedProcess:
    """``python -m engine <args>`` を実行する。"""
    exe = python_exe(python, repo)
    return run([exe, "-m", "engine", *args], cwd=repo, capture=capture, check=check)


def capture_engine(args: Iterable[str], python: str | None = None, repo: Path = REPO_ROOT) -> tuple[int, str]:
    """engine を実行して (終了コード, 標準出力+標準エラー) を返す。"""
    proc = engine(args, python=python, capture=True, repo=repo)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output.strip()


def ensure_web_deps(python: str | None = None) -> None:
    """Web UI の依存関係が無ければ入れる（初回のみ）。"""
    exe = python_exe(python)
    probe = run([exe, "-c", "import fastapi, uvicorn, multipart"], capture=True)
    if probe.returncode == 0:
        return
    ui.info("Web UI の依存関係をインストールします（初回のみ）…")
    result = run([exe, "-m", "pip", "install", "fastapi", "uvicorn", "python-multipart"], capture=False)
    if result.returncode != 0:
        raise RuntimeError("Web UI の依存関係のインストールに失敗しました")


def setup(cuda: bool = False, python: str | None = None, repo: Path = REPO_ROOT) -> Path:
    """仮想環境を作り、依存関係を入れる。"""
    base = Path(python) if python else Path(sys.executable)
    if not base.is_file():
        base = python_exe(python, repo)

    target = venv_dir(repo)
    if not target.is_dir():
        ui.info(f"仮想環境を作成: {target}")
        result = run([base, "-m", "venv", target], capture=False)
        if result.returncode != 0:
            raise RuntimeError("仮想環境の作成に失敗しました")

    exe = venv_python(repo)
    if exe is None:
        raise PythonNotFound(f"仮想環境の Python が見つかりません: {target}")

    ui.info("pip を更新")
    run([exe, "-m", "pip", "install", "--upgrade", "pip"], capture=False)

    ui.info("engine/requirements.txt をインストール")
    result = run([exe, "-m", "pip", "install", "-r", str(repo / "engine" / "requirements.txt")], capture=False)
    if result.returncode != 0:
        raise RuntimeError("依存関係のインストールに失敗しました")

    if cuda:
        ui.info("CUDA 版 torch をインストール（cu126）")
        result = run(
            [
                exe,
                "-m",
                "pip",
                "install",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
                "--force-reinstall",
                "--no-deps",
                "torch",
            ],
            capture=False,
        )
        if result.returncode != 0:
            raise RuntimeError("CUDA 版 torch のインストールに失敗しました")

    return exe
