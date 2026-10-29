"""コンソール出力。

色は「端末に直接出しているとき」だけ使う。パイプやファイルへのリダイレクト、
``NO_COLOR`` 指定時はプレーンテキストになる（CI でもログが崩れない）。
"""

from __future__ import annotations

import os
import sys
from typing import Iterable

_RESET = "\033[0m"
_STYLES = {
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "gray": "\033[90m",
    "white": "\033[97m",
    "bold": "\033[1m",
}

_color: bool | None = None


def _enable_windows_vt() -> bool:
    """Windows のコンソールで ANSI エスケープを有効にする（失敗しても続行）。"""
    if os.name != "nt":
        return True
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False
        enable_virtual_terminal_processing = 0x0004
        kernel32.SetConsoleMode(handle, mode.value | enable_virtual_terminal_processing)
        return True
    except Exception:  # noqa: BLE001 - 色が出ないだけなので握りつぶす
        return False


def color_enabled() -> bool:
    global _color
    if _color is None:
        if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
            _color = False
        else:
            _color = _enable_windows_vt()
    return _color


def paint(text: str, style: str) -> str:
    if not color_enabled() or style not in _STYLES:
        return text
    return f"{_STYLES[style]}{text}{_RESET}"


def header(text: str) -> None:
    print(paint(f" {text}", "white"))


def step(text: str) -> None:
    print()
    print(paint(f"== {text}", "cyan"))


def ok(text: str) -> None:
    print(paint(f"  [OK]   {text}", "green"))


def info(text: str) -> None:
    print(paint(f"  [INFO] {text}", "gray"))


def warn(text: str) -> None:
    print(paint(f"  [WARN] {text}", "yellow"))


def err(text: str) -> None:
    print(paint(f"  [NG]   {text}", "red"), file=sys.stderr)


def item(label: str, value: object, width: int = 26) -> None:
    print(f"   {label:<{width}} {value}")


def bullet(text: str, style: str = "gray") -> None:
    print(paint(f"       - {text}", style))


def note(text: str, style: str = "cyan") -> None:
    print()
    print(paint(text, style))


def table(headers: Iterable[str], rows: Iterable[Iterable[object]]) -> None:
    """簡易テーブル（Markdown ではなく等幅表示）。"""
    head = [str(h) for h in headers]
    body = [[str(c) if c is not None else "-" for c in row] for row in rows]
    widths = [len(h) for h in head]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(head))
    print("  " + line)
    print("  " + "  ".join("-" * w for w in widths))
    for row in body:
        print("  " + "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))


def confirm(question: str) -> bool:
    """対話コンソールが無い場合は False（勝手に進めない）。"""
    if not sys.stdin.isatty():
        return False
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}
