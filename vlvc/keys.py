"""キー入力（安定性テストの事象記録用）。

Windows は ``msvcrt``、macOS / Linux は ``termios`` で 1 文字ずつ読む。
対話コンソールが無い（パイプ・CI・リダイレクト）場合は自動で無効になる。
"""

from __future__ import annotations

import os
import select
import sys

IS_WINDOWS = os.name == "nt"


class KeyListener:
    def __init__(self, stream=None) -> None:
        self.enabled = False
        self._stream = stream if stream is not None else sys.stdin
        self._fd: int | None = None
        self._saved: list | None = None
        self._msvcrt = None

        if not self._stream.isatty():
            return
        try:
            if IS_WINDOWS:
                import msvcrt

                self._msvcrt = msvcrt
                self.enabled = True
            else:
                import termios
                import tty

                self._fd = self._stream.fileno()
                self._termios = termios
                self._tty = tty
                self._saved = termios.tcgetattr(self._fd)
                tty.setcbreak(self._fd)
                self.enabled = True
        except Exception:  # noqa: BLE001 - 端末が無ければ記録しないだけ
            self.enabled = False

    def poll(self) -> str | None:
        if not self.enabled:
            return None
        if IS_WINDOWS:
            if self._msvcrt is not None and self._msvcrt.kbhit():
                char = self._msvcrt.getwch()
                return char.upper() if char.isalpha() else None
            return None

        if self._fd is None:
            return None
        ready, _, _ = select.select([self._stream], [], [], 0)
        if not ready:
            return None
        data = os.read(self._fd, 1)
        if not data:
            return None
        char = data.decode("utf-8", "replace")
        return char.upper() if char.isalpha() else None

    def read_line(self, prompt: str) -> str:
        """1 行入力（メモ用）。端末モードを一時的に戻す。"""
        self._restore()
        try:
            return input(prompt)
        except (EOFError, KeyboardInterrupt):
            return ""
        finally:
            self._raw()

    def close(self) -> None:
        self._restore()

    # ---------------------------------------------------------------- 内部
    def _restore(self) -> None:
        if IS_WINDOWS or self._fd is None or self._saved is None:
            return
        try:
            self._termios.tcsetattr(self._fd, self._termios.TCSADRAIN, self._saved)
        except Exception:  # noqa: BLE001
            pass

    def _raw(self) -> None:
        if IS_WINDOWS or self._fd is None:
            return
        try:
            self._tty.setcbreak(self._fd)
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "KeyListener":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
