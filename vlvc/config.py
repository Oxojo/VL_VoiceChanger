"""設定ファイルの読み込み（config/settings.json / config/profiles/*.json）。

PowerShell 版の ``Common.ps1`` に相当する。パスは OS に依存しない形で解決する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = REPO_ROOT / "config" / "settings.json"

#: settings.json に無い項目の既定値（PowerShell 版と同じ）
DEFAULTS: dict[str, Any] = {
    "vcClient": {
        "root": "",
        "launcher": "start_http.bat",
        "port": 18000,
        "healthPath": "/api/hello",
        "startupTimeoutSeconds": 300,
    },
    "paths": {
        "profiles": "config/profiles",
        "models": "models",
        "reports": "reports",
        "logs": "logs",
    },
    "stabilityTest": {
        "durationMinutes": 10,
        "sampleSeconds": 5,
        "processName": "main,MMVC_Client*,MMVCServerSIO*,vcclient*",
    },
}


class SettingsError(RuntimeError):
    """設定ファイルが無い・読めない・JSON として壊れている。"""


def load_json(path: Path) -> dict[str, Any]:
    """UTF-8 の JSON を読む。``$comment`` キーは落とす。"""
    path = Path(path)
    if not path.is_file():
        raise SettingsError(f"設定ファイルが見つかりません: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"JSON の書式が不正です: {path} ({exc})") from exc
    if not isinstance(raw, dict):
        raise SettingsError(f"JSON のトップレベルがオブジェクトではありません: {path}")
    raw.pop("$comment", None)
    return raw


def empty_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def dot_get(obj: Any, path: str, default: Any = None) -> Any:
    """``"a.b.c"`` 形式で入れ子の値を取り出す。空文字・None は既定値に置き換える。"""
    current = obj
    for segment in path.split("."):
        if not isinstance(current, dict) or segment not in current:
            return default
        current = current[segment]
    return default if empty_value(current) else current


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    """相対パスをリポジトリ基準の絶対パスにする。"""
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    return (base / path).resolve()


def format_size(num_bytes: float) -> str:
    for unit, scale in (("TB", 1024**4), ("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if num_bytes >= scale:
            digits = 1 if unit == "TB" else 0
            return f"{num_bytes / scale:.{digits}f} {unit}"
    return f"{num_bytes:.0f} B"


class Settings:
    """settings.json の薄いラッパ。値が無いときは既定値を使う。"""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else SETTINGS_PATH
        self.raw: dict[str, Any] = load_json(self.path) if self.path.is_file() else {}

    def get(self, path: str, default: Any = None) -> Any:
        fallback = dot_get(DEFAULTS, path, default)
        return dot_get(self.raw, path, fallback)

    def int(self, path: str, default: int = 0) -> int:
        try:
            return int(self.get(path, default))
        except (TypeError, ValueError):
            return default

    def float(self, path: str, default: float = 0.0) -> float:
        try:
            return float(self.get(path, default))
        except (TypeError, ValueError):
            return default

    # ---------------------------------------------------------------- パス
    @property
    def vc_root(self) -> Path:
        value = str(self.get("vcClient.root", "") or "")
        return resolve_path(value) if value else Path()

    @property
    def launcher(self) -> str:
        return str(self.get("vcClient.launcher", "start_http.bat"))

    @property
    def port(self) -> int:
        return self.int("vcClient.port", 18000)

    @property
    def host(self) -> str:
        return "127.0.0.1"

    @property
    def health_path(self) -> str:
        return str(self.get("vcClient.healthPath", "/api/hello"))

    @property
    def startup_timeout(self) -> int:
        return self.int("vcClient.startupTimeoutSeconds", 300)

    def path_of(self, key: str) -> Path:
        """``paths.profiles`` などをリポジトリ基準で解決する。"""
        return resolve_path(str(self.get(f"paths.{key}", key)))

    @property
    def profiles_dir(self) -> Path:
        return self.path_of("profiles")

    @property
    def models_dir(self) -> Path:
        return self.path_of("models")

    @property
    def reports_dir(self) -> Path:
        return self.path_of("reports")

    @property
    def logs_dir(self) -> Path:
        return self.path_of("logs")


def resolve_profile(name_or_path: str, settings: Settings) -> Path:
    """プロファイル名（character-a）または JSON のパスを解決する。"""
    candidate = Path(name_or_path).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    if not candidate.suffix:
        candidate = Path(f"{name_or_path}.json")
    path = settings.profiles_dir / candidate
    if not path.is_file():
        raise SettingsError(f"プロファイルが見つかりません: {name_or_path}")
    return path.resolve()


def list_profiles(settings: Settings) -> list[str]:
    directory = settings.profiles_dir
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json") if not p.name.startswith("_"))


def load_profile(name_or_path: str, settings: Settings) -> tuple[Path, dict[str, Any]]:
    path = resolve_profile(name_or_path, settings)
    return path, load_json(path)


def profile_id(profile: dict[str, Any], path: Path) -> str:
    return str(dot_get(profile, "id", path.stem))


def profile_name(profile: dict[str, Any], path: Path) -> str:
    return str(dot_get(profile, "displayName", profile_id(profile, path)))
