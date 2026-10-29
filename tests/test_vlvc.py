"""``vlvc``（クロスプラットフォーム CLI）のテスト。

    .venv/Scripts/python.exe -m unittest discover -s tests -v

Windows 専用の処理（レジストリ・PnP デバイス）は呼ばない。
純関数と、一時ディレクトリで完結する処理だけを対象にする。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vlvc import cli, doctor, models, profiles, stability  # noqa: E402
from vlvc.config import (  # noqa: E402
    Settings,
    SettingsError,
    dot_get,
    format_size,
    list_profiles,
    resolve_path,
    resolve_profile,
)
from vlvc.keys import KeyListener  # noqa: E402
from vlvc.pyproc import PythonNotFound, python_exe  # noqa: E402


class DotGetTest(unittest.TestCase):
    def test_nested(self) -> None:
        data = {"a": {"b": {"c": 3}}}
        self.assertEqual(dot_get(data, "a.b.c"), 3)

    def test_missing(self) -> None:
        self.assertIsNone(dot_get({}, "a.b"))
        self.assertEqual(dot_get({}, "a.b", "x"), "x")

    def test_empty_string_uses_default(self) -> None:
        self.assertEqual(dot_get({"a": ""}, "a", "fallback"), "fallback")

    def test_non_dict_midway(self) -> None:
        self.assertIsNone(dot_get({"a": 1}, "a.b"))


class FormatTest(unittest.TestCase):
    def test_size(self) -> None:
        self.assertEqual(format_size(512), "512 B")
        self.assertEqual(format_size(2048), "2 KB")
        self.assertEqual(format_size(5 * 1024**3), "5 GB")


class SettingsTest(unittest.TestCase):
    def test_defaults_without_file(self) -> None:
        settings = Settings(Path("does-not-exist.json"))
        self.assertEqual(settings.port, 18000)
        self.assertEqual(settings.launcher, "start_http.bat")
        self.assertEqual(settings.get("stabilityTest.sampleSeconds"), 5)

    def test_reads_json_and_drops_comment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text(
                json.dumps(
                    {
                        "$comment": "メモ",
                        "vcClient": {"port": 19000, "root": "tools/vc"},
                        "paths": {"reports": "out"},
                    }
                ),
                encoding="utf-8",
            )
            settings = Settings(path)
            self.assertEqual(settings.port, 19000)
            self.assertEqual(settings.path_of("reports"), resolve_path("out"))
            self.assertNotIn("$comment", settings.raw)

    def test_broken_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text("{", encoding="utf-8")
            with self.assertRaises(SettingsError):
                Settings(path)


class ProfileTest(unittest.TestCase):
    def test_resolve_and_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profiles_dir = Path(tmp) / "profiles"
            profiles_dir.mkdir()
            (profiles_dir / "character-a.json").write_text("{}", encoding="utf-8")
            (profiles_dir / "_template.json").write_text("{}", encoding="utf-8")
            settings_path = Path(tmp) / "settings.json"
            settings_path.write_text(json.dumps({"paths": {"profiles": str(profiles_dir)}}), encoding="utf-8")
            settings = Settings(settings_path)

            self.assertEqual(list_profiles(settings), ["character-a"])
            self.assertEqual(resolve_profile("character-a", settings).name, "character-a.json")
            self.assertEqual(resolve_profile("character-a.json", settings).name, "character-a.json")

    def test_missing_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings_path = Path(tmp) / "settings.json"
            settings_path.write_text(json.dumps({"paths": {"profiles": tmp}}), encoding="utf-8")
            with self.assertRaises(SettingsError):
                resolve_profile("nobody", Settings(settings_path))


class DeviceMatchTest(unittest.TestCase):
    def test_exact_partial_missing(self) -> None:
        known = ["マイク (USB Audio)", "ヘッドホン", "Voicemeeter Out B1"]
        self.assertEqual(profiles.match_device(known, "ヘッドホン"), ("exact", "ヘッドホン"))
        self.assertEqual(profiles.match_device(known, "usb audio"), ("partial", "マイク (USB Audio)"))
        self.assertEqual(profiles.match_device(known, "存在しない"), ("missing", None))


class BuildChangesTest(unittest.TestCase):
    def profile(self) -> dict:
        return {
            "model": {"pth": "models/x.pth", "index": ""},
            "audio": {"mode": "server", "inputDevice": "マイク", "outputDevice": "ヘッドホン"},
            "gpu": {"deviceId": 0},
            "convert": {"extra": 8192, "noiseGateDb": -60, "pitch": 12},
        }

    def test_changes(self) -> None:
        slots = [{"slot_index": 2, "model_file": "x.pth"}]
        inputs = [{"index": 4, "name": "マイク (USB)"}]
        outputs = [{"index": 9, "name": "ヘッドホン"}]
        changes, manual = profiles.build_changes(
            None, self.profile(), {"current_slot_index": 0}, slots, inputs, outputs
        )
        self.assertEqual(changes["current_slot_index"], 2)
        self.assertEqual(changes["voice_changer_input_mode"], "server")
        self.assertEqual(changes["audio_input_device_index"], 4)
        self.assertEqual(changes["audio_output_device_index"], 9)
        self.assertEqual(changes["gpu_device_id_int"], 0)
        self.assertEqual(changes["extra_frame_sec"], 0.1707)
        self.assertTrue(any("TUNE" in line for line in manual))

    def test_unregistered_model_is_skipped(self) -> None:
        changes, _ = profiles.build_changes(None, self.profile(), {}, [], [], [])
        self.assertNotIn("current_slot_index", changes)
        self.assertNotIn("audio_input_device_index", changes)


class StatsTest(unittest.TestCase):
    def test_stats(self) -> None:
        result = stability.stats([1, None, 3, 5])
        self.assertEqual(result, {"avg": 3.0, "min": 1.0, "max": 5.0, "n": 3})

    def test_empty(self) -> None:
        self.assertIsNone(stability.stats([None, None]))


class StabilityReportTest(unittest.TestCase):
    def test_markdown_sections(self) -> None:
        text = stability.build_markdown(
            generated_at="2026-10-29 12:00:00",
            profile="character-a",
            minutes=10,
            actual_seconds=601.0,
            sample_seconds=5,
            sample_count=120,
            process_pattern="main,vcclient*",
            url="http://localhost:18000/api/hello",
            csv_path=Path("logs/stability.csv"),
            proc_uptime=100.0,
            http_uptime=99.2,
            proc_down=0,
            port_down=0,
            http_fail=1,
            proc_stats={"avg": 10.0, "min": 5.0, "max": 20.0, "n": 120},
            gpu_stats=None,
            temp_stats=None,
            mem_stats={"avg": 500.0, "min": 400.0, "max": 600.0, "n": 120},
            incidents=[
                {"elapsed": 12.3, "clock": "12:00:12", "kind": "音切れ", "detail": "操作者が検知"}
            ],
            operator_events=1,
        )
        self.assertIn("# 安定性検証レポート", text)
        self.assertIn("GUI 応答率 | 99.2 %", text)
        self.assertIn("音切れ", text)
        self.assertIn("MVP 判定", text)
        self.assertIn("取得できず（nvidia-smi なし）", text)


class DoctorReportTest(unittest.TestCase):
    def data(self) -> dict:
        return {
            "generated_at": "2026-10-29T12:00:00+09:00",
            "host": "test-host",
            "platform": {"system": "Windows 11", "version": "10.0", "python": "3.12.10"},
            "cpu": {"name": "Test CPU", "cores": 14, "threads": 20},
            "memory_gb": 15.6,
            "free_disk_gb": 21.0,
            "gpus": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3060",
                    "driver": "591.86",
                    "vram_mb": 6144,
                    "source": "nvidia-smi",
                }
            ],
            "audio_devices": [
                {"name": "マイク", "type": "input", "status": "", "source": "test"},
                {"name": "ヘッドホン", "type": "output", "status": "", "source": "test"},
            ],
            "vc_client": {
                "root": "C:/Tools/vcclient",
                "exists": True,
                "launcher": "start_http.bat",
                "port": 18000,
                "port_open": False,
                "edition": "cuda",
                "version": "2.1.4-alpha",
                "has_cuda": True,
                "has_onnx": False,
                "model_count": 3,
                "api_ok": None,
                "modules_valid": None,
                "modules_total": None,
            },
            "findings": [{"level": "warn", "message": "テスト"}],
        }

    def test_render(self) -> None:
        text = doctor.render_markdown(self.data())
        self.assertIn("# 環境確認レポート", text)
        self.assertIn("RTX 3060", text)
        self.assertIn("マイク", text)
        self.assertIn("2.1.4-alpha", text)
        self.assertIn("[warn] テスト", text)


class FreeSlotTest(unittest.TestCase):
    def test_prefers_empty_slot(self) -> None:
        slots = [
            {"slot_index": 0, "voice_changer_type": "RVC"},
            {"slot_index": 1, "voice_changer_type": ""},
            {"slot_index": 2, "voice_changer_type": "RVC"},
        ]
        self.assertEqual(models._free_slot(slots), 1)

    def test_appends_when_full(self) -> None:
        slots = [{"slot_index": 0, "voice_changer_type": "RVC"}, {"slot_index": 1, "voice_changer_type": "RVC"}]
        self.assertEqual(models._free_slot(slots), 2)

    def test_empty(self) -> None:
        self.assertEqual(models._free_slot([]), 0)


class ParserTest(unittest.TestCase):
    def test_doctor(self) -> None:
        args = cli.build_parser().parse_args(["doctor"])
        self.assertEqual(args.command, "doctor")

    def test_profile_apply(self) -> None:
        args = cli.build_parser().parse_args(["profile", "apply", "character-a", "--dry-run"])
        self.assertEqual(args.profile_command, "apply")
        self.assertTrue(args.dry_run)

    def test_stability(self) -> None:
        args = cli.build_parser().parse_args(["stability", "--minutes", "1", "--allow-not-running"])
        self.assertEqual(args.minutes, 1)
        self.assertTrue(args.allow_not_running)


class KeyListenerTest(unittest.TestCase):
    def test_disabled_without_tty(self) -> None:
        import io

        listener = KeyListener(io.StringIO())
        # 端末ではないストリームでは無効になる
        self.assertFalse(listener.enabled)
        self.assertIsNone(listener.poll())
        listener.close()


class PythonExeTest(unittest.TestCase):
    def test_missing_override(self) -> None:
        with self.assertRaises(PythonNotFound):
            python_exe("no-such-python-xyz")


if __name__ == "__main__":
    unittest.main()
