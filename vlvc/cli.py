"""``vlvc`` のコマンドライン入口。

すべての機能を ``vlvc <コマンド>`` にまとめている。Windows の PowerShell は不要で、
macOS / Linux でも同じコマンドが動く。
"""

from __future__ import annotations

import argparse
import sys
import traceback

from . import __version__, ui
from .config import Settings, SettingsError, list_profiles


def _settings(args: argparse.Namespace) -> Settings:
    return Settings(getattr(args, "config", None))


# ------------------------------------------------------------------ setup
def cmd_setup(args: argparse.Namespace) -> int:
    from . import pyproc

    ui.header("VL_VoiceChanger / セットアップ")
    ui.info("仮想環境 .venv と依存関係を用意します（初回のみ数分かかります）。")
    if args.cuda:
        ui.info("CUDA 版 torch（cu126）を入れます。GPU が無い場合は付けないでください。")
    try:
        exe = pyproc.setup(cuda=args.cuda, python=args.python)
    except Exception as exc:  # noqa: BLE001
        ui.err(str(exc))
        return 1
    ui.ok(f"完了: {exe}")
    ui.note("次: `vlvc doctor` で環境を確認し、`vlvc serve` でブラウザ UI を起動してください。")
    return 0


# ------------------------------------------------------------------ doctor
def cmd_doctor(args: argparse.Namespace) -> int:
    from . import doctor

    return doctor.run(_settings(args), vc_root=args.vc_root, output_dir=args.output)


# ------------------------------------------------------------------ convert
def run_convert(argv: list[str]) -> int:
    """``python -m engine convert`` へそのまま渡す（引数は engine と同じ）。

    ``vlvc convert`` だけは argparse を通さない。``-m`` などの engine 側の
    オプションを、vlvc のオプションと取り違えないようにするため。
    """
    from . import pyproc

    if not argv:
        return pyproc.engine(["--help"], capture=False).returncode
    proc = pyproc.engine(["convert", *argv], capture=False)
    return proc.returncode


# ------------------------------------------------------------------ serve
def cmd_serve(args: argparse.Namespace) -> int:
    from . import serve

    return serve.run(python=args.python, port=args.port, host=args.host, no_browser=args.no_browser)


# ------------------------------------------------------------------ quality
def cmd_quality(args: argparse.Namespace) -> int:
    from . import quality

    return quality.run(args.input, args.output, hop=args.hop, rmvpe_path=args.rmvpe)


# ------------------------------------------------------------------ selftest
def cmd_selftest(args: argparse.Namespace) -> int:
    from . import selftest

    return selftest.run(
        _settings(args),
        model=args.model,
        input_path=args.input,
        python=args.python,
        device=args.device,
        pitch=args.pitch,
        hop=args.hop,
        context=args.context,
        crossfade=args.crossfade,
        report=args.report,
    )


# ------------------------------------------------------------------ launch
def cmd_launch(args: argparse.Namespace) -> int:
    from . import launch

    return launch.run(_settings(args), args.profile, dry_run=args.dry_run, timeout=args.timeout)


# ------------------------------------------------------------------ profile
def cmd_profile_list(args: argparse.Namespace) -> int:
    settings = _settings(args)
    names = list_profiles(settings)
    ui.header("VL_VoiceChanger / プロファイル一覧")
    ui.info(f"ディレクトリ: {settings.profiles_dir}")
    if not names:
        ui.warn("プロファイルがありません（config/profiles/*.json）")
        return 1
    for name in names:
        print(f"  - {name}")
    return 0


def cmd_profile_check(args: argparse.Namespace) -> int:
    from . import launch

    return launch.run(_settings(args), args.profile, dry_run=True, timeout=args.timeout)


def cmd_profile_apply(args: argparse.Namespace) -> int:
    from . import profiles

    return profiles.apply(_settings(args), args.profile, dry_run=args.dry_run)


# ------------------------------------------------------------------ register-model
def cmd_register_model(args: argparse.Namespace) -> int:
    from . import models

    return models.register(
        _settings(args),
        args.model,
        name=args.name,
        slot_index=args.slot,
        index_file=args.index,
        terms_url=args.terms,
        embedder=args.embedder,
        force=args.force,
    )


# ------------------------------------------------------------------ stability
def cmd_stability(args: argparse.Namespace) -> int:
    from . import stability

    return stability.run(
        _settings(args),
        minutes=args.minutes,
        sample_seconds=args.interval,
        profile=args.profile,
        process_pattern=args.process_name,
        url=args.url,
        timeout=args.timeout,
        allow_not_running=args.allow_not_running,
    )


# ------------------------------------------------------------------ parser
def _add_python(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--python",
        default=argparse.SUPPRESS,
        help="使用する Python（既定: .venv の python、無ければ PATH 上）",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vlvc",
        description="VL_VoiceChanger のクロスプラットフォーム CLI（Windows / macOS / Linux）",
        epilog="例: vlvc doctor / vlvc serve / vlvc convert -m models/xxx.pth -i in.wav -o out.wav",
    )
    parser.add_argument("--version", action="version", version=f"vlvc {__version__}")
    parser.add_argument(
        "--config",
        default=None,
        help="config/settings.json の代替パス",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="使用する Python（既定: .venv の python、無ければ PATH 上）",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<コマンド>")

    p = sub.add_parser("setup", help="仮想環境と依存関係を用意する")
    p.add_argument("--cuda", action="store_true", help="CUDA 版 torch（cu126）を入れる")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("doctor", help="機材・実行環境を確認してレポートを出す")
    p.add_argument("--vc-root", default=None, help="VC Client の配置フォルダ（settings.json を上書き）")
    p.add_argument("-o", "--output", default=None, help="レポート出力先（既定: reports/）")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("convert", help="WAV を変換する（engine convert と同じ引数）", add_help=False)
    p.set_defaults(func=lambda args: run_convert([]))

    p = sub.add_parser("serve", help="ブラウザ UI を起動する")
    p.add_argument("--port", type=int, default=8765, help="待ち受けポート（既定 8765）")
    p.add_argument("--host", default="127.0.0.1", help="待ち受けアドレス（既定はこの PC のみ）")
    p.add_argument("--no-browser", action="store_true", help="ブラウザを自動で開かない")
    _add_python(p)
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("quality", help="変換前後の音声を数値で比較する")
    p.add_argument("input", help="入力 WAV")
    p.add_argument("output", help="出力 WAV")
    p.add_argument("--hop", type=float, default=None, help="分割変換の hop（秒）")
    p.add_argument("--rmvpe", default=None, help="RMVPE の ONNX パス")
    p.set_defaults(func=cmd_quality)

    p = sub.add_parser("selftest", help="変換〜計測をまとめて実行しレポートを出す")
    p.add_argument("-m", "--model", default="models/tsukuyomi/tsukuyomi-normal1.pth", help="RVC v2 の .pth")
    p.add_argument("-i", "--input", default="logs/verify/tts-input-ja.wav", help="入力 WAV")
    p.add_argument("--device", default="cpu", help="cpu / cuda（既定 cpu）")
    p.add_argument("--pitch", type=int, default=0, help="ピッチシフト（半音）")
    p.add_argument("--hop", type=float, default=0.5, help="分割変換の hop（秒）")
    p.add_argument("--context", type=float, default=0.15, help="分割変換の前後文脈（秒）")
    p.add_argument("--crossfade", type=float, default=0.05, help="継ぎ目のクロスフェード（秒）")
    p.add_argument("--report", default=None, help="レポート出力先（既定: reports/own-engine-latest.md）")
    _add_python(p)
    p.set_defaults(func=cmd_selftest)

    p = sub.add_parser("launch", help="VC Client を起動して GUI 設定項目を表示する")
    p.add_argument("profile", nargs="?", default="character-a", help="プロファイル名またはパス")
    p.add_argument("--dry-run", action="store_true", help="起動せず検証とチェックリストのみ")
    p.add_argument("--timeout", type=int, default=None, help="GUI の応答待ち（秒）")
    p.set_defaults(func=cmd_launch)

    p = sub.add_parser("profile", help="プロファイルの一覧・検証・適用")
    psub = p.add_subparsers(dest="profile_command", required=True, metavar="<操作>")
    pp = psub.add_parser("list", help="プロファイルを一覧する")
    pp.set_defaults(func=cmd_profile_list)
    pp = psub.add_parser("check", help="検証して GUI 設定項目を表示する（起動なし）")
    pp.add_argument("profile", nargs="?", default="character-a", help="プロファイル名またはパス")
    pp.add_argument("--timeout", type=int, default=None, help="GUI の応答待ち（秒）")
    pp.set_defaults(func=cmd_profile_check)
    pp = psub.add_parser("apply", help="REST API で設定を反映する")
    pp.add_argument("profile", help="プロファイル名またはパス")
    pp.add_argument("--dry-run", action="store_true", help="適用せず差分だけ表示する")
    pp.set_defaults(func=cmd_profile_apply)

    p = sub.add_parser("register-model", help="VC Client に RVC モデルを登録する")
    p.add_argument("model", help="モデルファイル（.pth / .onnx）")
    p.add_argument("--name", default="", help="モデル一覧に表示する名前")
    p.add_argument("--slot", type=int, default=-1, help="登録先スロット（既定: 空きスロット）")
    p.add_argument("--index", default="", help="index ファイル名")
    p.add_argument("--terms", default="", help="利用規約 URL")
    p.add_argument("--embedder", default="", help="embedder（既定: 自動判定）")
    p.add_argument("--force", action="store_true", help="同じモデルが登録済みでも登録する")
    p.set_defaults(func=cmd_register_model)

    p = sub.add_parser("stability", help="連続使用テストを記録する")
    p.add_argument("--minutes", type=int, default=0, help="連続使用時間（分、既定は settings.json）")
    p.add_argument("--interval", type=int, default=0, help="サンプリング間隔（秒、既定 5）")
    p.add_argument("--profile", default="", help="レポートに記録するプロファイル名")
    p.add_argument("--process-name", default="", help="プロセス名パターン（カンマ区切り）")
    p.add_argument("--url", default="", help="監視する URL（既定は settings.json の healthPath）")
    p.add_argument("--timeout", type=int, default=5, help="HTTP タイムアウト（秒）")
    p.add_argument("--allow-not-running", action="store_true", help="VC Client 未起動でも続行する")
    p.set_defaults(func=cmd_stability)

    p = sub.add_parser("version", help="バージョンを表示する")
    p.set_defaults(func=lambda args: print(f"vlvc {__version__}") or 0)

    return parser


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:  # noqa: BLE001
            pass

    parser = build_parser()

    # convert は engine の引数をそのまま受けるため、argparse に通さない
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "convert":
        try:
            return int(run_convert(argv[1:]))
        except SettingsError as exc:
            ui.err(str(exc))
            return 2
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            ui.err(str(exc))
            return 1

    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except SettingsError as exc:
        ui.err(str(exc))
        return 2
    except FileNotFoundError as exc:
        ui.err(str(exc))
        return 2
    except KeyboardInterrupt:
        ui.info("中断しました。")
        return 130
    except Exception as exc:  # noqa: BLE001 - 想定外でも原因を追えるようにする
        traceback.print_exc()
        ui.err(str(exc))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
