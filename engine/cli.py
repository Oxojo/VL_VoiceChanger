"""自前エンジンのコマンドライン入口。

例:
    python -m engine inspect -m models/tsukuyomi/tsukuyomi-normal1.pth
    python -m engine convert -m models/tsukuyomi/tsukuyomi-normal1.pth ^
        -i logs/verify/tts-input-ja.wav -o logs/verify/own-ja.wav --pitch 0
    python -m engine analyze -i logs/verify/own-ja.wav
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .audio import SAMPLE_RATE, load_mono, rms_envelope, save_wav
from .config import resolve_asset
from .convert import ConvertOptions, VoiceConverter, interpolate_unvoiced
from .denoise import noise_floor_db, reduce_noise
from .features import RMVPE
from .model import build_model, load_checkpoint
from .realtime import ChunkConfig, ChunkedConverter


def _setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _fmt_seconds(sec: float) -> str:
    return f"{sec:.2f} 秒"


def cmd_inspect(args: argparse.Namespace) -> int:
    ckpt = load_checkpoint(args.model)
    model, info = build_model(ckpt, path=args.model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"モデル        : {Path(args.model).resolve()}")
    print(f"バージョン    : {info.version} / f0={info.has_f0} / 話者数={info.speakers}")
    print(f"サンプリング  : {info.sample_rate} Hz（潜在 1 フレーム = {model.hop_length} サンプル）")
    print(f"情報          : {info.info or '(なし)'}")
    print(f"パラメータ数  : {n_params / 1e6:.2f} M")
    print(f"config        : {info.config}")
    return 0


def cmd_convert(args: argparse.Namespace) -> int:
    options = ConvertOptions(
        pitch_shift=args.pitch,
        speaker=args.speaker,
        rms_mix=args.rms_mix,
        seed=None if args.seed < 0 else args.seed,
        output_sr=args.output_sr,
        denoise=args.denoise,
        dehum=args.dehum,
    )

    print("モデルと特徴量抽出器を読み込み中...")
    conv = VoiceConverter(
        args.model,
        device=args.device,
        contentvec_path=args.contentvec,
        rmvpe_path=args.rmvpe,
    )
    print(
        f"  モデル: {conv.info.sample_rate} Hz / 話者数 {conv.info.speakers} / "
        f"読み込み {_fmt_seconds(conv.load_seconds)}"
    )

    print(f"変換中: {args.input}")
    report = conv.convert_file(args.input, args.output, options)
    print(f"出力: {Path(args.output).resolve()}")
    print(
        f"  長さ {report.input_seconds:.2f} 秒 → {report.output_seconds:.2f} 秒 / "
        f"{report.output_sr} Hz / {report.frames} フレーム"
    )
    print(
        f"  F0 中央値 {report.f0_median:.1f} Hz / 有声率 {report.f0_voiced_ratio * 100:.1f} % / "
        f"RMS {report.rms_in:.4f} → {report.rms_out:.4f}"
    )
    if options.denoise > 0 or options.dehum > 0:
        print(
            f"  ノイズ除去: 強さ {options.denoise:.2f} / ハム {options.dehum or 'なし'} / "
            f"ノイズフロア {report.noise_floor_in_db:.1f} → {report.noise_floor_clean_db:.1f} dBFS"
        )
    print("  所要時間: " + ", ".join(f"{k} {v:.2f}s" for k, v in report.timings.items()))
    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "model": str(Path(args.model).resolve()),
                    "input": str(Path(args.input).resolve()),
                    "output": str(Path(args.output).resolve()),
                    "options": {
                        "pitch_shift": options.pitch_shift,
                        "speaker": options.speaker,
                        "rms_mix": options.rms_mix,
                        "seed": options.seed,
                        "output_sr": options.output_sr,
                        "denoise": options.denoise,
                        "dehum": options.dehum,
                    },
                    "report": {
                        "input_seconds": report.input_seconds,
                        "output_seconds": report.output_seconds,
                        "output_sr": report.output_sr,
                        "frames": report.frames,
                        "f0_median": report.f0_median,
                        "f0_voiced_ratio": report.f0_voiced_ratio,
                        "rms_in": report.rms_in,
                        "rms_out": report.rms_out,
                        "noise_floor_in_db": report.noise_floor_in_db,
                        "noise_floor_clean_db": report.noise_floor_clean_db,
                        "timings": report.timings,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"  JSON: {Path(args.json).resolve()}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    wav = load_mono(args.input, SAMPLE_RATE)
    f0_raw = RMVPE(args.rmvpe or resolve_asset("rmvpe")).infer(wav)
    voiced = f0_raw[f0_raw > 0]
    f0 = interpolate_unvoiced(f0_raw)
    rms, _ = rms_envelope(wav, SAMPLE_RATE)

    print(f"ファイル      : {Path(args.input).resolve()}")
    print(f"長さ          : {wav.shape[0] / SAMPLE_RATE:.3f} 秒（16 kHz 換算）")
    print(f"ピーク        : {float(abs(wav).max()):.4f} / RMS {float(rms.mean()):.4f}")
    if voiced.size:
        q = np.percentile(voiced, [5, 25, 50, 75, 95])
        print(
            "F0（有声のみ）: "
            f"中央値 {np.median(voiced):.1f} Hz / "
            f"5-95% {q[0]:.1f}-{q[4]:.1f} Hz / "
            f"有声率 {voiced.size / f0_raw.size * 100:.1f} %"
        )
        print(f"F0（補間後）  : 中央値 {np.median(f0[f0 > 0]):.1f} Hz")
    else:
        print("F0            : 有声区間が検出できませんでした")
    return 0


def cmd_denoise(args: argparse.Namespace) -> int:
    """ノイズ除去だけを行う（変換せずに効果を確認したいとき用）。"""
    import soundfile as sf

    audio, sr = sf.read(args.input, dtype="float32", always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)

    t = time.perf_counter()
    cleaned = reduce_noise(audio, sr, strength=args.strength, dehum_hz=args.dehum)
    seconds = time.perf_counter() - t

    save_wav(args.output, cleaned, sr)
    before = noise_floor_db(audio, sr)
    after = noise_floor_db(cleaned, sr)

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))

    print(f"入力: {Path(args.input).resolve()}（{sr} Hz / {audio.shape[0] / sr:.2f} 秒）")
    print(f"出力: {Path(args.output).resolve()}")
    print(f"  ノイズ除去: 強さ {args.strength:.2f} / ハム {args.dehum or 'なし'}")
    print(f"  ノイズフロア: {before:.1f} → {after:.1f} dBFS（{after - before:+.1f} dB）")
    print(f"  RMS: {rms(audio):.4f} → {rms(cleaned):.4f} / 処理 {seconds:.2f} 秒")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    conv = VoiceConverter(
        args.model,
        device=args.device,
        contentvec_path=args.contentvec,
        rmvpe_path=args.rmvpe,
    )
    wav = load_mono(args.input, SAMPLE_RATE)
    cfg = ChunkConfig(
        hop_seconds=args.hop,
        context_seconds=args.context,
        crossfade_seconds=args.crossfade,
        seed=None if args.seed < 0 else args.seed,
    )
    chunked = ChunkedConverter(conv, cfg)
    options = ConvertOptions(
        pitch_shift=args.pitch,
        speaker=args.speaker,
        rms_mix=args.rms_mix,
        seed=cfg.seed,
        output_sr=args.output_sr,
        denoise=args.denoise,
        dehum=args.dehum,
    )

    print(
        f"分割変換: hop {cfg.hop_seconds}s / context {cfg.context_seconds}s "
        f"/ crossfade {cfg.crossfade_seconds}s / デバイス {args.device}"
    )
    if options.denoise > 0 or options.dehum > 0:
        print(f"  ノイズ除去: 強さ {options.denoise:.2f} / ハム {options.dehum or 'なし'}（全体に 1 回）")
    out, stats = chunked.process(wav, options)

    if args.output:
        save_wav(args.output, out, int(options.output_sr or conv.info.sample_rate))
        print(f"出力: {Path(args.output).resolve()}")

    verdict = "可" if stats.realtime_ok(cfg.hop_seconds) else "不可"
    print(f"  区間数 {stats.chunks} / 音声 {stats.audio_seconds:.2f} 秒 / 処理 {stats.process_seconds:.2f} 秒")
    print(f"  RTF {stats.rtf:.3f}（1.0 未満で実時間より速い）")
    print(f"  区間あたり遅延 平均 {stats.mean_latency * 1000:.0f} ms / 最大 {stats.max_latency * 1000:.0f} ms")
    print(f"  ライブ成立（最大遅延 < hop）: {verdict}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """ブラウザ UI を起動する。"""
    try:
        import uvicorn
    except ImportError:
        print(
            "エラー: Web UI に必要なパッケージがありません。\n"
            "  .\\.venv\\Scripts\\python.exe -m pip install fastapi uvicorn python-multipart\n"
            "を実行してください。",
            file=sys.stderr,
        )
        return 2

    from .web.server import app

    url = f"http://{args.host}:{args.port}"
    print(f"ブラウザで {url} を開いてください（停止は Ctrl+C）")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, access_log=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="engine", description="自前 RVC v2 音声変換エンジン")
    p.add_argument("--version", action="version", version=f"engine {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pi = sub.add_parser("inspect", help="モデルの構成を表示する")
    pi.add_argument("-m", "--model", required=True, help="RVC v2 の .pth")
    pi.set_defaults(func=cmd_inspect)

    pc = sub.add_parser("convert", help="WAV を変換する")
    pc.add_argument("-m", "--model", required=True, help="RVC v2 の .pth")
    pc.add_argument("-i", "--input", required=True, help="入力 WAV")
    pc.add_argument("-o", "--output", required=True, help="出力 WAV")
    pc.add_argument("--pitch", type=int, default=0, help="ピッチシフト（半音、既定 0）")
    pc.add_argument("--speaker", type=int, default=0, help="話者 ID（既定 0）")
    pc.add_argument("--rms-mix", type=float, default=0.25, help="音量を入力に寄せる強さ 0-1")
    pc.add_argument("--seed", type=int, default=1234, help="乱数シード（-1 で毎回ランダム）")
    pc.add_argument("--output-sr", type=int, default=None, help="出力サンプリングレート（既定はモデル準拠）")
    pc.add_argument("--denoise", type=float, default=0.0, help="ノイズ除去の強さ 0-1（0 で無効、0.5 前後が無難）")
    pc.add_argument("--dehum", type=float, default=0.0, help="電源ハム除去（50 / 60 Hz、0 で無効）")
    pc.add_argument("--device", default="cpu", help="torch のデバイス（cpu / cuda）")
    pc.add_argument("--contentvec", default=None, help="ContentVec の ONNX パス")
    pc.add_argument("--rmvpe", default=None, help="RMVPE の ONNX パス")
    pc.add_argument("--json", default=None, help="結果 JSON の保存先")
    pc.set_defaults(func=cmd_convert)

    pa = sub.add_parser("analyze", help="WAV の F0・音量を表示する")
    pa.add_argument("-i", "--input", required=True, help="解析する WAV")
    pa.add_argument("--rmvpe", default=None, help="RMVPE の ONNX パス")
    pa.set_defaults(func=cmd_analyze)

    pb = sub.add_parser("bench", help="分割変換の速度（RTF）を計測する")
    pb.add_argument("-m", "--model", required=True, help="RVC v2 の .pth")
    pb.add_argument("-i", "--input", required=True, help="入力 WAV")
    pb.add_argument("-o", "--output", default=None, help="分割変換の結果を書き出す WAV")
    pb.add_argument("--hop", type=float, default=0.5, help="1 回で進める長さ（秒）")
    pb.add_argument("--context", type=float, default=0.15, help="前後に付ける文脈（秒）")
    pb.add_argument("--crossfade", type=float, default=0.05, help="継ぎ目のクロスフェード長（秒）")
    pb.add_argument("--pitch", type=int, default=0, help="ピッチシフト（半音）")
    pb.add_argument("--speaker", type=int, default=0, help="話者 ID")
    pb.add_argument("--rms-mix", type=float, default=0.25, help="音量を入力に寄せる強さ 0-1")
    pb.add_argument("--seed", type=int, default=1234, help="乱数シード（-1 で毎回ランダム）")
    pb.add_argument("--output-sr", type=int, default=None, help="出力サンプリングレート")
    pb.add_argument("--denoise", type=float, default=0.0, help="ノイズ除去の強さ 0-1（全体に 1 回かける）")
    pb.add_argument("--dehum", type=float, default=0.0, help="電源ハム除去（50 / 60 Hz、0 で無効）")
    pb.add_argument("--device", default="cpu", help="torch のデバイス（cpu / cuda）")
    pb.add_argument("--contentvec", default=None, help="ContentVec の ONNX パス")
    pb.add_argument("--rmvpe", default=None, help="RMVPE の ONNX パス")
    pb.set_defaults(func=cmd_bench)

    pn = sub.add_parser("denoise", help="ノイズ除去だけを行う（変換なし）")
    pn.add_argument("-i", "--input", required=True, help="入力 WAV")
    pn.add_argument("-o", "--output", required=True, help="出力 WAV")
    pn.add_argument("--strength", type=float, default=0.6, help="ノイズ除去の強さ 0-1（既定 0.6）")
    pn.add_argument("--dehum", type=float, default=0.0, help="電源ハム除去（50 / 60 Hz、0 で無効）")
    pn.set_defaults(func=cmd_denoise)

    ps = sub.add_parser("serve", help="ブラウザ UI を起動する")
    ps.add_argument("--host", default="127.0.0.1", help="待ち受けアドレス（既定はこの PC のみ）")
    ps.add_argument("--port", type=int, default=8765, help="待ち受けポート（既定 8765）")
    ps.add_argument("--log-level", default="warning", help="uvicorn のログレベル")
    ps.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    _setup_console()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        return int(args.func(args))
    except FileNotFoundError as exc:
        print(f"エラー: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # 想定外でも traceback を出して原因を追えるようにする
        import traceback

        traceback.print_exc()
        print(f"エラー: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
