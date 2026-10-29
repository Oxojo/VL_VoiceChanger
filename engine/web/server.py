"""自前エンジンをブラウザから使うための Web サーバー（FastAPI）。

    python -m engine serve

ブラウザで http://127.0.0.1:8765 を開くと、音声ファイルの変換・試聴ができる。
変換は Python 側（このプロセス）で実行し、ブラウザは UI と音声の受け渡しだけを担当する。

設計上の前提:
- 変換は 1 本のワーカースレッドで直列実行（GPU を奪い合わない）
- 変換器（モデル + ONNX）はプロセス内でキャッシュし、2 回目以降の待ち時間を無くす
- ジョブはメモリ上に保持し、音声は logs/web/<job_id>/ に置く（Git 管理外）
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from ..audio import SAMPLE_RATE, load_mono, resample, save_wav
from ..config import REPO_ROOT
from ..convert import ConvertOptions, VoiceConverter
from ..denoise import noise_floor_db, reduce_noise
from ..metrics import boundary_jumps, compare
from ..realtime import ChunkConfig, ChunkedConverter

STATIC_DIR = Path(__file__).resolve().parent / "static"
MODELS_DIR = REPO_ROOT / "models"
WORK_DIR = REPO_ROOT / "logs" / "web"
DENOISE_DIR = WORK_DIR / "denoise"

#: 受け付ける最大の音声長（秒）と最大アップロードサイズ（バイト）
MAX_SECONDS = 180.0
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
#: メモリに残すジョブ数
MAX_JOBS = 40
#: 入力用ノイズ除去の結果を残す件数
MAX_DENOISE = 20

STAGE_LABELS = {
    "queued": "待機中",
    "loading": "モデル読み込み中",
    "start": "前処理中",
    "denoise": "ノイズ除去中",
    "f0": "ピッチ推定中",
    "contentvec": "特徴量抽出中",
    "synth": "音声合成中",
    "post": "後処理中",
    "metrics": "品質を計測中",
    "done": "完了",
}


# ---------------------------------------------------------------- モデル探索
def _model_id(path: Path) -> str:
    return path.relative_to(MODELS_DIR).with_suffix("").as_posix()


def list_models() -> list[dict]:
    if not MODELS_DIR.is_dir():
        return []
    models = []
    for p in sorted(MODELS_DIR.rglob("*.pth")):
        models.append({"id": _model_id(p), "name": p.stem, "size_mb": round(p.stat().st_size / 1024 / 1024, 1)})
    return models


def model_path(model_id: str) -> Path:
    """ID（models/ からの相対パス、拡張子なし）を実パスに解決する。"""
    candidate = (MODELS_DIR / f"{model_id}.pth").resolve()
    if MODELS_DIR.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"モデルが見つかりません: {model_id}")
    return candidate


# ------------------------------------------------------------------ ジョブ
@dataclass
class Job:
    id: str
    request: dict
    status: str = "queued"
    stage: str = "待機中"
    progress: float = 0.0
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    input_path: Path | None = None
    output_path: Path | None = None
    input_seconds: float = 0.0
    output_seconds: float = 0.0
    output_sr: int = 0
    timings: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    chunks: int = 0
    rtf: float | None = None
    latency_mean_ms: float | None = None
    latency_max_ms: float | None = None
    realtime_ok: bool | None = None
    cancel_requested: bool = False

    def summary(self) -> dict:
        return {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "progress": round(self.progress, 3),
            "error": self.error,
            "request": self.request,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "input_seconds": self.input_seconds,
            "output_seconds": self.output_seconds,
            "output_sr": self.output_sr,
            "timings": self.timings,
            "metrics": self.metrics,
            "chunks": self.chunks,
            "rtf": self.rtf,
            "latency_mean_ms": self.latency_mean_ms,
            "latency_max_ms": self.latency_max_ms,
            "realtime_ok": self.realtime_ok,
            "audio": f"/api/jobs/{self.id}/audio" if self.output_path else None,
            "input_audio": f"/api/jobs/{self.id}/audio?which=input" if self.input_path else None,
        }


class JobManager:
    """変換ジョブの受け付けと実行（1 本のワーカーで直列実行）。"""

    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vc-job")
        self._converters: dict[tuple[str, str], VoiceConverter] = {}
        self._model_info: dict[str, dict] = {}
        WORK_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------- 参照系
    def get(self, job_id: str) -> Job:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="ジョブが見つかりません")
        return job

    def recent(self, limit: int = 20) -> list[dict]:
        with self._lock:
            ids = list(reversed(self._order))[:limit]
            return [self._jobs[i].summary() for i in ids if i in self._jobs]

    def cached_model_info(self, model_id: str) -> dict | None:
        return self._model_info.get(model_id)

    def model_info(self, model_id: str) -> dict:
        if model_id in self._model_info:
            return self._model_info[model_id]
        path = model_path(model_id)
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        cfg = list(ckpt.get("config", []))
        sr = cfg[17] if len(cfg) > 17 else ckpt.get("sr", "")
        info = {
            "id": model_id,
            "name": path.stem,
            "sample_rate": int(str(sr).rstrip("kK")) * 1000 if str(sr).endswith("k") else int(sr or 40000),
            "version": str(ckpt.get("version", "v1")),
            "speakers": int(cfg[15]) if len(cfg) > 15 else 1,
        }
        del ckpt
        self._model_info[model_id] = info
        return info

    # -------------------------------------------------------- 実行系
    def submit(self, job: Job, upload: UploadFile) -> Job:
        job_dir = WORK_DIR / job.id
        job_dir.mkdir(parents=True, exist_ok=True)

        suffix = Path(upload.filename or "input.wav").suffix.lower() or ".wav"
        if suffix not in {".wav", ".flac", ".ogg", ".mp3"}:
            raise HTTPException(status_code=400, detail="WAV / FLAC / OGG のいずれかを送ってください")
        input_path = job_dir / f"input{suffix}"
        size = 0
        with input_path.open("wb") as f:
            while chunk := upload.file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    f.close()
                    input_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=413, detail="ファイルが大きすぎます（80 MB まで）")
                f.write(chunk)

        try:
            seconds = float(sf.info(str(input_path)).duration)
        except Exception as exc:  # noqa: BLE001 - 壊れたファイルはそのまま 400 にする
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"音声として読めません: {exc}") from exc
        if seconds > MAX_SECONDS:
            input_path.unlink(missing_ok=True)
            raise HTTPException(status_code=413, detail=f"音声が長すぎます（{MAX_SECONDS:.0f} 秒まで）")

        job.input_path = input_path
        job.input_seconds = round(seconds, 3)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim_locked()
        self._executor.submit(self._run, job)
        return job

    def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        job.cancel_requested = True
        if job.status == "queued":
            job.status = "canceled"
            job.stage = "中断しました"
        return job

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)
            if job_id in self._order:
                self._order.remove(job_id)
        job_dir = WORK_DIR / job_id
        if job_dir.is_dir():
            for p in job_dir.iterdir():
                p.unlink(missing_ok=True)
            job_dir.rmdir()

    # -------------------------------------------------------- 内部
    def _trim_locked(self) -> None:
        while len(self._order) > MAX_JOBS:
            old = self._order.pop(0)
            job = self._jobs.pop(old, None)
            if job is not None:
                job_dir = WORK_DIR / old
                if job_dir.is_dir():
                    for p in job_dir.iterdir():
                        p.unlink(missing_ok=True)
                    job_dir.rmdir()

    def _converter(self, model_id: str, device: str, job: Job) -> VoiceConverter:
        key = (model_id, device)
        conv = self._converters.get(key)
        if conv is not None:
            return conv
        job.stage = STAGE_LABELS["loading"]
        path = model_path(model_id)
        conv = VoiceConverter(path, device=device)
        # 直近 2 つだけ保持（GPU メモリとロード時間のバランス）
        if len(self._converters) >= 2:
            self._converters.pop(next(iter(self._converters)))
        self._converters[key] = conv
        return conv

    def _run(self, job: Job) -> None:
        job.started_at = time.time()
        job.status = "running"
        try:
            req = job.request
            conv = self._converter(req["model"], req["device"], job)
            wav = load_mono(job.input_path, SAMPLE_RATE)  # type: ignore[arg-type]

            options = ConvertOptions(
                pitch_shift=req["pitch"],
                speaker=req["speaker"],
                rms_mix=req["rms_mix"],
                seed=req["seed"],
                output_sr=req["output_sr"],
                denoise=req.get("denoise", 0.0),
                dehum=req.get("dehum", 0.0),
            )

            job.stage = STAGE_LABELS["start"]
            if req["mode"] == "chunk":
                cfg = ChunkConfig(
                    hop_seconds=req["hop"],
                    context_seconds=req["context"],
                    crossfade_seconds=req["crossfade"],
                    seed=req["seed"],
                )
                chunked = ChunkedConverter(conv, cfg)

                def on_progress(done: int, total: int) -> None:
                    job.stage = f"分割変換中（{done}/{total} 区間）"
                    job.progress = done / max(total, 1)

                out, stats = chunked.process(
                    wav,
                    options,
                    on_progress=on_progress,
                    should_stop=lambda: job.cancel_requested,
                )
                job.chunks = stats.chunks
                job.rtf = round(float(stats.rtf), 3)
                job.latency_mean_ms = round(float(stats.mean_latency) * 1000, 1)
                job.latency_max_ms = round(float(stats.max_latency) * 1000, 1)
                job.timings = {"process": round(stats.process_seconds, 3)}
                if stats.canceled:
                    job.status = "canceled"
                    job.stage = "中断しました"
                    job.finished_at = time.time()
                    return
                sr_out = int(options.output_sr or conv.info.sample_rate)
                job.realtime_ok = bool(stats.max_latency < cfg.hop_seconds)
            else:
                def on_stage(stage: str, progress: float) -> None:
                    job.stage = STAGE_LABELS.get(stage, stage)
                    job.progress = progress

                out, report = conv.convert_array(wav, options, on_stage=on_stage)
                sr_out = report.output_sr
                job.timings = {k: round(v, 3) for k, v in report.timings.items()}
                convert_seconds = sum(v for k, v in report.timings.items() if k in {"f0", "contentvec", "synth", "post"})
                job.timings["convert"] = round(convert_seconds, 3)
                job.rtf = round(convert_seconds / max(report.input_seconds, 1e-6), 3)

            if job.cancel_requested:
                job.status = "canceled"
                job.stage = "中断しました"
                job.finished_at = time.time()
                return

            job.progress = 0.95
            output_path = WORK_DIR / job.id / "output.wav"
            save_wav(output_path, out, sr_out)
            job.output_path = output_path
            job.output_sr = sr_out
            job.output_seconds = round(out.shape[0] / sr_out, 3)

            if req.get("metrics", True):
                job.stage = STAGE_LABELS["metrics"]
                out16 = resample(out, sr_out, SAMPLE_RATE)
                f_in = conv.rmvpe.infer(wav)
                f_out = conv.rmvpe.infer(out16)
                job.metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in compare(wav, out16, f_in, f_out).items()}
                job.metrics["noise_floor_in_db"] = round(noise_floor_db(wav, SAMPLE_RATE), 1)
                job.metrics["noise_floor_out_db"] = round(noise_floor_db(out16, SAMPLE_RATE), 1)
                if req["mode"] == "chunk":
                    mean_jump, max_jump = boundary_jumps(out, sr_out, req["hop"])
                    job.metrics["boundary_jump_mean"] = round(mean_jump, 4)
                    job.metrics["boundary_jump_max"] = round(max_jump, 4)

            job.progress = 1.0
            job.status = "done"
            job.stage = STAGE_LABELS["done"]
        except Exception as exc:  # noqa: BLE001 - 失敗理由は UI にそのまま出す
            job.status = "error"
            job.stage = "エラー"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = time.time()
            self._write_sidecar(job)

    @staticmethod
    def _write_sidecar(job: Job) -> None:
        try:
            path = WORK_DIR / job.id / "job.json"
            path.write_text(json.dumps(job.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass


class DenoiseStore:
    """入力音声に単体でノイズ除去をかけた結果の一時保存（直近 MAX_DENOISE 件）。"""

    def __init__(self) -> None:
        self._items: dict[str, dict] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        DENOISE_DIR.mkdir(parents=True, exist_ok=True)

    def add(self, item: dict) -> None:
        with self._lock:
            self._items[item["id"]] = item
            self._order.append(item["id"])
            self._trim_locked()

    def get(self, item_id: str) -> dict:
        with self._lock:
            item = self._items.get(item_id)
        if item is None:
            raise HTTPException(status_code=404, detail="処理結果が見つかりません")
        return item

    def _trim_locked(self) -> None:
        while len(self._order) > MAX_DENOISE:
            old = self._order.pop(0)
            self._items.pop(old, None)
            d = DENOISE_DIR / old
            if d.is_dir():
                for p in d.iterdir():
                    p.unlink(missing_ok=True)
                d.rmdir()


manager = JobManager()
denoise_store = DenoiseStore()
app = FastAPI(title="VL VoiceChanger", docs_url="/api/docs", redoc_url=None)


# ------------------------------------------------------------------ ルート
@app.get("/api/health")
def health() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "app": "VL VoiceChanger",
        "torch": torch.__version__,
        "cuda": cuda,
        "device": torch.cuda.get_device_name(0) if cuda else "CPU",
        "default_device": "cuda" if cuda else "cpu",
        "models_dir": str(MODELS_DIR),
        "limits": {"max_seconds": MAX_SECONDS, "max_upload_mb": MAX_UPLOAD_BYTES // 1024 // 1024},
    }


@app.get("/api/models")
def get_models() -> dict:
    models = list_models()
    for m in models:
        cached = manager.cached_model_info(m["id"])
        if cached:
            m.update({k: v for k, v in cached.items() if k not in {"id", "name"}})
    return {"models": models}


@app.get("/api/models/{model_id:path}")
def get_model(model_id: str) -> dict:
    try:
        return manager.model_info(model_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"モデル情報を読めません: {exc}") from exc


@app.post("/api/denoise")
async def denoise_input(
    audio: UploadFile = File(...),
    strength: float = Form(0.6),
    dehum: float = Form(0.0),
) -> dict:
    """入力音声にだけノイズ除去をかける（変換前の任意ステップ）。

    返した `audio` をそのまま変換に使える。処理は軽い（数秒の音声で 0.1 秒未満）。
    """
    if float(dehum) not in {0.0, 50.0, 60.0}:
        raise HTTPException(status_code=400, detail="dehum は 0 / 50 / 60 のいずれかです")
    strength = max(0.0, min(1.0, float(strength)))

    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="ファイルが大きすぎます（80 MB まで）")

    item_id = secrets.token_hex(6)
    item_dir = DENOISE_DIR / item_id
    item_dir.mkdir(parents=True, exist_ok=True)
    src = item_dir / "input.wav"
    src.write_bytes(raw)

    try:
        x, sr = sf.read(str(src), dtype="float32", always_2d=False)
    except Exception as exc:  # noqa: BLE001 - 壊れたファイルは 400 にする
        src.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"音声として読めません: {exc}") from exc
    if getattr(x, "ndim", 1) > 1:
        x = x.mean(axis=1)
    if x.shape[0] > MAX_SECONDS * sr:
        raise HTTPException(status_code=413, detail=f"音声が長すぎます（{MAX_SECONDS:.0f} 秒まで）")

    t = time.perf_counter()
    cleaned = reduce_noise(x, sr, strength=strength, dehum_hz=float(dehum))
    seconds = time.perf_counter() - t

    save_wav(item_dir / "clean.wav", cleaned, sr)
    item = {
        "id": item_id,
        "audio": f"/api/denoise/{item_id}/audio",
        "strength": strength,
        "dehum": float(dehum),
        "before_db": round(noise_floor_db(x, sr), 1),
        "after_db": round(noise_floor_db(cleaned, sr), 1),
        "duration": round(float(x.shape[0]) / sr, 3),
        "sample_rate": int(sr),
        "seconds": round(seconds, 3),
    }
    denoise_store.add(item)
    return item


@app.get("/api/denoise/{item_id}/audio")
def denoise_audio(item_id: str) -> FileResponse:
    denoise_store.get(item_id)
    path = DENOISE_DIR / item_id / "clean.wav"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="音声がありません")
    return FileResponse(path, media_type="audio/wav", filename="clean.wav")


@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> dict:
    return {"jobs": manager.recent(limit)}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    return manager.get(job_id).summary()


@app.get("/api/jobs/{job_id}/audio")
def get_audio(job_id: str, which: str = "output") -> FileResponse:
    job = manager.get(job_id)
    path = job.input_path if which == "input" else job.output_path
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="音声がまだありません")
    media = {".mp3": "audio/mpeg", ".ogg": "audio/ogg", ".flac": "audio/flac"}.get(path.suffix.lower(), "audio/wav")
    return FileResponse(path, media_type=media, filename=path.name)


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    return manager.cancel(job_id).summary()


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    manager.delete(job_id)
    return {"deleted": job_id}


@app.post("/api/jobs")
async def create_job(
    audio: UploadFile = File(...),
    model: str = Form(...),
    device: str = Form("auto"),
    pitch: int = Form(0),
    speaker: int = Form(0),
    rms_mix: float = Form(0.25),
    seed: int = Form(1234),
    output_sr: int = Form(0),
    mode: str = Form("batch"),
    hop: float = Form(0.5),
    context: float = Form(0.15),
    crossfade: float = Form(0.05),
    denoise: float = Form(0.0),
    dehum: float = Form(0.0),
    metrics: bool = Form(True),
) -> JSONResponse:
    model_path(model)  # 存在しなければ 404 を返す

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise HTTPException(status_code=400, detail="device は cpu / cuda / auto のいずれかです")
    if mode not in {"batch", "chunk"}:
        raise HTTPException(status_code=400, detail="mode は batch / chunk のいずれかです")
    if float(dehum) not in {0.0, 50.0, 60.0}:
        raise HTTPException(status_code=400, detail="dehum は 0 / 50 / 60 のいずれかです")

    request = {
        "model": model,
        "device": device,
        "pitch": max(-24, min(24, int(pitch))),
        "speaker": max(0, int(speaker)),
        "rms_mix": max(0.0, min(1.0, float(rms_mix))),
        "seed": int(seed),
        "output_sr": int(output_sr) if int(output_sr) > 0 else None,
        "mode": mode,
        "hop": max(0.1, min(2.0, float(hop))),
        "context": max(0.0, min(1.0, float(context))),
        "crossfade": max(0.0, min(0.2, float(crossfade))),
        "denoise": max(0.0, min(1.0, float(denoise))),
        "dehum": float(dehum),
        "metrics": bool(metrics),
    }

    job = Job(id=secrets.token_hex(6), request=request)
    job.stage = STAGE_LABELS["queued"]
    manager.submit(job, audio)
    return JSONResponse({"id": job.id}, status_code=202)


# ------------------------------------------------------------------ 静的配信
class NoCacheStaticFiles(StaticFiles):
    """開発中に UI の変更が即座に反映されるよう、キャッシュさせない。"""

    async def get_response(self, path: str, scope):  # type: ignore[override]
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response


app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index() -> HTMLResponse:
    """画面を返す。

    `?v=__BUILD__` を静的ファイルの更新時刻に置き換える。これで UI を直したときに
    ブラウザのキャッシュで古い画面が出ることがなくなる（毎回 bump しなくてよい）。
    """
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    stamp = int(
        max(
            (STATIC_DIR / name).stat().st_mtime
            for name in ("index.html", "app.js", "style.css")
        )
    )
    html = html.replace("?v=__BUILD__", f"?v={stamp}")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})
