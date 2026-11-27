"""Web entry point for the explicitly untrained, audio-free StreamVC probe."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Literal

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict
import torch


class ProbeRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    device: Literal['cpu', 'cuda'] = 'cpu'
    threads: Literal[1, 2, 4] = 1


def run_probe(request: ProbeRequest):
    # Isolate RNG, thread counts, weights and CUDA allocations from the RVC
    # process. subprocess.run kills/waits for the child on timeout.
    with tempfile.TemporaryDirectory(prefix='vlvc-streamvc-') as directory:
        result_path = Path(directory) / 'result.json'
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(request.threads)
        env['MKL_NUM_THREADS'] = str(request.threads)
        subprocess.run(
            [sys.executable, '-m', 'engine.experimental.probe', '--frames', '100',
             '--threads', str(request.threads), '--device', request.device,
             '--json', str(result_path)],
            check=True, timeout=120, capture_output=True, text=True, env=env,
        )
        return json.loads(result_path.read_text(encoding='utf-8'))


def register_streamvc_routes(app, manager, static_dir):
    @app.get('/streamvc', response_class=HTMLResponse)
    def page():
        return HTMLResponse((static_dir / 'streamvc.html').read_text(encoding='utf-8'),
                            headers={'Cache-Control': 'no-store'})

    @app.post('/api/streamvc/probe')
    def probe(request: ProbeRequest):
        if request.device == 'cuda' and not torch.cuda.is_available():
            raise HTTPException(400, 'CUDAを利用できません。CUDA版PyTorchとGPUを確認してください。')
        try:
            manager.reserve_stream()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        try:
            return run_probe(request)
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(504, '計測が120秒以内に終了しませんでした。子プロセスを停止しました。') from exc
        except subprocess.CalledProcessError as exc:
            # Do not expose subprocess stderr containing local paths to clients.
            raise HTTPException(500, '計測に失敗しました。Python / PyTorchの実行環境を確認してください。') from exc
        except (OSError, ValueError) as exc:
            raise HTTPException(500, '計測結果を取得できませんでした。') from exc
        finally:
            manager.release_stream()
