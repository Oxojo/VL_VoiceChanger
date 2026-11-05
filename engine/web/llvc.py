"""LLVC asset setup; uses the same reservation as microphone/file conversion."""
from urllib.error import URLError
from urllib.parse import urlsplit

from fastapi import HTTPException, Request

from ..llvc.assets import install_checkpoint, model_status


def register_llvc_routes(app, manager):
    @app.get('/api/llvc')
    def status():
        return model_status()

    @app.post('/api/llvc/install')
    def install(request: Request):
        origin = request.headers.get("origin")
        if origin and urlsplit(origin).netloc != request.headers.get("host"):
            raise HTTPException(403, "別のサイトからモデル取得を開始することはできません")
        try:
            manager.reserve_stream()
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        try:
            manifest = install_checkpoint()
            return {**model_status(), 'manifest': manifest}
        except (URLError, TimeoutError) as exc:
            raise HTTPException(502, 'LLVCモデルを取得できませんでした。Hugging Faceへの接続を確認してください。'
                                '手動で取得する場合は models/llvc/G_500000.pth に配置してください。') from exc
        except (ValueError, OSError) as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            manager.release_stream()
