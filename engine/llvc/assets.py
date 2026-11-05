"""Download a revision-pinned official LLVC checkpoint with LFS SHA256 check."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.request import urlopen

from ..config import REPO_ROOT

MODEL_DIR = REPO_ROOT / 'models' / 'llvc'
CHECKPOINT = MODEL_DIR / 'G_500000.pth'
MODEL_REPO = 'KoeAI/llvc'
MODEL_FILE = 'models/checkpoints/llvc/G_500000.pth'
MAX_BYTES = 512 * 1024 * 1024


def install_checkpoint():
    """Only the known official asset; no caller-supplied URLs or pickle execution."""
    with urlopen(f'https://huggingface.co/api/models/{MODEL_REPO}/revision/main?blobs=true', timeout=30) as response:
        info = json.load(response)
    revision = info.get('sha', '')
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('配布元のモデルrevisionを確認できませんでした')
    sibling = next((s for s in info.get('siblings', []) if s.get('rfilename') == MODEL_FILE), {})
    lfs = sibling.get('lfs') or {}
    expected = lfs.get('sha256', '')
    size = lfs.get('size', 0)
    if not re.fullmatch(r'[0-9a-f]{64}', expected) or not 0 < size <= MAX_BYTES:
        raise ValueError('配布元のファイルサイズ/SHA256を確認できませんでした')
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=MODEL_DIR, suffix='.part', delete=False) as output:
            temporary = Path(output.name)
            digest, received = hashlib.sha256(), 0
            url = f'https://huggingface.co/{MODEL_REPO}/resolve/{revision}/{MODEL_FILE}?download=true'
            with urlopen(url, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > size:
                        raise ValueError('ダウンロードしたモデルが想定サイズを超えています')
                    digest.update(chunk)
                    output.write(chunk)
        if received != size or digest.hexdigest() != expected:
            raise ValueError('モデルのSHA256またはサイズが一致しません')
        manifest = {'repo': MODEL_REPO, 'revision': revision, 'file': MODEL_FILE,
                    'sha256': expected, 'size': size,
                    'license': (info.get('cardData') or {}).get('license')}
        # The server prevents downloads while a conversion session is active.
        os.replace(temporary, CHECKPOINT)
        CHECKPOINT.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
        return manifest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def model_status():
    return {'installed': CHECKPOINT.is_file(), 'checkpoint': 'models/llvc/G_500000.pth',
            'source': f'https://huggingface.co/{MODEL_REPO}', 'backend': 'llvc',
            'target': 'LLVC配布モデルの固定ターゲット声', 'sample_rate': 16000}
