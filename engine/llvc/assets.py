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
            'target': '選択したLLVCモデルの固定ターゲット声', 'sample_rate': 16000, 'voices': list_voices()}


def resolve_voice(model_id='llvc-default'):
    if model_id == 'llvc-default':
        return CHECKPOINT
    if not model_id.startswith('llvc:') or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', model_id[5:]):
        raise ValueError('LLVCの声IDが不正です')
    root = (MODEL_DIR/'voices').resolve()
    path = (root/model_id[5:]/'model.pth').resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError('選択したLLVCの声が見つかりません')
    return path


def list_voices():
    voices = [{'id': 'llvc-default', 'name': 'LLVC配布モデルの声', 'installed': CHECKPOINT.is_file()}]
    root = MODEL_DIR/'voices'
    if root.is_dir():
        for directory in sorted(root.iterdir()):
            if not directory.is_dir() or not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', directory.name):
                continue
            try:
                resolve_voice('llvc:'+directory.name)
                metadata = json.loads((directory/'voice.json').read_text(encoding='utf-8'))
                if not isinstance(metadata,dict) or not isinstance(metadata.get('name'),str) or not metadata['name']:
                    continue
                voices.append({'id': 'llvc:'+directory.name, 'name': str(metadata['name'])[:80]+'（追加学習・試験）',
                               'installed': True, 'step': metadata.get('step'), 'experimental': True})
            except (OSError, ValueError, KeyError):
                continue
    return voices
