"""Prepare paired source/RVC audio for single-target LLVC fine-tuning."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch

from ..audio import resample
from ..convert import ConvertOptions, VoiceConverter


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(1024*1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
            json.dump(value, file, indent=2, ensure_ascii=False, allow_nan=False)
            file.write('\n')
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def prepare_pairs(teacher, audio_dir, output, *, device='cuda', pitch=0, speaker=0,
                  rms_mix=.25, clip_seconds=4., val_fraction=.1, seed=1234, resume=False):
    teacher, audio_dir, output = Path(teacher).resolve(), Path(audio_dir).resolve(), Path(output).resolve()
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA版PyTorchの環境を使用するか、--device cpuを指定してください")
    if not teacher.is_file():
        raise ValueError('RVC教師モデルが見つかりません')
    if not 2 <= clip_seconds <= 20 or not 0 < val_fraction <= .5:
        raise ValueError('clip_secondsは2〜20、val_fractionは0より大きく0.5以下にしてください')
    if not -24 <= pitch <= 24 or speaker < 0 or not 0 <= rms_mix <= 1:
        raise ValueError('教師のピッチ/話者/RMS設定が不正です')
    if output == audio_dir or audio_dir in output.parents or output in audio_dir.parents:
        raise ValueError('出力は入力音声フォルダーと重ならない場所にしてください')
    files = sorted(p for p in audio_dir.rglob('*') if p.is_file() and p.suffix.lower() in {'.wav', '.flac', '.ogg'})
    if len(files) < 2:
        raise ValueError('検証用を分離するため、入力音声ファイルを2本以上用意してください')
    # Split by entire recording; clips from the same file never cross splits.
    files.sort(key=lambda p: hashlib.sha256(f'{seed}:{p.relative_to(audio_dir).as_posix()}'.encode()).hexdigest())
    unique, seen = [], set()
    for path in files:
        digest = sha256(path)
        if digest not in seen:
            unique.append((path,digest))
            seen.add(digest)
    if len(unique) < 2:
        raise ValueError('同一ファイルの複製を除いて、音声を2本以上用意してください')
    val_count = max(1, math.ceil(len(unique)*val_fraction))
    files = [p for p,_ in unique]
    sources = [{'name': p.relative_to(audio_dir).as_posix(), 'sha256': digest,
                'split': 'val' if i < val_count else 'train'} for i,(p,digest) in enumerate(unique)]
    signature = {'version': 1, 'teacher_name': teacher.stem, 'teacher_sha256': sha256(teacher),
                 'pitch': pitch, 'speaker': speaker, 'rms_mix': rms_mix, 'seed': seed,
                 'clip_seconds': clip_seconds, 'val_fraction': val_fraction, 'sample_rate': 16000, 'sources': sources}
    manifest_path = output/'manifest.json'
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not resume or manifest.get('config') != signature:
            raise ValueError('出力先に別のデータがあります。同じ設定なら--resumeを使ってください')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('出力先は空のフォルダーにしてください')
        output.mkdir(parents=True, exist_ok=True)
        manifest = {'config': signature, 'pairs': [], 'complete': False}
        atomic_json(manifest_path, manifest)
    completed = {p['id']: p for p in manifest['pairs']}
    converter = VoiceConverter(teacher, device=device)
    if speaker >= converter.info.speakers:
        raise ValueError('指定した話者IDは教師モデルにありません')
    options = ConvertOptions(pitch_shift=pitch, speaker=speaker, rms_mix=rms_mix, seed=seed, output_sr=16000)
    for path, source in zip(files, sources):
        with sf.SoundFile(path) as reader:
            blocksize = int(round(reader.samplerate*clip_seconds))
            for index, block in enumerate(reader.blocks(blocksize=blocksize, dtype='float32', always_2d=True)):
                audio = resample(block.mean(axis=1), reader.samplerate, 16000)
                # RVC synthesis uses complete 10ms frames. Align SOURCE before
                # rendering; mismatched teacher output is still an error.
                audio = audio[:audio.size//160*160]
                # Training uses random central crops with preceding context.
                if audio.size < 32000:
                    continue
                if not np.isfinite(audio).all():
                    raise ValueError(f'入力音声に非有限値があります: {source["name"]}')
                if float(np.sqrt(np.mean(audio.astype(np.float64)**2))) < 1e-5:
                    continue
                identity = hashlib.sha256(f'{source["sha256"]}:{source["name"]}:{index}'.encode()).hexdigest()[:24]
                split = source['split']
                original = output/split/f'{identity}_original.wav'
                target = output/split/f'{identity}_converted.wav'
                if identity in completed:
                    entry = completed[identity]
                    if (not original.is_file() or not target.is_file()
                            or sha256(original) != entry['source_sha256'] or sha256(target) != entry['target_sha256']):
                        raise ValueError('既存の音声ペアが欠落/変更されています。別の出力先で作成してください')
                    continue
                converted, _ = converter.convert_array(audio, options)
                converted = np.asarray(converted, dtype=np.float32)
                if converted.ndim != 1 or converted.size != audio.size or not np.isfinite(converted).all():
                    raise ValueError('RVC教師の出力長/値が不正です。無理に切り詰めず処理を停止しました')
                peak = max(float(np.max(np.abs(audio))), float(np.max(np.abs(converted))))
                joint_gain = min(1., .99/max(peak,1e-6))
                audio, converted = audio*joint_gain, converted*joint_gain
                original.parent.mkdir(parents=True, exist_ok=True)
                # Apply a shared gain; never normalize source/teacher independently.
                sf.write(original, audio, 16000, subtype='FLOAT')
                sf.write(target, converted, 16000, subtype='FLOAT')
                entry = {'id': identity, 'split': split, 'samples': int(audio.size), 'joint_gain': joint_gain,
                         'source': original.relative_to(output).as_posix(),
                         'target': target.relative_to(output).as_posix(),
                         'source_sha256': sha256(original), 'target_sha256': sha256(target)}
                manifest['pairs'].append(entry)
                completed[identity] = entry
                atomic_json(manifest_path, manifest)
                print(f'pair {len(manifest["pairs"])}: {split} / {source["name"]} / {index}', flush=True)
    for split in ('train', 'val'):
        if not any(p['split'] == split for p in manifest['pairs']):
            raise ValueError(f'{split}に2秒以上の音声がありません。入力データを増やしてください')
    if sha256(teacher) != signature['teacher_sha256'] or any(sha256(p) != s['sha256'] for p,s in zip(files,sources)):
        raise ValueError('準備中に元モデル/入力音声が変更されました。別の出力先でやり直してください')
    summary = {split: sum(p['samples'] for p in manifest['pairs'] if p['split']==split)/16000
               for split in ('train','val')}
    manifest['seconds'] = summary
    manifest['complete'] = True
    atomic_json(manifest_path, manifest)
    return manifest_path
