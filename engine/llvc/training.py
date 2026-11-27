"""Experimental single-target LLVC fine-tuning from paired RVC teacher audio.

Keeps the public LLVC architecture/latency. Uses MR-STFT + log-mel,
least-squares GAN and discriminator feature matching. This is NOT an exact
reproduction of upstream training (no HuBERT perceptual loss).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F

from . import assets
from .distill import atomic_json, sha256
from .discriminators import MultiPeriodDiscriminator, discriminator_loss, generator_loss, feature_loss
from .model import Net


def atomic_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            name = Path(temporary.name)
        torch.save(value, name)
        os.replace(name, path)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def model_params():
    return json.loads(Path(__file__).with_name('config.json').read_text())['model_params']


def load_pair(root, entry):
    arrays = []
    root = Path(root).resolve()
    for kind in ('source', 'target'):
        path = (root/entry[kind]).resolve()
        if root not in path.parents:
            raise ValueError('音声ペアのパスがデータセットの外を参照しています')
        audio, sr = sf.read(path, dtype='float32', always_2d=True)
        if sr != 16000 or audio.shape[1] != 1 or not np.isfinite(audio).all():
            raise ValueError('音声ペアは有限値の16kHz monoである必要があります')
        if sha256(path) != entry[kind+'_sha256']:
            raise ValueError('音声ペアがmanifest作成時から変更されています')
        arrays.append(audio[:,0])
    if len(arrays[0]) != len(arrays[1]) or len(arrays[0]) != entry['samples']:
        raise ValueError('元音声と教師音声の長さが一致しません')
    return arrays


class PairedCrops:
    def __init__(self, root, manifest, split, context, segment):
        self.root = Path(root)
        self.entries = [entry for entry in manifest['pairs'] if entry['split'] == split]
        self.context, self.segment = context, segment
        self.length = context+segment
        if not self.entries or any(e['samples'] < self.length for e in self.entries):
            raise ValueError(f'{split}の音声長が不足しています。context+segment以上の音声を用意してください')
        # Verify pairs once; detect later changes with file identity per crop.
        self.identities = {}
        for entry in self.entries:
            load_pair(self.root, entry)
            for kind in ("source", "target"):
                path = self.root/entry[kind]
                stat = path.stat()
                self.identities[entry[kind]] = (stat.st_size, stat.st_mtime_ns)

    def crop(self, index, rng=None):
        entry = self.entries[index]
        start = 0 if rng is None else int(torch.randint((entry['samples']-self.length)//208+1, (), generator=rng))*208
        values = []
        for kind in ('source', 'target'):
            stat = (self.root/entry[kind]).stat()
            if (stat.st_size, stat.st_mtime_ns) != self.identities[entry[kind]]:
                raise ValueError('学習中に音声ペアが変更されました')
            audio, sr = sf.read(self.root/entry[kind], start=start, frames=self.length, dtype='float32')
            if sr != 16000 or audio.ndim != 1 or len(audio) != self.length or not np.isfinite(audio).all():
                raise ValueError('学習中に音声ペアが変更/破損しました')
            values.append(torch.from_numpy(audio.copy())[None, :])
        return values

    def batch(self, count, rng):
        examples = [self.crop(int(torch.randint(len(self.entries), (), generator=rng)), rng) for _ in range(count)]
        return tuple(torch.stack([e[i] for e in examples]) for i in (0,1))


class SpectralLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sizes = (256, 512, 1024)
        for size in self.sizes:
            self.register_buffer(f'window_{size}', torch.hann_window(size))
        mel = torch.linspace(2595*np.log10(1+40/700), 2595*np.log10(1+7600/700), 82)
        hz = 700*(10**(mel/2595)-1)
        frequencies = torch.linspace(0,8000,513)[None,:]
        rising = (frequencies-hz[:-2,None])/(hz[1:-1,None]-hz[:-2,None])
        falling = (hz[2:,None]-frequencies)/(hz[2:,None]-hz[1:-1,None])
        self.register_buffer('mel', torch.minimum(rising, falling).clamp_min(0))

    def magnitude(self, audio, size):
        return torch.stft(audio.float().squeeze(1), size, hop_length=size//4,
                          window=getattr(self, f'window_{size}'), return_complex=True).abs().clamp_min(1e-5)

    def forward(self, generated, target):
        total = generated.new_zeros((), dtype=torch.float32)
        for size in self.sizes:
            pred, real = self.magnitude(generated, size), self.magnitude(target, size)
            convergence = torch.linalg.vector_norm(pred-real)/torch.linalg.vector_norm(real).clamp_min(1e-5)
            total = total+convergence+F.l1_loss(pred.log(), real.log())
        pred, real = self.magnitude(generated,1024), self.magnitude(target,1024)
        logmel = F.l1_loss((self.mel@pred).clamp_min(1e-5).log(), (self.mel@real).clamp_min(1e-5).log())
        return total/len(self.sizes)+logmel


def freeze_content(model, freeze=True):
    for module in (model.convnet_pre, model.in_conv, model.mask_gen.encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(not freeze)
        if freeze:
            module.eval()


def train_step(model, discriminator, opt_g, opt_d, spectral, batch, context, scaler, amp=False):
    source, teacher = batch
    for parameter in discriminator.parameters():
        parameter.requires_grad_(True)
    opt_g.zero_grad(set_to_none=True)
    opt_d.zero_grad(set_to_none=True)
    with torch.autocast(device_type=source.device.type, dtype=torch.float16, enabled=amp):
        generated = model(source)
    if generated.shape != teacher.shape:
        raise ValueError('学習モデルの出力長が教師音声と一致しません')
    generated, teacher = generated[...,context:].float(), teacher[...,context:].float()
    with torch.autocast(device_type=source.device.type, dtype=torch.float16, enabled=amp):
        real_score, fake_score, _, _ = discriminator(teacher, generated.detach())
        loss_d, _, _ = discriminator_loss(real_score, fake_score)
    if not torch.isfinite(loss_d):
        raise ValueError('Discriminatorのlossが非有限値になりました')
    scaler.scale(loss_d).backward()
    scaler.unscale_(opt_d)
    torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 10., error_if_nonfinite=not amp)
    scaler.step(opt_d)
    for parameter in discriminator.parameters():
        parameter.requires_grad_(False)
    with torch.autocast(device_type=source.device.type, dtype=torch.float16, enabled=amp):
        _, fake_score, real_features, fake_features = discriminator(teacher, generated)
        adversarial, _ = generator_loss(fake_score)
        matching = feature_loss(real_features, fake_features)
    reconstruction = spectral(generated, teacher)
    loss_g = 10*reconstruction+2*matching+adversarial
    if not torch.isfinite(loss_g):
        raise ValueError('Generatorのlossが非有限値になりました')
    scaler.scale(loss_g).backward()
    scaler.unscale_(opt_g)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=not amp)
    scaler.step(opt_g)
    scaler.update()
    return {name: float(value.detach().cpu()) for name,value in
            {'generator':loss_g,'discriminator':loss_d,'spectral':reconstruction,'feature':matching,'adversarial':adversarial}.items()}


@torch.inference_mode()
def validate(model, crops, spectral, device, preview_dir=None):
    previous_modes = [(module,module.training) for module in model.modules()]
    model.eval()
    scores = []
    try:
        for index in range(min(len(crops.entries),8)):
            source, teacher = (x[None].to(device) for x in crops.crop(index))
            generated = model(source)
            score = spectral(generated[...,crops.context:], teacher[...,crops.context:])
            if not torch.isfinite(score):
                raise ValueError('検証lossが非有限値になりました')
            scores.append(float(score.cpu()))
            if preview_dir is not None and index < 3:
                preview_dir.mkdir(parents=True, exist_ok=True)
                original,target = load_pair(crops.root,crops.entries[index])
                length = max(crops.length,min(len(original),64000))
                original = torch.from_numpy(original[:length].copy())[None,None].to(device)
                target = torch.from_numpy(target[:length].copy())[None,None].to(device)
                preview = model(original)
                if not torch.isfinite(preview).all():
                    raise ValueError('試聴用の出力に非有限値があります')
                for label,audio in [('source',original),('teacher',target),('student',preview)]:
                    sf.write(preview_dir/f'{index}-{label}.wav', audio[0,0,crops.context:].cpu().numpy(), 16000, subtype='FLOAT')
    finally:
        for module,mode in previous_modes:
            module.training = mode
    return sum(scores)/len(scores)


def fine_tune(dataset, run_dir, *, base=None, device='cuda', steps=10000, batch_size=1,
              segment=8192, context=8192, lr=1e-4, save_every=100, eval_every=250,
              freeze_encoder=True, seed=1234, amp=False, resume=False):
    dataset, run_dir = Path(dataset).resolve(), Path(run_dir).resolve()
    base = Path(base or assets.CHECKPOINT).resolve()
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('追加学習にはCUDA版PyTorchとGPUが必要です。CPUは--device cpuで小規模動作確認に使えます')
    if amp and device != 'cuda':
        raise ValueError('--ampはCUDA学習で使ってください')
    if steps < 1 or batch_size < 1 or save_every < 1 or eval_every < 1 or not 0 < lr <= .01:
        raise ValueError('学習回数・batch・保存間隔・学習率が不正です')
    if segment < 1024 or context < 0 or segment % 16 or context % 16:
        raise ValueError('segmentは1024以上、contextは0以上で、それぞれ16の倍数にしてください')
    if not base.is_file():
        raise ValueError('学習済みLLVC初期モデルがありません。先に配布モデルを取得してください')
    manifest_path = dataset/'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if not manifest.get('complete'):
        raise ValueError('教師音声の作成が完了していません')
    config = {'version':1, 'dataset_sha256':sha256(manifest_path), 'base_sha256':sha256(base),
              'model_params':model_params(),'batch_size':batch_size,'segment':segment,'context':context,
              'lr':lr,'freeze_encoder':freeze_encoder,'seed':seed,'amp':amp,'periods':[2,3,5]}
    config_path, latest = run_dir/'config.json', run_dir/'latest.pth'
    if run_dir.exists() and any(run_dir.iterdir()):
        if not resume or not latest.is_file() or not config_path.is_file():
            raise ValueError('学習出力先が空ではありません。再開する場合は--resumeを使ってください')
        if json.loads(config_path.read_text()) != config:
            raise ValueError('再開時の学習設定/データ/初期モデルが異なります')
    train_data = PairedCrops(dataset,manifest,'train',context,segment)
    val_data = PairedCrops(dataset,manifest,'val',context,segment)
    seconds = {split: sum(p['samples'] for p in manifest['pairs'] if p['split']==split)/16000
               for split in ('train','val')}
    print(f"データ量: train {seconds['train']:.1f}秒 / val {seconds['val']:.1f}秒", flush=True)
    if seconds['train'] < 600 or seconds['val'] < 60:
        print('小規模データです。動作確認と音質評価を分け、短いデータでの過学習に注意してください。', flush=True)
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = Net(**config['model_params'])
    initial = torch.load(base, map_location='cpu', weights_only=True)
    model.load_state_dict(initial['model'], strict=True)
    del initial
    model.train().to(device)
    freeze_content(model,freeze_encoder)
    discriminator = MultiPeriodDiscriminator(periods=config['periods']).to(device).train()
    spectral = SpectralLoss().to(device)
    opt_g = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr, betas=(.8,.99))
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr, betas=(.8,.99))
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    step, best = 0, None
    if resume:
        saved = torch.load(latest, map_location=device, weights_only=True)
        if saved['config'] != config:
            raise ValueError('再開checkpointの設定が一致しません')
        model.load_state_dict(saved['model'], strict=True)
        discriminator.load_state_dict(saved['discriminator'], strict=True)
        opt_g.load_state_dict(saved['optimizer_g'])
        opt_d.load_state_dict(saved['optimizer_d'])
        scaler.load_state_dict(saved['scaler'])
        rng.set_state(saved['crop_rng'].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in saved['cuda_rng']])
        step, best = saved['step'], saved['best_validation']
        del saved
    if steps <= step:
        raise ValueError('--stepsは保存済みstepより大きくしてください（追加回数ではなく総step数）')
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(config_path,config)
    metadata = {'teacher':{k:v for k,v in manifest['config'].items() if k != 'sources'},
                'dataset_sha256':config['dataset_sha256'],'experimental':True,'model_params':config['model_params']}

    def weights(validation=None):
        return {'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'model_params':config['model_params'],'step':step,'validation':validation,'metadata':metadata}

    def save_latest():
        atomic_save(latest, {**weights(), 'config':config,'discriminator':discriminator.state_dict(),
                    'optimizer_g':opt_g.state_dict(),'optimizer_d':opt_d.state_dict(),'scaler':scaler.state_dict(),
                    'crop_rng':rng.get_state(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if device == 'cuda' else [],'best_validation':best})

    # Establish an unmodified-base validation reference and a resumable step 0.
    if step == 0:
        baseline = validate(model,val_data,spectral,device,run_dir/'previews'/'base')
        atomic_json(run_dir/'baseline.json',{'validation_spectral':baseline,'step':0})
        save_latest()
    with (run_dir/'metrics.jsonl').open('a', encoding='utf-8') as log:
        while step < steps:
            batch = tuple(x.to(device) for x in train_data.batch(batch_size,rng))
            metrics = train_step(model,discriminator,opt_g,opt_d,spectral,batch,context,scaler,amp)
            step += 1
            metrics['step'] = step
            if step % eval_every == 0 or step == steps:
                score = validate(model,val_data,spectral,device,run_dir/'previews'/f'step-{step}')
                metrics['validation_spectral'] = score
                if best is None or score < best:
                    best = score
                    atomic_save(run_dir/'best.pth', weights(score))
            log.write(json.dumps(metrics, allow_nan=False)+'\n')
            log.flush()
            print(json.dumps(metrics, allow_nan=False), flush=True)
            if step % save_every == 0 or step == steps:
                save_latest()
    return run_dir/'best.pth'


def export_voice(checkpoint, voice_id, name):
    # Export is explicit: audition previews first; never overwrite another voice.
    import re
    import shutil
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}', voice_id) or not 1 <= len(name) <= 80:
        raise ValueError('声IDは英数字/_/-で64文字以内、表示名は1〜80文字にしてください')
    payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if payload.get('step',0) < 1 or payload.get('model_params') != model_params() or 'metadata' not in payload:
        raise ValueError('この手順で追加学習したLLVC checkpointを指定してください')
    # Verify inference-compatible weights; do not export optimizer/discriminator.
    model = Net(**model_params())
    model.load_state_dict(payload['model'],strict=True)
    if any(not torch.isfinite(v).all() for v in model.state_dict().values()):
        raise ValueError('checkpointに非有限値があります')
    root = assets.MODEL_DIR/'voices'
    root.mkdir(parents=True,exist_ok=True)
    destination = root/voice_id
    if destination.exists():
        raise FileExistsError('この声IDは既にあります。別のIDで書き出してください')
    temporary = Path(tempfile.mkdtemp(prefix='.export-',dir=root))
    try:
        atomic_save(temporary/'model.pth', {k:payload[k] for k in ('model','model_params','step','validation','metadata')})
        atomic_json(temporary/'voice.json', {'id':'llvc:'+voice_id,'name':name,'step':payload['step'],
                    'validation':payload.get('validation'),'teacher_name':payload['metadata']['teacher']['teacher_name'],
                    'experimental':True})
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return destination
