"""Small CPU training/teacher fixtures verify mechanics, NOT a learned voice."""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from engine.llvc import assets
from engine.llvc.distill import prepare_pairs
from engine.llvc.model import Net
from engine.llvc.streaming import load_stream
from engine.llvc.training import fine_tune, export_voice, model_params, load_pair


class TeacherFixture:
    """Test-only teacher, no production substitution for actual RVC."""
    info = SimpleNamespace(speakers=1)
    def __init__(self,*args,**kwargs): pass
    def convert_array(self,audio,options):
        self.assert_options = options
        return audio*.5,None


class SmallDiscriminator(torch.nn.Module):
    """Test-only small GAN; generator uses the full official architecture."""
    def __init__(self,**kwargs):
        super().__init__()
        self.conv = torch.nn.Conv1d(1,4,7,padding=3)
        self.out = torch.nn.Conv1d(4,1,3,padding=1)
    def forward(self,real,fake):
        a,b = torch.nn.functional.leaky_relu(self.conv(real),.1),torch.nn.functional.leaky_relu(self.conv(fake),.1)
        return [self.out(a).flatten(1)],[self.out(b).flatten(1)],[[a]],[[b]]


def make_dataset(root):
    source = root/'audio'
    source.mkdir()
    for index in range(2):
        audio = (.05*np.sin(np.arange(40013)*(index+1)*.08)).astype(np.float32)
        sf.write(source/f'{index}.wav',audio,16000,subtype='FLOAT')
    teacher = root/'teacher.pth'
    teacher.write_bytes(b'fixture not a real RVC model')
    output = root/'pairs'
    with patch('engine.llvc.distill.VoiceConverter',TeacherFixture),redirect_stdout(io.StringIO()):
        prepare_pairs(teacher,source,output,device='cpu',pitch=5)
    return teacher,source,output


class TestLLVCDataset(unittest.TestCase):
    def test_pairs_split_alignment_and_resume_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            teacher,source,output=make_dataset(root)
            manifest=json.loads((output/'manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual({p['split'] for p in manifest['pairs']},{'train','val'})
            self.assertEqual(manifest['config']['pitch'],5)
            for pair in manifest['pairs']:
                original,target=load_pair(output,pair)
                self.assertEqual(original.size%160,0)
                np.testing.assert_allclose(target,original*.5,atol=1e-7)
            with patch('engine.llvc.distill.VoiceConverter',TeacherFixture):
                prepare_pairs(teacher,source,output,device='cpu',pitch=5,resume=True)
                self.assertEqual(len(json.loads((output/'manifest.json').read_text())['pairs']),2)
                with self.assertRaises(ValueError):
                    prepare_pairs(teacher,source,output,device='cpu',pitch=6,resume=True)
                pair=manifest['pairs'][0]
                sf.write(output/pair['target'],np.zeros(pair['samples'],dtype=np.float32),16000)
                with self.assertRaises(ValueError):
                    prepare_pairs(teacher,source,output,device='cpu',pitch=5,resume=True)

    def test_invalid_teacher_length_never_becomes_completed_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);teacher,source,_=make_dataset(root)
            bad=TeacherFixture()
            bad.convert_array=lambda audio,options:(audio[:-1],None)
            output=root/'invalid'
            with patch('engine.llvc.distill.VoiceConverter',return_value=bad):
                with self.assertRaisesRegex(ValueError,'出力長'):
                    prepare_pairs(teacher,source,output,device='cpu')
            self.assertFalse(json.loads((output/'manifest.json').read_text())['complete'])


class TestLLVCFineTune(unittest.TestCase):
    def test_train_resume_export_and_select_full_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            _,_,dataset=make_dataset(root)
            torch.manual_seed(22)
            model=Net(**model_params())
            base=root/'base.pth'
            torch.save({'model':model.state_dict()},base)
            first,second=root/'resume',root/'whole'
            options=dict(base=base,device='cpu',batch_size=1,segment=1024,context=0,
                         save_every=1,eval_every=1,seed=24)
            with patch('engine.llvc.training.MultiPeriodDiscriminator',SmallDiscriminator),redirect_stdout(io.StringIO()):
                fine_tune(dataset,first,steps=2,**options)
                fine_tune(dataset,first,steps=3,resume=True,**options)
                fine_tune(dataset,second,steps=3,**options)
            resumed=torch.load(first/'latest.pth',map_location='cpu',weights_only=True)
            whole=torch.load(second/'latest.pth',map_location='cpu',weights_only=True)
            self.assertEqual(resumed['step'],3)
            for key,value in resumed['model'].items():
                torch.testing.assert_close(value,whole['model'][key],atol=0,rtol=0)
            torch.testing.assert_close(resumed['model']['in_conv.0.weight'],model.state_dict()['in_conv.0.weight'],atol=0,rtol=0)
            self.assertFalse(torch.equal(resumed['model']['out_conv.0.weight'],model.state_dict()['out_conv.0.weight']))
            self.assertTrue((first/'previews'/'step-3'/'0-student.wav').is_file())
            self.assertTrue((first/'baseline.json').is_file())
            with patch.object(assets,'MODEL_DIR',root/'models'),patch.object(assets,'CHECKPOINT',base):
                destination=export_voice(first/'best.pth','my-voice','My voice')
                self.assertEqual(assets.resolve_voice('llvc:my-voice'),destination/'model.pth')
                self.assertIn('llvc:my-voice',[v['id'] for v in assets.list_voices()])
                with self.assertRaises(FileExistsError):
                    export_voice(first/'best.pth','my-voice','My voice')
                with patch.object(assets,'CHECKPOINT',root/'missing-default.pth'):
                    self.assertFalse(assets.model_status()['installed'])
                    stream=load_stream('cpu',{},model_id='llvc:my-voice')
                samples=np.zeros(320,dtype=np.float32)
                result=np.concatenate([stream.push(samples),stream.flush()])
                self.assertEqual(len(result),320)
                self.assertEqual(stream.runtime_name,'ONNX Runtime CPU / FP32')
                self.assertTrue(np.isfinite(result).all())

    def test_rvc_catalog_excludes_llvc_weights(self):
        from engine.web import server
        from fastapi import HTTPException
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'llvc').mkdir()
            (root/'rvc.pth').write_bytes(b'fixture')
            (root/'llvc'/'G_500000.pth').write_bytes(b'fixture')
            with patch.object(server,'MODELS_DIR',root):
                self.assertEqual([m['id'] for m in server.list_models()],['rvc'])
                with self.assertRaises(HTTPException):
                    server.model_path('llvc/G_500000')

    def test_default_context_step_has_finite_gradients(self):
        from engine.llvc.training import SpectralLoss, freeze_content, train_step
        model=Net(**model_params()).train()
        freeze_content(model)
        discriminator=SmallDiscriminator()
        opt_g=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
        opt_d=torch.optim.AdamW(discriminator.parameters(),lr=1e-4)
        source=torch.randn(1,1,16384)*.05
        metrics=train_step(model,discriminator,opt_g,opt_d,SpectralLoss(),(source,source*.5),
                           8192,torch.amp.GradScaler('cuda',enabled=False))
        self.assertTrue(all(np.isfinite(v) for v in metrics.values()))

    def test_voice_paths_cannot_escape_catalog(self):
        for identity in ('llvc:../bad','llvc:a/b','llvc:','other'):
            with self.assertRaises(ValueError):
                assets.resolve_voice(identity)

    def test_official_discriminator_and_spectral_losses_backpropagate(self):
        from engine.llvc.training import SpectralLoss
        from engine.llvc.discriminators import MultiPeriodDiscriminator,discriminator_loss,feature_loss,generator_loss
        torch.manual_seed(2)
        discriminator=MultiPeriodDiscriminator(periods=[2])
        real=torch.randn(1,1,1024)*.05
        generated=(torch.randn(1,1,1024)*.05).requires_grad_()
        r,f,real_features,fake_features=discriminator(real,generated)
        loss=discriminator_loss(r,f)[0]+generator_loss(f)[0]+feature_loss(real_features,fake_features)+SpectralLoss()(generated,real)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(generated.grad).all())
        self.assertTrue(any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in discriminator.parameters()))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA fine-tuning/AMP unavailable')
class TestLLVCFineTuneCUDA(unittest.TestCase):
    def test_full_generator_adversarial_step_with_amp(self):
        from engine.llvc.training import SpectralLoss, freeze_content, train_step
        model=Net(**model_params()).train().to('cuda')
        freeze_content(model)
        discriminator=SmallDiscriminator().to('cuda')
        opt_g=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-4)
        opt_d=torch.optim.AdamW(discriminator.parameters(),lr=1e-4)
        source=torch.randn(1,1,1024,device='cuda')*.05
        result=train_step(model,discriminator,opt_g,opt_d,SpectralLoss().to('cuda'),
                          (source,source*.5),0,torch.amp.GradScaler('cuda',enabled=True),amp=True)
        self.assertTrue(all(np.isfinite(v) for v in result.values()))
