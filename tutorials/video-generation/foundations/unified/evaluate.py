"""Matched-NFE, class-conditional evaluation with counted network calls and no filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch import nn

from digit_model import DigitUNet
from quality import Classifier,classify,dataset,feature_distance
from .processes import generate,constant_velocity_defect


class Counted(nn.Module):
    def __init__(self,model):super().__init__();self.model=model;self.calls=0
    def forward(self,*args):
        self.calls+=1
        return self.model(*args)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def within_class_variance(values,labels):
    flat=values.flatten(1).double()
    return torch.stack([(flat[labels==label]-flat[labels==label].mean(0)).square().sum(1).mean() for label in range(10)]).mean().item()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('results/unified'))
    args=parser.parse_args();root=args.root
    torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Evaluate inside a GPU allocation.')
    device=torch.device('cuda')
    cpath=Path('results/quality/classifier.pt')
    ckpt=torch.load(cpath,weights_only=True,map_location=device)
    classifier=Classifier().to(device).eval();classifier.load_state_dict(ckpt['model'])
    test_images,test_labels=dataset('test')
    reference=torch.cat([test_images[test_labels==label][:100] for label in range(10)])
    second=torch.cat([test_images[test_labels==label][100:200] for label in range(10)])
    real_logits,real_features=classify(classifier,reference.to(device))
    _,second_features=classify(classifier,second.to(device))
    y=torch.arange(10).repeat_interleave(100)
    initial=torch.randn(1000,1,28,28,generator=torch.Generator().manual_seed(2026))
    visual=torch.cat([torch.arange(10)+100*label for label in range(10)])
    out=root/'evaluation';out.mkdir(exist_ok=True)
    torch.save({'noise':initial,'labels':y,'visual_indices':visual},out/'input_bank.pt')
    report={'protocol':json.loads(Path('unified/protocol.json').read_text()),
            'hardware':torch.cuda.get_device_name(),'classifier':ckpt['metadata'],
            'timing_note':'Shared GPU; training times include retained segments, not queue waiting or discarded post-checkpoint work',
            'classifier_sha256':sha(cpath),'input_bank_sha256':sha(out/'input_bank.pt'),
            'reference':'first 100 official test images per true class; no classifier-based selection',
            'real_reference_class_agreement':(real_logits.argmax(1)==y).float().mean().item(),
            'real_vs_real_feature_distance':feature_distance(real_features,second_features),
            'real_within_class_feature_variance':within_class_variance(real_features,y),
            'real_within_class_pixel_variance':within_class_variance(reference,y),
            'feature_metric':'128D independent-classifier feature Frechet distance; not Inception FID',
            'configurations':{},'checkpoints':{}}
    configurations=[('ddpm','diffusion'),('ddim','diffusion'),('fm','flow'),('rf2','reflow'),('continued','continued')]
    endpoints={}
    for name,stage in configurations:
        path=root/stage/'checkpoint.pt'
        checkpoint=torch.load(path,weights_only=True,map_location=device)
        if checkpoint['config']['steps'] != checkpoint['config']['planned_steps']:
            raise RuntimeError(f'Unfinished training stage: {stage}')
        model=DigitUNet(num_classes=10).to(device).eval();model.load_state_dict(checkpoint['ema'])
        counted=Counted(model)
        report['checkpoints'][stage]={'sha256':sha(path),'metadata':checkpoint['config'],
            'random_stream_state_sha256':{key:hashlib.sha256(checkpoint[key].cpu().numpy().tobytes()).hexdigest()
                for key in ['data_rng','noise_rng','time_rng']}}
        report['configurations'][name]={}
        endpoints[name]={}
        budgets=[1,4,16,64]+([1000] if name=='ddpm' else [])
        for nfe in budgets:
            # A small, untimed warmup; reset the actual counter and stochastic stream afterwards.
            generate(model,initial[:2].to(device),y[:2].to(device),name,min(nfe,4))
            rng=torch.Generator(device=device).manual_seed(9000+nfe)
            counted.calls=0
            outputs=[]
            torch.cuda.synchronize();started=time.perf_counter()
            for offset in range(0,len(initial),100):
                result=generate(counted,initial[offset:offset+100].to(device),y[offset:offset+100].to(device),
                                name,nfe,generator=rng,keep_trajectory=False)
                outputs.append(result.cpu())
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            output=torch.cat(outputs)
            assert torch.isfinite(output).all()
            assert counted.calls==10*nfe
            logits,features=classify(classifier,output.to(device).clamp(-1,1))
            predictions=logits.argmax(1)
            confidence=logits.softmax(1).max(1).values
            row={'nfe':nfe,'counted_network_calls':counted.calls,'batch_size':100,'sample_count':1000,
                 'requested_class_agreement':(predictions==y).float().mean().item(),
                 'per_class_agreement':[(predictions[y==label]==label).float().mean().item() for label in range(10)],
                 'mean_classifier_confidence':confidence.mean().item(),
                 'fraction_confidence_above_0_95':(confidence>.95).float().mean().item(),
                 'classifier_feature_frechet_distance':feature_distance(features,real_features),
                 'within_class_feature_variance_ratio':within_class_variance(features,y)/report['real_within_class_feature_variance'],
                 'within_class_pixel_variance_ratio':within_class_variance(output.clamp(-1,1),y)/report['real_within_class_pixel_variance'],
                 'predicted_class_histogram':predictions.bincount(minlength=10).tolist(),
                 'sampler_seconds_including_output_transfer':elapsed,
                 'seconds_per_image':elapsed/1000,'samples_per_second':1000/elapsed}
            report['configurations'][name][str(nfe)]=row
            endpoints[name][str(nfe)]=output
            if nfe==64:
                # Untimed replay records animations without charging their host-copy
                # overhead to the main sampler timing. Keep the same noise stream.
                rng=torch.Generator(device=device).manual_seed(9000+nfe)
                visual_states=[]
                for offset in range(0,len(initial),100):
                    states=generate(model,initial[offset:offset+100].to(device),y[offset:offset+100].to(device),
                                    name,nfe,generator=rng,keep_trajectory=True)
                    assert torch.allclose(states[-1],output[offset:offset+100],atol=1e-6,rtol=1e-6)
                    visual_states.append(states[:,:10])
                trajectory=torch.cat(visual_states,dim=1)
                assert torch.equal(trajectory[0],initial[visual])
                assert torch.allclose(trajectory[-1],output[visual],atol=1e-6,rtol=1e-6)
                torch.save(trajectory,out/f'{name}_trajectory.pt')
            print(name,nfe,json.dumps(row),flush=True)
        if name in ['fm','rf2','continued']:
            dense=generate(model,initial[visual].to(device),y[visual].to(device),name,128,keep_trajectory=True)
            report['configurations'][name]['dense_path']={'nfe':128,'constant_velocity_defect':constant_velocity_defect(dense)}
            torch.save(dense,out/f'{name}_dense_trajectory.pt')
        torch.save(endpoints[name],out/f'{name}_endpoints.pt')
    for key in ['rf2','continued']:
        assert report['checkpoints'][key if key=='continued' else 'reflow']['metadata']['initial_weights_sha256']==report['checkpoints']['reflow']['metadata']['initial_weights_sha256']
    diffusion=report['checkpoints']['diffusion']['metadata'];flow=report['checkpoints']['flow']['metadata']
    assert diffusion['initial_weights_sha256']==flow['initial_weights_sha256']
    assert diffusion['class_presentations']==flow['class_presentations']
    assert report['checkpoints']['continued']['metadata']['class_presentations']==report['checkpoints']['reflow']['metadata']['class_presentations']
    report['pair_generation']=json.loads((root/'pairs/metadata.json').read_text())
    assert report['pair_generation']['teacher_sha256']==report['checkpoints']['flow']['sha256']
    assert report['checkpoints']['reflow']['metadata']['teacher_sha256']==report['checkpoints']['flow']['sha256']
    streams={key:row['random_stream_state_sha256'] for key,row in report['checkpoints'].items()}
    assert streams['diffusion']==streams['flow']
    assert all(streams['reflow'][key]==streams['continued'][key] for key in ['data_rng','time_rng'])
    assert torch.equal(endpoints['ddpm']['1'],endpoints['ddim']['1'])
    report['audit_checks']={'base_initial_weights_identical':True,'base_class_streams_identical':True,
        'followup_initial_weights_identical':True,'followup_class_streams_identical':True,
        'all_visual_samples_included_in_metrics':True,'ddpm_ddim_share_checkpoint':True,
        'teacher_and_pair_parent_hashes_match':True,'shared_random_stream_final_states_match':True,
        'one_step_diffusion_endpoints_identical':True}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('Completed controlled evaluation',flush=True)


if __name__=='__main__':main()
