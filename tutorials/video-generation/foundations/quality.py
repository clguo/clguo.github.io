"""An independent MNIST classifier and transparent diagnostics, never generation guidance."""
import argparse
import gzip
import json
from pathlib import Path
import struct
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F



def dataset(split):
    Path('data').mkdir(exist_ok=True)
    prefix='train' if split=='train' else 't10k'
    arrays=[]
    for kind in ['images-idx3-ubyte','labels-idx1-ubyte']:
        name=f'{prefix}-{kind}.gz';path=Path('data')/name
        if not path.exists():urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/'+name,path)
        with gzip.open(path,'rb') as stream:
            magic,n=struct.unpack('>II',stream.read(8))
            if magic==2051:
                h,w=struct.unpack('>II',stream.read(8))
                array=np.frombuffer(stream.read(),dtype=np.uint8).copy().reshape(n,1,h,w)
                arrays.append(torch.from_numpy(array).float()/127.5-1)
            else:
                assert magic==2049
                arrays.append(torch.from_numpy(np.frombuffer(stream.read(),dtype=np.uint8).copy()).long())
    return arrays


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(1,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                                nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2))
        self.features=nn.Sequential(nn.Flatten(),nn.Linear(64*7*7,128),nn.ReLU())
        self.head=nn.Linear(128,10)

    def forward(self,x,features=False):
        h=self.features(self.conv(x))
        return (self.head(h),h) if features else self.head(h)


@torch.no_grad()
def classify(model,images):
    logits,features=[],[]
    for x in images.split(256):
        a,b=model(x,features=True)
        logits.append(a.cpu());features.append(b.cpu())
    return torch.cat(logits),torch.cat(features)


def feature_distance(a,b):
    """Fréchet distance in this classifier's 128D features; NOT Inception FID."""
    a,b=a.numpy().astype(np.float64),b.numpy().astype(np.float64)
    ca,cb=np.cov(a,rowvar=False),np.cov(b,rowvar=False)
    values,vectors=np.linalg.eigh(ca)
    square=(vectors*np.sqrt(np.maximum(values,0)))@vectors.T
    eigenvalues=np.linalg.eigvalsh(square@cb@square)
    return float(max(0,np.square(a.mean(0)-b.mean(0)).sum()+np.trace(ca)+np.trace(cb)-2*np.sqrt(np.maximum(eigenvalues,0)).sum()))


def train_classifier(out,device):
    torch.manual_seed(17)
    images,labels=dataset('train');images,labels=images.to(device),labels.to(device)
    model=Classifier().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(6):
        model.train()
        order=torch.randperm(50000,device=device)
        for indices in order.split(256):
            loss=F.cross_entropy(model(images[indices]),labels[indices])
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        model.eval()
        logits,_=classify(model,images[50000:])
        accuracy=(logits.argmax(1)==labels[50000:].cpu()).float().mean().item()
        print('Classifier epoch',epoch+1,'validation accuracy',accuracy,flush=True)
    test_images,test_labels=dataset('test')
    logits,_=classify(model,test_images.to(device))
    accuracy=(logits.argmax(1)==test_labels).float().mean().item()
    metadata={'test_accuracy':accuracy,'train_images':50000,'validation_images':10000,
              'test_images':10000,'epochs':6,'seed':17,'guidance_used':False,
              'architecture':'two-convolution MNIST classifier, 128D features'}
    out.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'metadata':metadata},out/'classifier.pt')
    (out/'classifier_metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Classifier final test accuracy',accuracy,flush=True)


@torch.no_grad()
def evaluate(root,out,device,count):
    from train import build_model, sample  # Legacy, unconditional diagnostics only.
    ckpt=torch.load(out/'classifier.pt',weights_only=True,map_location=device)
    classifier=Classifier().to(device).eval();classifier.load_state_dict(ckpt['model'])
    real,labels=dataset('test')
    real_logits,real_features=classify(classifier,real[:count].to(device))
    _,second_features=classify(classifier,real[count:2*count].to(device))
    report={'classifier':ckpt['metadata'],'feature_metric':'128D classifier-feature Fréchet distance, not Inception FID',
            'samples_per_configuration':count,'real_vs_real_feature_distance':feature_distance(real_features,second_features),
            'real_mean_confidence':real_logits.softmax(1).max(1).values.mean().item(),
            'warning':'Classifier confidence is not a digit-validity certificate. All samples are retained.',
            'configurations':{}}
    torch.manual_seed(2027)
    initial=torch.randn(count,1,28,28,device=device)
    for label,folder,method,stochastic in [('fm','mnist_fm','fm',False),('ddpm','mnist_ddpm','ddpm',True),
                                           ('ddim','mnist_ddpm','ddpm',False),('rf2','mnist_rf2','fm',False)]:
        path=root/folder/'checkpoint.pt'
        if not path.exists():continue
        checkpoint=torch.load(path,weights_only=True,map_location=device)
        config=checkpoint['config']
        model=build_model('mnist',config.get('architecture','legacy')).to(device).eval()
        model.load_state_dict(checkpoint['ema'])
        samples=[]
        torch.manual_seed(2028)
        for x in initial.split(64):
            states=sample(model,method,x,steps=100,stochastic=stochastic,
                          diffusion_steps=config.get('diffusion_steps',200),clip_denoised=config.get('clip_denoised',False))
            samples.append(states[-1])
        samples=torch.cat(samples)
        logits,features=classify(classifier,samples.to(device).clamp(-1,1))
        probabilities=logits.softmax(1);confidence=probabilities.max(1).values
        row={'mean_classifier_confidence':confidence.mean().item(),
             'fraction_confidence_above_0_95':(confidence>.95).float().mean().item(),
             'predicted_class_histogram':logits.argmax(1).bincount(minlength=10).tolist(),
             'classifier_feature_frechet_distance':feature_distance(features,real_features),
             'out_of_range_pixel_fraction':((samples< -1)|(samples>1)).float().mean().item(),
             'checkpoint':str(path),'architecture':config.get('architecture','legacy')}
        report['configurations'][label]=row
        print(label,row,flush=True)
        torch.save(samples,out/f'{root.name}_{label}_evaluation_samples.pt')
    (out/f'{root.name}_quality.json').write_text(json.dumps(report,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--train-classifier',action='store_true')
    parser.add_argument('--root',type=Path,default=Path('results/improved'))
    parser.add_argument('--out',type=Path,default=Path('results/quality'))
    parser.add_argument('--count',type=int,default=512)
    args=parser.parse_args();torch.set_num_threads(4)
    if not torch.cuda.is_available():raise RuntimeError('Use a GPU allocation.')
    if args.train_classifier:train_classifier(args.out,'cuda')
    else:evaluate(args.root,args.out,'cuda',args.count)


if __name__=='__main__':main()
