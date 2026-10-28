"""Common mathematical kernels. No repository-specific architectures or pretrained weights."""
import math
import torch


def cosine_schedule(steps=1000,device='cpu'):
    grid=torch.linspace(0,1,steps+1,device=device,dtype=torch.float64)
    values=torch.cos((grid+.008)/1.008*math.pi/2).square()
    betas=(1-values[1:]/values[:-1]).clamp(0,.999).float()
    return (1-betas).cumprod(0)


def broadcast(t,x):return t.reshape(-1,*([1]*(x.ndim-1)))


def training_pair(data,noise,time,family,abar=None):
    if family=='flow':
        tb=broadcast(time,data)
        return (1-tb)*noise+tb*data,data-noise,time
    if family!='diffusion':raise ValueError(family)
    indices=(time*len(abar)).long().clamp_max(len(abar)-1)
    a=broadcast(abar[indices],data)
    return a.sqrt()*data+(1-a).sqrt()*noise,noise,(indices+1).float()/len(abar)


@torch.no_grad()
def generate(model,initial,labels,method,nfe,*,generator=None,keep_trajectory=False):
    """Exactly nfe model calls; DDPM is posterior sampling on a respaced cosine chain.

    nfe=1000 gives the original adjacent DDPM schedule. nfe<1000 uses its
    respaced posterior. DDIM uses that same index grid with eta=0.
    Both diffusion samplers clip x_clean, then recompute consistent epsilon.
    FM/RF2 use unclipped Euler integration. CFG is deliberately disabled.
    """
    if nfe<1 or nfe>1000:raise ValueError('NFE must be between 1 and 1000.')
    x=initial.clone()
    states=[x.cpu()] if keep_trajectory else None
    if method in ['fm','rf2','continued']:
        for k in range(nfe):
            time=torch.full((len(x),),k/nfe,device=x.device)
            x=x+model(x,time,labels)/nfe
            if states is not None:states.append(x.cpu())
    elif method in ['ddpm','ddim']:
        abar=cosine_schedule(1000,x.device)
        indices=torch.linspace(999,0,nfe,device=x.device).round().long().tolist()
        for j,index in enumerate(indices):
            a=abar[index]
            previous=abar[indices[j+1]] if j+1<len(indices) else torch.ones((),device=x.device)
            time=torch.full((len(x),),(index+1)/1000,device=x.device)
            prediction=model(x,time,labels)
            clean=((x-(1-a).sqrt()*prediction)/a.sqrt()).clamp(-1,1)
            # Keep the same clean-image clipping rule in DDPM and DDIM.
            epsilon=(x-a.sqrt()*clean)/(1-a).sqrt()
            sigma=((1-previous)/(1-a)*(1-a/previous)).clamp_min(0).sqrt() if method=='ddpm' else torch.zeros((),device=x.device)
            x=previous.sqrt()*clean+(1-previous-sigma.square()).clamp_min(0).sqrt()*epsilon
            if method=='ddpm' and j+1<len(indices):
                x=x+sigma*torch.randn(x.shape,device=x.device,generator=generator)
            if states is not None:states.append(x.cpu())
    else:raise ValueError(method)
    return torch.stack(states) if states is not None else x


def constant_velocity_defect(states):
    states=states.flatten(2).double()
    displacement=states[-1]-states[0]
    velocities=(len(states)-1)*(states[1:]-states[:-1])
    return float((velocities-displacement).square().sum(-1).mean()/displacement.square().sum(-1).mean().clamp_min(1e-12))
