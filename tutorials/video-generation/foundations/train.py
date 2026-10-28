"""Readable flow matching and DDPM experiments; all comments intentionally English."""
import argparse
import copy
import gzip
import json
import math
from pathlib import Path
import struct
import time
import urllib.request

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.register_buffer('frequencies', torch.exp(torch.linspace(0, math.log(1000), width // 2)))

    def forward(self, t):
        phase = t[:, None] * self.frequencies[None]  # [B, 32]
        return torch.cat((phase.sin(), phase.cos()), dim=-1)


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time = TimeEmbedding()
        self.net = nn.Sequential(nn.Linear(66, 128), nn.SiLU(), nn.Linear(128, 128),
                                 nn.SiLU(), nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 2))

    def forward(self, x, t):
        return self.net(torch.cat((x, self.time(t)), dim=-1))


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.time = nn.Linear(128, cout)
        self.norm2 = nn.GroupNorm(8, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time(t)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return (h + self.skip(x)) / math.sqrt(2)


class ImageNet(nn.Module):
    """Small U-Net, deliberately simpler to read than a DiT. Input/output [B,1,28,28]."""
    def __init__(self):
        super().__init__()
        self.time = nn.Sequential(TimeEmbedding(), nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
        self.input = nn.Conv2d(1, 32, 3, padding=1)
        self.enc = ResBlock(32, 32)
        self.down = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.mid1 = ResBlock(64, 64)
        self.mid2 = ResBlock(64, 64)
        self.up = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.dec = ResBlock(64, 32)
        self.output = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv2d(32, 1, 3, padding=1))

    def forward(self, x, t):
        t = self.time(t)
        skip = self.enc(self.input(x), t)
        h = self.mid2(self.mid1(self.down(skip), t), t)
        return self.output(self.dec(torch.cat((self.up(h), skip), dim=1), t))


def toy_data(n, device):
    k = torch.randint(8, (n,), device=device)
    angle = k * (2 * math.pi / 8)
    centers = torch.stack((angle.cos(), angle.sin()), dim=1) * 2
    return centers + 0.12 * torch.randn(n, 2, device=device)


def load_mnist():
    """Read the public IDX format directly, without torchvision."""
    root = Path('data')
    root.mkdir(exist_ok=True)
    name = 'train-images-idx3-ubyte.gz'
    path = root / name
    if not path.exists():
        urllib.request.urlretrieve('https://storage.googleapis.com/cvdf-datasets/mnist/' + name, path)
    with gzip.open(path, 'rb') as f:
        magic, n, h, w = struct.unpack('>IIII', f.read(16))
        assert magic == 2051 and (h, w) == (28, 28)
        array = np.frombuffer(f.read(), dtype=np.uint8).copy().reshape(n, 1, h, w)
    return torch.from_numpy(array).float() / 127.5 - 1


def build_model(dataset, architecture='legacy'):
    if dataset == 'toy':
        return ToyNet()
    if architecture == 'digit_unet_v2':
        from digit_model import DigitUNet
        return DigitUNet()
    if architecture != 'legacy':
        raise ValueError(f'Unknown architecture: {architecture}')
    return ImageNet()


def schedule(device, steps=200):
    # Linear beta schedule rescaled from 1000 steps; terminal alpha_bar is near zero.
    betas = torch.linspace(1e-4 * 1000 / steps, 0.02 * 1000 / steps, steps, device=device)
    alphas = 1 - betas
    return betas, alphas, alphas.cumprod(0)


def expand(values, x):
    return values.reshape(-1, *([1] * (x.ndim - 1)))


@torch.no_grad()
def sample(model, method, initial, steps=100, stochastic=True, diffusion_steps=200, clip_denoised=False):
    """FM: Euler ODE. DDPM: ancestral chain; optional deterministic DDIM baseline."""
    x = initial.clone()
    trajectory = [x.cpu()]
    batch = len(x)
    if method == 'fm':
        for i in range(steps):
            t = torch.full((batch,), i / steps, device=x.device)
            x = x + model(x, t) / steps
            trajectory.append(x.cpu())
    else:
        betas, alphas, abar = schedule(x.device, diffusion_steps)
        # Ancestral DDPM uses all 200 training timesteps; DDIM supports a subset.
        indices = list(range(diffusion_steps-1, -1, -1)) if stochastic else torch.linspace(diffusion_steps-1, 0, steps).long().tolist()
        for j, i in enumerate(indices):
            t = torch.full((batch,), (i + 1) / diffusion_steps, device=x.device)
            epsilon = model(x, t)
            if stochastic:
                previous = abar[i - 1] if i else torch.ones((), device=x.device)
                variance = betas[i] * (1 - previous) / (1 - abar[i])
                mean = (x - betas[i] / (1 - abar[i]).sqrt() * epsilon) / alphas[i].sqrt()
                if clip_denoised:
                    clean = ((x - (1-abar[i]).sqrt()*epsilon)/abar[i].sqrt()).clamp(-1,1)
                    mean = clean if i == 0 else betas[i]*previous.sqrt()/(1-abar[i])*clean + (1-previous)*alphas[i].sqrt()/(1-abar[i])*x
                x = mean + variance.sqrt() * torch.randn_like(x) if i else mean
            else:
                clean = (x - (1 - abar[i]).sqrt() * epsilon) / abar[i].sqrt()
                if clip_denoised:
                    clean = clean.clamp(-1,1)
                previous = abar[indices[j + 1]] if j + 1 < len(indices) else torch.ones((), device=x.device)
                x = previous.sqrt() * clean + (1 - previous).sqrt() * epsilon
            trajectory.append(x.cpu())
    return torch.stack(trajectory)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['toy', 'mnist'], default='toy')
    parser.add_argument('--method', choices=['fm', 'ddpm'], default='fm')
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('GPU training requested but no CUDA device is visible. Request an allocation first.')
    out = Path('results') / f'{args.dataset}_{args.method}'
    out.mkdir(parents=True, exist_ok=True)
    model = (ToyNet() if args.dataset == 'toy' else ImageNet()).to(device)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    images = load_mnist().to(device) if args.dataset == 'mnist' else None
    _, _, abar = schedule(device)
    losses = []
    started = time.perf_counter()
    for step in range(1, args.steps + 1):
        model.train()
        x1 = toy_data(args.batch_size, device) if images is None else images[torch.randint(len(images), (args.batch_size,), device=device)]
        noise = torch.randn_like(x1)
        if args.method == 'fm':
            t = torch.rand(len(x1), device=device)
            xt = (1 - expand(t, x1)) * noise + expand(t, x1) * x1
            target = x1 - noise
        else:
            indices = torch.randint(200, (len(x1),), device=device)
            a = expand(abar[indices], x1)
            xt = a.sqrt() * x1 + (1 - a).sqrt() * noise
            t = (indices + 1).float() / 200
            target = noise
        prediction = model(xt, t)
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        with torch.no_grad():
            for average, current in zip(ema.parameters(), model.parameters()):
                average.lerp_(current, 0.005)
        losses.append(float(loss.detach()))
        if step % 200 == 0:
            print(f'{args.dataset} {args.method}: {step}/{args.steps}, loss={np.mean(losses[-200:]):.5f}', flush=True)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    metadata = {**vars(args), 'training_seconds': seconds, 'parameters': sum(p.numel() for p in model.parameters()),
                'torch_version': str(torch.__version__), 'hardware': torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU',
                'precision': 'FP32', 'ema_decay': 0.995, 'learning_rate': 3e-4, 'diffusion_steps': 200,
                'last_200_loss': float(np.mean(losses[-200:]))}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    np.save(out / 'loss.npy', np.array(losses))
    torch.save({'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
                'config': metadata, 'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state() if device.type == 'cuda' else None}, out / 'checkpoint.pt')
    # Same initial seed for each method; all frames are actual model sampling states.
    torch.manual_seed(2026)
    shape = (1024, 2) if args.dataset == 'toy' else (64, 1, 28, 28)
    initial = torch.randn(*shape, device=device)
    trajectory = sample(ema, args.method, initial)
    torch.save(trajectory, out / 'trajectory.pt')
    if args.method == 'ddpm':
        torch.save(sample(ema, 'ddpm', initial, steps=100, stochastic=False), out / 'ddim_trajectory.pt')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
