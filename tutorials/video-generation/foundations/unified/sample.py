"""Generate a labelled sample grid from one checkpoint, without a classifier.

Run from the foundations directory, for example:
python -m unified.sample --checkpoint results/student/flow/checkpoint.pt --method fm
"""
import argparse
import hashlib
import json
import textwrap
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from digit_model import DigitUNet
from .processes import generate

METHOD_STAGES = {
    'fm': 'flow', 'ddpm': 'diffusion', 'ddim': 'diffusion',
    'rf2': 'reflow', 'continued': 'continued',
}


def grid(samples, labels, columns, title):
    pixels = ((samples.detach().cpu().float().clamp(-1, 1) + 1) * 127.5).byte().numpy()[:, 0]
    cell, margin = 112, 32
    title_lines = textwrap.wrap(title, width=max(10, (margin + columns * cell - 16) // 6))
    heading = 16 + 12 * len(title_lines)
    rows = (len(pixels) + columns - 1) // columns
    canvas = Image.new('RGB', (margin + columns * cell, heading + rows * cell), '#f5f7fb')
    draw = ImageDraw.Draw(canvas)
    for line_number, line in enumerate(title_lines):
        draw.text((8, 8 + 12 * line_number), line, fill='#223248')
    for index, pixel in enumerate(pixels):
        row, column = divmod(index, columns)
        if column == 0:
            draw.text((10, heading + row * cell + cell // 2), str(int(labels[index])), fill='#223248')
        image = Image.fromarray(pixel).resize((cell - 4, cell - 4), Image.Resampling.NEAREST)
        canvas.paste(image.convert('RGB'), (margin + column * cell, heading + row * cell))
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--method', choices=METHOD_STAGES, required=True)
    parser.add_argument('--nfe', type=int, default=64, help='Network evaluations per image (1 to 1000).')
    parser.add_argument('--per-class', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--out', type=Path, default=Path('outputs/samples.png'))
    parser.add_argument('--gif', action='store_true', help='Also save the actual sampled trajectory as a GIF.')
    args = parser.parse_args()
    if not 1 <= args.nfe <= 1000 or not 1 <= args.per_class <= 20:
        parser.error('Use 1 <= --nfe <= 1000 and 1 <= --per-class <= 20.')
    if args.out.suffix.lower() != '.png':
        parser.error('--out must end in .png.')
    if not args.checkpoint.is_file():
        parser.error(f'Checkpoint not found: {args.checkpoint}. Run the matching training stage first.')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable. Use --device cpu for sampling, or select a CUDA GPU runtime.')
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    config = checkpoint['config']
    expected = METHOD_STAGES[args.method]
    if config['stage'] != expected:
        parser.error(f'{args.method} needs a {expected} checkpoint; this checkpoint is {config["stage"]}.')
    model = DigitUNet(num_classes=10).to(device).eval()
    model.load_state_dict(checkpoint['ema'])
    labels = torch.arange(10).repeat_interleave(args.per_class)
    # CPU noise creation keeps starting tensors identical across CPU/CUDA runs.
    initial = torch.randn(len(labels), 1, 28, 28, generator=torch.Generator().manual_seed(args.seed))
    sampler_rng = torch.Generator(device=device).manual_seed(args.seed + 1)
    with torch.no_grad():
        result = generate(model, initial.to(device), labels.to(device), args.method, args.nfe,
                          generator=sampler_rng, keep_trajectory=args.gif)
    samples = result[-1] if args.gif else result
    if not torch.isfinite(samples).all():
        raise RuntimeError('Sampler produced non-finite values. Check the checkpoint and training loss.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    title = f'{args.method} | {args.nfe} NFE | seed {args.seed} | trained {config["steps"]} updates'
    grid(samples, labels, args.per_class, title).save(args.out)
    if args.gif:
        # Include both endpoints while keeping long diffusion trajectories compact.
        indices = sorted(set(torch.linspace(0, args.nfe, min(args.nfe + 1, 33)).round().long().tolist()))
        frames = [grid(result[i], labels, args.per_class, f'{args.method} | {i}/{args.nfe} model calls') for i in indices]
        durations = [120] * len(frames)
        durations[-1] = 1500
        frames[0].save(args.out.with_suffix('.gif'), save_all=True, append_images=frames[1:],
                       duration=durations, loop=0)
    metadata = {
        'checkpoint': str(args.checkpoint), 'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'stage': config['stage'], 'training_steps': config['steps'], 'planned_steps': config['planned_steps'],
        'weights': 'ema', 'method': args.method, 'nfe': args.nfe, 'seed': args.seed,
        'sample_count': len(labels), 'per_class': args.per_class, 'labels': labels.tolist(),
        'selection': 'all generated samples, no filtering', 'device': device, 'torch_version': str(torch.__version__),
        'display_projection': 'clamp [-1,1] only for display',
    }
    args.out.with_suffix('.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saved {args.out} and {args.out.with_suffix(".json")}')
    if args.gif:
        print(f'Saved {args.out.with_suffix(".gif")}')
    if config['steps'] < config['planned_steps'] or (config['stage'] in ['flow', 'diffusion'] and config['planned_steps'] < 20000):
        print('Short learning run: inspect the pipeline; do not compare this grid with the published 20k-update results.')


if __name__ == '__main__':
    main()
