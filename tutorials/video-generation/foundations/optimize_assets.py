"""Produce small MP4s and compact GIF previews for homepage publication.

Requires ffmpeg on PATH. Re-run after render.py if figures are regenerated.
"""
import argparse
from pathlib import Path
import subprocess

from PIL import Image, ImageSequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='*', default=['training_paths', 'toy_generation', 'mnist_generation', 'toy_reflow', 'mnist_reflow', 'toy_four_way', 'mnist_four_way'])
    args = parser.parse_args()
    encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True, check=True).stdout
    if 'libx264' in encoders:
        codec = ['-c:v', 'libx264', '-crf', '24']
    elif 'libopenh264' in encoders:
        codec = ['-c:v', 'libopenh264', '-b:v', '2M']
    else:
        raise RuntimeError('Install ffmpeg with a software H.264 encoder: libx264 or libopenh264.')
    for name in args.names:
        source = Path('assets') / f'{name}.gif'
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-vf',
                        'scale=trunc(iw/2)*2:trunc(ih/2)*2', *codec,
                        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(source.with_suffix('.mp4'))], check=True)
        frames, durations = [], []
        with Image.open(source) as animation:
            for frame in ImageSequence.Iterator(animation):
                rgb = frame.convert('RGB')
                width = min(960, rgb.width)
                rgb = rgb.resize((width, round(rgb.height * width / rgb.width)), Image.Resampling.LANCZOS)
                frames.append(rgb.quantize(colors=64, dither=Image.Dither.NONE))
                durations.append(frame.info.get('duration', 80))
            if name == 'training_paths':
                rgb.save('assets/training_paths_poster.png')
        temporary = source.with_name(source.stem + '_compact.gif')
        frames[0].save(temporary, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
        temporary.replace(source)
        print(name, 'GIF bytes', source.stat().st_size, 'MP4 bytes', source.with_suffix('.mp4').stat().st_size)


if __name__ == '__main__':
    main()
