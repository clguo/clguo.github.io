"""Export Markdown and its relative assets as a homepage-ready HTML article."""
import argparse
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('blog.md'))
    parser.add_argument('--output', type=Path, default=Path('index.html'))
    args = parser.parse_args()
    source = args.source.read_text()
    title = source.splitlines()[0].lstrip('# ').strip()
    equations = []

    def protect(match):
        equations.append(match.group())
        return f'MATHEQUATIONPLACEHOLDER{len(equations)-1}END'

    source = re.sub(r'\\\[.*?\\\]', protect, source, flags=re.S)
    renderer = MarkdownIt('commonmark').enable('table')
    body = renderer.render(source)
    for name, poster in [('unified_generation', 'unified_samples'), ('training_paths', 'training_paths_poster'), ('toy_generation', 'toy_samples'), ('mnist_generation', 'mnist_samples'), ('toy_reflow', 'toy_reflow_budgets'), ('mnist_reflow', 'mnist_reflow_budgets'), ('toy_four_way', 'toy_four_way_samples'), ('mnist_four_way', 'mnist_four_way_samples')]:
        pattern = r'<img src="assets/' + name + r'\.gif" alt="([^"]*)"\s*/?>'
        body = re.sub(pattern, lambda match: '<video controls loop muted playsinline preload="none" poster="assets/' + poster + '.png" aria-label="' + match.group(1) + '"><source src="assets/' + name + '.mp4" type="video/mp4">' + match.group() + '</video>', body)
    for i, equation in enumerate(equations):
        body = body.replace(f'MATHEQUATIONPLACEHOLDER{i}END', html.escape(equation))
    headings = []

    def anchor(match):
        title = match.group(1)
        identifier = re.sub(r'[^a-z0-9]+', '-', re.sub('<[^>]*>', '', title).lower()).strip('-')
        contents_title = re.sub(r'^\d+\.\s+', '', title)
        headings.append(f'<li><a href="#{identifier}">{contents_title}</a></li>')
        return f'<h2 id="{identifier}">{title}</h2>'

    body = re.sub(r'<h2>(.*?)</h2>', anchor, body)
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Learn generative modeling with PyTorch: a step-by-step student lab, controlled experiments, and real sampling animations.">
<title>Flow Matching and Rectified Flow from Scratch</title>
<style>
:root { color-scheme: light; --ink:#182339; --muted:#526078; --blue:#305db7; }
* {box-sizing:border-box} body{margin:0;color:var(--ink);background:#fcfcfe;font:18px/1.8 system-ui,sans-serif}
main{max-width:900px;margin:auto;padding:64px 28px 100px} h1{font-size:clamp(2.1rem,5vw,3.3rem);line-height:1.15;letter-spacing:-.035em}
h2{font-size:1.6rem;line-height:1.35;margin-top:3rem;scroll-margin-top:30px} a{color:var(--blue)}
p{margin:1.2em 0} img,video{max-width:100%;height:auto;display:block;margin:24px auto;border-radius:8px}
pre{background:#111b2d;color:#e4eaf7;padding:22px;border-radius:10px;overflow:auto;font-size:14px;line-height:1.65}
code{font-family:ui-monospace,monospace} p code,li code{background:#eaf0fa;padding:.12em .3em;border-radius:4px;font-size:.86em}
table{display:block;overflow:auto;border-collapse:collapse;width:100%;font-size:15px} th,td{border-bottom:1px solid #dce3ef;padding:12px;text-align:left}
th{background:#eef2f9} nav{background:#eef2f9;padding:18px 25px;border-radius:10px;font-size:15px} nav ol{columns:2;padding-left:20px}
.series-nav{background:transparent;padding:0;margin:0 0 24px;font-size:14px}.student-links{background:#eef2f9;padding:14px 18px;border-radius:8px;font-size:14px;margin-bottom:28px}.student-links a{display:inline-block;margin-right:18px}:focus-visible{outline:3px solid #305db7;outline-offset:4px}
.eyebrow{text-transform:uppercase;color:var(--blue);font-size:13px;font-weight:700;letter-spacing:.12em}
footer{border-top:1px solid #dce3ef;margin-top:50px;padding-top:20px;color:var(--muted);font-size:14px}
.MathJax{max-width:100%;overflow-x:auto;overflow-y:hidden}
@media(max-width:600px){main{padding:28px 18px}body{font-size:16px}nav ol{columns:1}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
</style>
<script>window.MathJax={tex:{displayMath:[['\\\\[','\\\\]']]},svg:{fontCache:'global'}};</script>
<script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head><body><main><nav class="series-nav" aria-label="Series navigation"><a href="../../../">Changlu Guo</a> / <a href="../">Video Generation from Scratch</a> / Foundations</nav>
<div class="student-links"><a href="start-here.html">Start coding, step by step →</a><a href="https://github.com/clguo/clguo.github.io/tree/main/tutorials/video-generation/foundations">View code on GitHub ↗</a><a href="mnist-generative-models-unified.zip" download>Download code ↓</a></div>
<div class="eyebrow">Generative modeling · from scratch</div>
'''
    # Place the navigable contents after the opening introduction.
    first_heading = body.find('<h2')
    body = body[:first_heading] + '<nav aria-label="Contents"><strong>In this tutorial</strong><ol>' + ''.join(headings) + '</ol></nav>' + body[first_heading:]
    page += body + '<footer>Built from blog.md. Figures and GIFs are local; mathematical typesetting loads MathJax from a CDN. Companion source: unified/train.py, unified/processes.py, unified/evaluate.py and digit_model.py.</footer></main></body></html>'
    page = page.replace('<title>Flow Matching and Rectified Flow from Scratch</title>', '<title>' + html.escape(title) + '</title>')
    page = page.replace('Built from blog.md.', 'Built from ' + html.escape(args.source.name) + '.')
    canonical = 'https://clguo.github.io/tutorials/video-generation/foundations/' + ('' if args.output.name == 'index.html' else args.output.name)
    page = page.replace('</head>', '<link rel="canonical" href="' + canonical + '">\n</head>')
    args.output.write_text(page)
    print(f'Wrote {args.output}; publish with the assets/ directory.')


if __name__ == '__main__':
    main()
