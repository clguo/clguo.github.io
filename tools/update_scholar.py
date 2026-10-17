"""Refresh this site's public Google Scholar metrics with one daily request."""
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
PROFILE_URL = 'https://scholar.google.com/citations?user=y6XNTKEAAAAJ&hl=en&pagesize=100'


class Element:
    def __init__(self, tag='', attrs=()):
        self.tag, self.attrs, self.children = tag, dict(attrs), []

    def text(self):
        return ''.join(c.text() if isinstance(c, Element) else c for c in self.children)

    def find_all(self, *, tag=None, id=None, cls=None):
        matches = []
        for child in self.children:
            if not isinstance(child, Element):
                continue
            if ((tag is None or child.tag == tag)
                    and (id is None or child.attrs.get('id') == id)
                    and (cls is None or cls in child.attrs.get('class', '').split())):
                matches.append(child)
            matches.extend(child.find_all(tag=tag, id=id, cls=cls))
        return matches


class Document(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.root = Element()
        self.stack = [self.root]
        self.feed(html)
        self.close()

    def handle_starttag(self, tag, attrs):
        node = Element(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in {'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                       'link', 'meta', 'param', 'source', 'track', 'wbr'}:
            self.stack.append(node)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        self.stack[-1].children.append(data)


def title_key(title):
    return re.sub(r'[^a-z0-9]', '', title.lower())


def number(text):
    value = text.strip().replace(',', '').replace('\u00a0', '')
    if not re.fullmatch(r'[0-9]+', value):
        raise ValueError('Missing or invalid citation number')
    return int(value)


def parse_profile(html, allowed_titles):
    root = Document(html).root
    names = root.find_all(id='gsc_prf_in')
    if len(names) != 1 or names[0].text().strip() != 'Changlu Guo':
        raise ValueError('Expected public Scholar profile is missing; keeping previous data')
    tables = root.find_all(id='gsc_rsb_st')
    if len(tables) != 1:
        raise ValueError('Scholar metrics table is missing; keeping previous data')
    fields = {'Citations': 'citations', 'h-index': 'h_index', 'i10-index': 'i10_index'}
    metrics = {}
    for row in tables[0].find_all(tag='tr'):
        cells = row.find_all(tag='td')
        values = row.find_all(tag='td', cls='gsc_rsb_std')
        if cells and cells[0].text().strip() in fields and values:
            metrics[fields[cells[0].text().strip()]] = number(values[0].text())
    if set(metrics) != set(fields.values()) or not (
            0 < metrics['h_index'] <= metrics['citations']
            and 0 <= metrics['i10_index'] * 10 <= metrics['citations']):
        raise ValueError('Incomplete or inconsistent Scholar metrics; keeping previous data')
    allowed = {title_key(title) for title in allowed_titles}
    publications = {}
    for row in root.find_all(tag='tr', cls='gsc_a_tr'):
        titles = row.find_all(cls='gsc_a_at')
        counts = row.find_all(cls='gsc_a_ac')
        if not titles or not counts:
            continue
        key = title_key(titles[0].text())
        if key in allowed:
            text = counts[0].text().strip()
            publications[key] = number(text) if text else 0
    if not publications:
        raise ValueError('No known publications found; keeping previous data')
    return {**metrics, 'publications': publications}


def update(output=ROOT / 'data/scholar.json'):
    # Whitelist existing website titles. Never import new papers or manuscript text.
    source = (ROOT / 'publications.js').read_text(encoding='utf-8')
    publications = json.loads(source.removeprefix('const publicationsData = ').strip().removesuffix(';'))
    request = Request(PROFILE_URL, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; AcademicHomepageMetrics/1.0; +https://clguo.github.io/)',
        'Accept-Language': 'en',
    })
    with urlopen(request, timeout=30) as response:
        html = response.read().decode('utf-8')
    result = parse_profile(html, [p['title'] for p in publications['publications']])
    result.update(profile_id='y6XNTKEAAAAJ', source=PROFILE_URL,
                  updated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output)
    print(f"Updated Scholar metrics: {result['citations']} citations, h-index {result['h_index']}, i10-index {result['i10_index']}")


if __name__ == '__main__':
    update()
