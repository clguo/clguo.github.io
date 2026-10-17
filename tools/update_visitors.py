#!/usr/bin/env python3
"""Archive public, aggregated visitor locations; never store visitor identifiers."""
import datetime as dt
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/visitor-locations.json'
SOURCE = 'https://mapmyvisitors.com/web/1c6nq'
SINCE = dt.date(2026, 7, 18)


def parse_locations(html):
    match = re.search(r'var addressPoints\s*=\s*(\[.*?\]);', html, re.S)
    if not match:
        raise ValueError('Visitor location data missing; keeping the previous archive.')
    rows = json.loads(match.group(1))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Empty visitor data; keeping the previous archive.')
    dates = re.findall(r'var (?:start|end)\s*=\s*"(\d{2}/\d{2}/\d{4})"', html)
    if len(dates) != 2:
        raise ValueError('Cannot verify the source date range.')
    start, end = [dt.datetime.strptime(d, '%m/%d/%Y').date() for d in dates]
    locations = []
    for row in rows:
        lat, lon, name, count = float(row[0]), float(row[1]), row[2], row[3]
        if not isinstance(name, str) or not isinstance(count, int) or count < 1:
            raise ValueError('Invalid aggregate location record.')
        # MapMyVisitors gives unknown locations an artificial ocean coordinate.
        if 'unknown' in name.lower():
            continue
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError('Invalid geographic coordinates.')
        locations.append({'name': name, 'lat': lat, 'lon': lon})
    if not locations:
        raise ValueError('No known locations returned.')
    return start, end, locations


def merge_locations(previous, current):
    def key(item):
        return (item['name'], round(item['lat'], 5), round(item['lon'], 5))
    # Keep every previously recorded source even when the provider's window rolls.
    merged = {key(item): item for item in previous}
    merged.update({key(item): item for item in current})
    return sorted(merged.values(), key=lambda item: item['name'].casefold())


def main():
    now = dt.datetime.now(dt.timezone.utc)
    today = now.date()
    previous = json.loads(OUTPUT.read_text()) if OUTPUT.exists() else None
    cookie = 'daterange=' + quote(f'custom|{SINCE}|{today}', safe='')
    result = subprocess.run([
        'curl', '--fail', '--silent', '--show-error', '--location',
        '--retry', '2', '--max-time', '45', '--cookie', cookie, SOURCE,
    ], check=True, capture_output=True, text=True)
    start, end, current = parse_locations(result.stdout)
    coverage_start = (dt.date.fromisoformat(previous['updatedAt'][:10])
                      + dt.timedelta(days=1)) if previous else SINCE
    if start > coverage_start or end < today:
        raise ValueError('Source date range leaves a gap in history; archive not changed.')
    locations = merge_locations(previous['locations'] if previous else [], current)
    data = {
        'source': SOURCE,
        'since': str(SINCE),
        'updatedAt': now.isoformat(timespec='seconds'),
        'sourcePeriod': {'start': str(start), 'end': str(end)},
        'locations': locations,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(OUTPUT)
    print(f'Saved {len(locations)} historical locations; current source has {len(current)}.')


if __name__ == '__main__':
    main()
