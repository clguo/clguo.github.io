# Changlu Guo's academic homepage

Static academic homepage published from the main branch with GitHub Pages.
Citation metrics are maintained manually; the page shows their reference date.

## Historical visitor map

The homepage plots every known geographic source since the MapMyVisitors counter
was installed on 18 July 2026. `tools/update_visitors.py` reads only the public
aggregated `addressPoints` from https://mapmyvisitors.com/web/1c6nq and merges
locations into `data/visitor-locations.json`. It never publishes IP addresses,
visitor IDs, or individual visit records. Previously recorded locations are not
removed when the provider changes its displayed date range. Unknown-location
placeholder coordinates are excluded. Colocated geographic labels share a dot.

The `Update historical visitor map` GitHub Actions workflow syncs hourly (subject
to GitHub scheduling delays) and deploys the updated site. Failed requests or
invalid/incomplete source date ranges leave the previous archive and deployment
intact. The page displays the last successful data timestamp. The original
MapMyVisitors widget continues recording visits and displaying the live total;
its recent-only map is hidden. Citation metrics remain manual.

Run locally: `python3 tools/update_visitors.py`.
Validation: `python3 -m unittest discover -s tools`.

Map rendering uses Leaflet 1.9.4 (BSD-2-Clause, license in `vendor/leaflet/LICENSE`).
Basemap: Natural Earth 1:110m countries, public domain, from
https://github.com/nvkelso/natural-earth-vector/blob/master/geojson/ne_110m_admin_0_countries.geojson
(properties reduced to country names).
