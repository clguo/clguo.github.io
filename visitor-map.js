/* Historical geographic sources, independent of the counter's rolling map. */
(async function () {
  const container = document.getElementById('visitor-history-map');
  const status = document.getElementById('visitor-history-status');
  const list = document.getElementById('visitor-location-list');
  if (!container || !status || !list) return;

  try {
    const responses = await Promise.all([
      fetch('data/visitor-locations.json', { cache: 'no-cache' }),
      fetch('data/world-countries.geojson')
    ]);
    if (responses.some(response => !response.ok)) throw new Error('Map data unavailable');
    const [history, countries] = await Promise.all(responses.map(response => response.json()));
    if (!history.locations.length) throw new Error('No historical locations');

    for (const location of history.locations) {
      const item = document.createElement('li');
      item.textContent = location.name;
      list.appendChild(item);
    }
    document.getElementById('visitor-location-summary').textContent =
      `View all ${history.locations.length} recorded sources`;
    const updated = new Date(history.updatedAt);
    const formatted = new Intl.DateTimeFormat('en-GB', {
      day: 'numeric', month: 'short', year: 'numeric',
      hour: '2-digit', minute: '2-digit', timeZone: 'UTC'
    }).format(updated);
    status.textContent = `All recorded sources since 18 July 2026 · Updated ${formatted} UTC`;
    document.getElementById('visitor-map-note').textContent =
      'Historical locations are retained. New sources sync hourly. Overlapping locations share a dot; visits without geolocation are not plotted.';

    if (!window.L) throw new Error('Map library unavailable');
    const map = L.map(container, {
      scrollWheelZoom: false,
      worldCopyJump: false,
      minZoom: 0,
      maxZoom: 8,
      maxBounds: [[-85, -210], [85, 210]],
      maxBoundsViscosity: 1
    }).setView([20, 0], 1);
    L.geoJSON(countries, {
      interactive: false,
      style: { fillColor: '#ffffff', fillOpacity: 1, color: '#d6e4ed', weight: 0.6 }
    }).addTo(map);

    // Merge colocated labels so Copenhagen / Kobenhavn S can both be inspected.
    const points = new Map();
    for (const location of history.locations) {
      const key = `${location.lat},${location.lon}`;
      if (!points.has(key)) points.set(key, { ...location, names: [] });
      points.get(key).names.push(location.name);
    }
    for (const point of points.values()) {
      const label = document.createElement('span');
      label.textContent = point.names.join(' / ');
      L.circleMarker([point.lat, point.lon], {
        radius: 5, color: '#ffffff', weight: 1, className: 'visitor-source-marker',
        fillColor: '#f65b64', fillOpacity: 0.95
      }).bindTooltip(label).bindPopup(label.cloneNode(true)).addTo(map);
    }
    const fitWorld = () => map.fitBounds([[-56, -180], [80, 180]], { padding: [8, 8] });
    fitWorld();
    new ResizeObserver(() => { map.invalidateSize(); fitWorld(); }).observe(container);
    map.attributionControl.addAttribution('Map data: Natural Earth');
  } catch (error) {
    container.textContent = 'The map could not load. Use the source list below or open visitor statistics.';
    container.classList.add('visitor-map-unavailable');
    if (!list.children.length) status.textContent = 'Historical visitor data is temporarily unavailable.';
    console.warn('Visitor history map:', error.stack);
  }
}());
