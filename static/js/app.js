/* FloodSense Assam — Application JS
   Google Maps-style tiles, location pins, colored district borders, plain English UI
*/

'use strict';

const state = {
  districts: [],
  metrics: null,
  selected: null,
  map: null,
  geojsonLayer: null,
  markers: [],
};

const RISK = {
  high:     { color: '#c0392b', fill: 'rgba(192,57,43,0.18)', border: '#c0392b', weight: 3 },
  moderate: { color: '#e67e22', fill: 'rgba(230,126,34,0.15)', border: '#e67e22', weight: 2.5 },
  low:      { color: '#27ae60', fill: 'rgba(39,174,96,0.08)',  border: '#27ae60', weight: 1.5 },
};

const TIPS = {
  high: [
    'Move to higher ground immediately if near a river or low-lying area',
    'Do not attempt to cross flooded roads or bridges',
    'Contact the ASDMA helpline (1070) for evacuation assistance',
    'Carry essential documents and emergency supplies',
    'Stay tuned to All India Radio (AIR) and official ASDMA updates',
  ],
  moderate: [
    'Keep essential items ready in case of sudden rise in water level',
    'Monitor river levels closely through local authorities',
    'Avoid travelling to flood-prone areas unless necessary',
    'Store drinking water and dry food for at least 3 days',
    'Keep emergency contact numbers saved on your phone',
  ],
  low: [
    'No immediate flood threat — continue monitoring weather updates',
    'Check ASDMA website for seasonal flood preparedness guides',
    'Ensure drains near your home are clear of blockages',
    'Report any unusual rise in river or water body levels to local authorities',
  ],
};

const FACTOR_ICONS = {
  'river_level':   { icon: '🌊', label: 'River level' },
  'pct_to_danger': { icon: '📏', label: 'Distance from danger level' },
  'rain_3d':       { icon: '🌧️', label: '3-day rainfall' },
  'rain_7d':       { icon: '☔', label: '7-day rainfall' },
  'rain_1d':       { icon: '🌦️', label: "Today's rainfall" },
  'level_delta':   { icon: '📈', label: 'Water level change' },
  'level_trend':   { icon: '📊', label: 'River level trend' },
  'upstream_3d':   { icon: '🏔️', label: 'Upstream catchment rain' },
  'is_monsoon':    { icon: '🌀', label: 'Monsoon season active' },
};

// ── Boot ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadData();
});

async function loadData() {
  setStatus('loading', 'Loading risk data…');
  try {
    const [distRes, metRes] = await Promise.all([
      fetch('/api/districts'),
      fetch('/api/metrics'),
    ]);
    const distData = await distRes.json();
    state.districts = distData.districts || [];
    state.metrics = await metRes.json();

    renderMapData();
    renderSummaryStrip();
    document.getElementById('updateTime').textContent =
      new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', hour12: true });

    document.getElementById('loadingOverlay').style.display = 'none';
    document.getElementById('summaryStrip').style.display = 'block';
    setStatus('ready', 'Live data active');
  } catch (err) {
    console.error(err);
    setStatus('loading', 'Retrying…');
    setTimeout(loadData, 3000);
  }
}

function setStatus(type, text) {
  const dot = document.getElementById('statusDot');
  dot.className = 'status-dot-live ' + type;
  document.getElementById('statusText').textContent = text;
}

// ── Map Init — Google Maps Light tiles ───────────────────────────────
function initMap() {
  state.map = L.map('map', {
    zoomControl: true,
    attributionControl: false,
    scrollWheelZoom: true,
  });

  // Google Maps-style road map tiles (Carto Light is closest free equivalent)
  L.tileLayer(
    'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
    { subdomains: 'abcd', maxZoom: 14 }
  ).addTo(state.map);

  // Move zoom control to top-right
  state.map.zoomControl.setPosition('topright');
}

// ── Render map layers ─────────────────────────────────────────────────
function renderMapData() {
  const riskMap = {};
  state.districts.forEach(d => { riskMap[d.district] = d; });

  fetch('/static/data/assam_districts.geojson')
    .then(r => r.json())
    .then(geojson => {
      if (state.geojsonLayer) state.geojsonLayer.remove();
      state.markers.forEach(m => m.remove());
      state.markers = [];

      // GeoJSON district polygons with colored borders
      state.geojsonLayer = L.geoJSON(geojson, {
        style: feature => {
          const name = getPropName(feature);
          const d = matchDistrict(name, riskMap);
          const risk = d ? d.risk : 'low';
          const s = RISK[risk];
          return {
            fillColor:   s.fill,
            fillOpacity: 1,
            color:       s.border,
            weight:      s.weight,
            opacity:     0.9,
            dashArray:   risk === 'low' ? '4 3' : null,
          };
        },
        onEachFeature: (feature, layer) => {
          const name = getPropName(feature);
          const d = matchDistrict(name, riskMap);

          if (!d) return;

          // Tooltip on hover
          layer.bindTooltip(makeTooltip(d), {
            sticky: true, opacity: 1, className: '',
          });

          // Highlight on hover
          layer.on('mouseover', e => {
            const l = e.target;
            l.setStyle({ weight: RISK[d.risk].weight + 1.5, fillOpacity: 0.35 });
          });
          layer.on('mouseout', e => {
            state.geojsonLayer.resetStyle(e.target);
          });
          layer.on('click', () => {
            selectDistrict(d);
            document.getElementById('mapHint').classList.add('hidden');
          });
        },
      }).addTo(state.map);

      state.map.fitBounds(state.geojsonLayer.getBounds(), { padding: [20, 20] });

      // Add location pins for high + moderate districts
      state.districts
        .filter(d => d.risk !== 'low')
        .forEach(d => addPin(d));

      // Alert banner
      const highCount = state.districts.filter(d => d.risk === 'high').length;
      if (highCount > 0) {
        const banner = document.getElementById('alertBanner');
        const highNames = state.districts
          .filter(d => d.risk === 'high')
          .map(d => d.district)
          .join(', ');
        document.getElementById('alertBannerText').textContent =
          `⚠ Flood alert active in: ${highNames}. Follow ASDMA advisories and call 1070 for help.`;
        banner.style.display = 'block';
        document.body.classList.add('has-alert');
      }
    });
}

function getPropName(feature) {
  return feature.properties?.district || feature.properties?.DISTRICT ||
         feature.properties?.NAME_2   || feature.properties?.name || '';
}

function matchDistrict(geoName, riskMap) {
  if (!geoName) return null;
  const clean = geoName.trim();
  if (riskMap[clean]) return riskMap[clean];
  for (const key of Object.keys(riskMap)) {
    if (clean.toLowerCase().includes(key.toLowerCase()) ||
        key.toLowerCase().includes(clean.toLowerCase())) {
      return riskMap[key];
    }
  }
  return null;
}

// ── Location Pin Markers ──────────────────────────────────────────────
function addPin(d) {
  // Get centroid of district from geojson (approximate via bounds of layer)
  // We'll use a custom DivIcon shaped like a Google Maps pin
  const pinColor = d.risk === 'high' ? '#c0392b' : '#e67e22';
  const svgPin = `
    <svg xmlns="http://www.w3.org/2000/svg" width="28" height="36" viewBox="0 0 24 32">
      <path d="M12 0C7.58 0 4 3.58 4 8c0 7.5 8 16 8 16s8-8.5 8-16c0-4.42-3.58-8-8-8z"
            fill="${pinColor}" stroke="white" stroke-width="1.2"/>
      <circle cx="12" cy="8.5" r="3.2" fill="white"/>
    </svg>`;

  const icon = L.divIcon({
    html: svgPin,
    className: '',
    iconSize: [28, 36],
    iconAnchor: [14, 36],
    tooltipAnchor: [0, -30],
  });

  // We need a lat/lng — derive from geojson centroid. Since we can't easily
  // get it without turf, we'll use known approximate district centroids.
  const centroid = DISTRICT_CENTROIDS[d.district];
  if (!centroid) return;

  const marker = L.marker(centroid, { icon, zIndexOffset: 1000 });
  marker.bindTooltip(makeTooltip(d), { opacity: 1, offset: [0, -32] });
  marker.on('click', () => selectDistrict(d));
  marker.addTo(state.map);
  state.markers.push(marker);
}

function makeTooltip(d) {
  return `<div>
    <div class="tt-name">${d.district}</div>
    <div class="tt-river">${(d.river || '').replace(/_/g,' ')} River</div>
    <div class="tt-badge ${d.risk}">${d.risk.toUpperCase()}</div>
    ${d.pct_to_danger ? `<div class="tt-pct">${d.pct_to_danger}% to danger</div>` : ''}
  </div>`;
}

// Approximate centroids for Assam districts [lat, lng]
const DISTRICT_CENTROIDS = {
  'Kamrup Metro':  [26.1445, 91.7362],
  'Kamrup':        [26.3058, 91.3740],
  'Morigaon':      [26.2500, 92.3300],
  'Nagaon':        [26.3472, 92.6836],
  'Golaghat':      [26.5190, 93.9690],
  'Jorhat':        [26.7509, 94.2037],
  'Majuli':        [27.0000, 94.1667],
  'Sivasagar':     [26.9800, 94.6350],
  'Dibrugarh':     [27.4728, 94.9120],
  'Tinsukia':      [27.4893, 95.3596],
  'Lakhimpur':     [27.2340, 94.1020],
  'Dhemaji':       [27.4826, 94.5660],
  'Sonitpur':      [26.6340, 92.7970],
  'Biswanath':     [26.7340, 93.1450],
  'Darrang':       [26.4830, 91.9860],
  'Udalguri':      [26.7540, 92.1060],
  'Barpeta':       [26.3240, 91.0020],
  'Nalbari':       [26.4430, 91.4350],
  'Chirang':       [26.5880, 90.5710],
  'Bongaigaon':    [26.4770, 90.5580],
  'Kokrajhar':     [26.4010, 90.2700],
  'Dhubri':        [26.0200, 89.9750],
  'Goalpara':      [26.1720, 90.6220],
  'South Salmara': [25.9100, 89.9900],
  'Cachar':        [24.8330, 92.7580],
  'Hailakandi':    [24.6840, 92.5610],
  'Karimganj':     [24.8640, 92.3510],
  'Dima Hasao':    [25.5740, 93.0200],
  'Karbi Anglong': [26.1330, 93.8220],
  'West Karbi':    [26.3000, 93.1000],
  'Charaideo':     [27.0200, 94.8350],
  'Hojai':         [26.0050, 92.8500],
  'Bajali':        [26.4600, 91.2500],
};

// ── Summary Strip ─────────────────────────────────────────────────────
function renderSummaryStrip() {
  const high = state.districts.filter(d => d.risk === 'high').length;
  const mod  = state.districts.filter(d => d.risk === 'moderate').length;
  const low  = state.districts.filter(d => d.risk === 'low').length;
  document.getElementById('highCount').textContent = high;
  document.getElementById('modCount').textContent  = mod;
  document.getElementById('lowCount').textContent  = low;
}

// ── District selection ────────────────────────────────────────────────
function selectDistrict(d) {
  state.selected = d;
  document.getElementById('panelEmpty').style.display   = 'none';
  document.getElementById('panelContent').style.display = 'block';

  // Header
  document.getElementById('pcDistrictName').textContent = d.district;
  document.getElementById('pcRiverName').textContent = `${(d.river||'').replace(/_/g,' ')} River`;
  const badge = document.getElementById('pcRiskBadge');
  badge.textContent = d.risk === 'high' ? 'High Risk' : d.risk === 'moderate' ? 'Moderate Risk' : 'Safe';
  badge.className = `pc-risk-badge ${d.risk}`;

  // Risk statement in plain English
  const stmt = document.getElementById('riskStatement');
  stmt.className = `risk-statement ${d.risk}`;
  if (d.risk === 'high') {
    stmt.textContent = `⚠ Flood danger in ${d.district}. The ${(d.river||'').replace(/_/g,' ')} River is at ${d.pct_to_danger}% of its danger level. Immediate precautions are advised.`;
  } else if (d.risk === 'moderate') {
    stmt.textContent = `🔶 Moderate flood risk in ${d.district}. The ${(d.river||'').replace(/_/g,' ')} River is rising and at ${d.pct_to_danger}% of the danger threshold. Stay prepared.`;
  } else {
    stmt.textContent = `✓ ${d.district} is currently safe. River levels are normal with no immediate flood threat.`;
  }

  // Water level visual indicator
  const pct = Math.min(100, d.pct_to_danger || 0);
  // Track is: 0-50% = safe (green), 50-75% = warning (orange), 75-100% = danger (red)
  // Pin position across full bar
  const indicator = document.getElementById('wlbIndicator');
  indicator.style.left = `${Math.min(95, Math.max(5, pct))}%`;
  const pinIcon = indicator.querySelector('.pin-icon');
  pinIcon.className = `pin-icon ${d.risk}`;
  document.getElementById('wlbLevelVal').textContent = (d.river_level || 0).toFixed(2) + 'm';

  // Note below bar
  document.getElementById('wlbNote').textContent = pct >= 90
    ? `⛔ Above danger level (${d.danger_level}m). Evacuate low-lying areas.`
    : pct >= 65
    ? `⚠ Above warning level (${d.warning_level}m). Stay alert for sudden rises.`
    : `River level is within safe range. Warning threshold: ${d.warning_level}m.`;

  // Rainfall
  document.getElementById('rainToday').textContent = (d.rain_1d || d.rain_3d/3 || 0).toFixed(1) + ' mm';
  document.getElementById('rain3d').textContent    = (d.rain_3d || 0).toFixed(1) + ' mm';
  document.getElementById('rain7d').textContent    = (d.rain_7d || 0).toFixed(1) + ' mm';

  // Forecast
  const fcRow = document.getElementById('forecastRow');
  fcRow.innerHTML = '';
  (d.forecast || []).forEach(f => {
    const riskClass = f.pct >= 90 ? 'high' : f.pct >= 65 ? 'moderate' : 'low';
    const riskWord  = riskClass === 'high' ? 'High Risk' : riskClass === 'moderate' ? 'Moderate' : 'Safe';
    fcRow.innerHTML += `
      <div class="fc-item ${riskClass}">
        <div class="fc-date">${f.date}</div>
        <div class="fc-level">${f.level}m</div>
        <div class="fc-risk-label">${riskWord}</div>
      </div>`;
  });

  // Crop card
  const cropCard = document.getElementById('cropCard');
  if (d.crop_ha && d.crop_ha > 100 && d.risk !== 'low') {
    cropCard.style.display = 'block';
    document.getElementById('cropNumber').textContent = d.crop_ha.toLocaleString('en-IN');
  } else {
    cropCard.style.display = 'none';
  }

  // Factors (plain English SHAP)
  const fl = document.getElementById('factorsList');
  fl.innerHTML = '';
  const factors = (d.shap_features || []).slice(0, 5);
  factors.forEach(f => {
    const meta = FACTOR_ICONS[f.feature] || { icon: '•', label: f.label };
    const barW = Math.min(100, Math.abs(f.contribution) * 180);
    const dir  = f.contribution > 0 ? 'up' : 'down';
    const effect = dir === 'up'
      ? 'is <strong>increasing</strong> the flood risk'
      : 'is <strong>reducing</strong> the flood risk';
    fl.innerHTML += `
      <div class="factor-item">
        <div class="factor-icon">${meta.icon}</div>
        <div class="factor-text">${meta.label} ${effect}</div>
        <div class="factor-bar">
          <div class="fb-track">
            <div class="fb-fill ${dir}" style="width:${barW}%"></div>
          </div>
          <div class="fb-dir ${dir}">${dir === 'up' ? '↑ risk' : '↓ risk'}</div>
        </div>
      </div>`;
  });

  // Advisory
  const advHeader = document.getElementById('advHeader');
  advHeader.className = `adv-header ${d.risk}`;
  const icons = { high: '⚠️', moderate: '🔶', low: '✅' };
  const titles = { high: 'Flood Warning — Take Immediate Action', moderate: 'Flood Watch — Be Prepared', low: 'Conditions are Safe' };
  document.getElementById('advIcon').textContent  = icons[d.risk];
  document.getElementById('advTitle').textContent = titles[d.risk];
  const tips = document.getElementById('advTips');
  tips.innerHTML = (TIPS[d.risk] || TIPS.low).map(t => `<li>${t}</li>`).join('');

  // Scroll panel to top
  document.getElementById('sidePanel').scrollTop = 0;
}

function clearSelection() {
  state.selected = null;
  document.getElementById('panelEmpty').style.display   = 'block';
  document.getElementById('panelContent').style.display = 'none';
}
