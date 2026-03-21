'use strict';

console.log('[FloodSense] app.js loaded');

const ASSAM_VIEW = {
  center:  [26.20, 92.94],
  bounds:  [[24.0, 89.45], [28.45, 96.15]],
  minZoom: 7,
  maxZoom: 11.5,
};

// Risk colours
const RISK = {
  low:      { color: '#27ae60', shadow: 'rgba(39,174,96,0.35)',    label: 'Low',      text: '#fff' },
  moderate: { color: '#e6b800', shadow: 'rgba(230,184,0,0.35)',    label: 'Moderate', text: '#2b2000' },
  high:     { color: '#e31a1c', shadow: 'rgba(227,26,28,0.40)',    label: 'High',     text: '#fff' },
};

// District centroids [lat, lng]
const CENTROIDS = {
  'Dhubri':        [26.02, 90.10], 'South Salmara': [25.70, 90.02],
  'Kokrajhar':     [26.50, 90.28], 'Chirang':        [26.58, 90.92],
  'Bongaigaon':    [26.48, 90.78], 'Goalpara':       [26.18, 90.62],
  'Barpeta':       [26.40, 91.28], 'Bajali':         [26.50, 91.68],
  'Tamulpur':      [26.60, 91.55], 'Nalbari':        [26.44, 91.60],
  'Kamrup':        [26.16, 91.60], 'Kamrup Metro':   [26.12, 91.82],
  'Darrang':       [26.58, 92.28], 'Udalguri':       [26.78, 92.22],
  'Morigaon':      [26.22, 92.36], 'Nagaon':         [26.22, 92.95],
  'Hojai':         [26.08, 92.98], 'Sonitpur':       [26.72, 93.08],
  'Biswanath':     [26.98, 93.00], 'Majuli':         [26.96, 94.02],
  'Jorhat':        [26.62, 94.48], 'Golaghat':       [26.38, 93.88],
  'Kaziranga':     [26.48, 93.72], 'Sibsagar':       [26.58, 94.72],
  'Charaideo':     [26.92, 94.88], 'Dhemaji':        [27.00, 94.78],
  'Lakhimpur':     [27.18, 94.42], 'Dibrugarh':      [27.30, 95.12],
  'Tinsukia':      [27.42, 95.38], 'Karbi Anglong':  [25.88, 93.38],
  'West Karbi':    [25.88, 92.68], 'Dima Hasao':     [25.12, 93.12],
  'Cachar':        [24.62, 92.88], 'Hailakandi':     [24.50, 92.72],
  'Karimganj':     [24.48, 92.36],
};

const state = {
  map:              null,
  markerLayer:      null,   // L.LayerGroup holding all pins
  districtData:     [],
  markerIndex:      new Map(), // district name → marker
  selectedDistrict: null,
  loading:          false,
};

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  console.log('[FloodSense] DOMContentLoaded fired');
  try {
    initMap();
    bindUI();
    loadDashboardData();
  } catch (err) {
    console.error('[FloodSense] bootstrap failed', err);
  }
});

window.addEventListener('error', event => {
  console.error('[FloodSense] window error', event.message, event.error);
});

function bindUI() {
  document.getElementById('refreshButton').addEventListener('click', () => {
    loadDashboardData({ forceRefresh: true });
  });
  document.getElementById('panelClose').addEventListener('click', closePanel);
  document.getElementById('panelScrim').addEventListener('click', closePanel);
}

// ── Map ───────────────────────────────────────────────────────────────────────

function initMap() {
  state.map = L.map('map', {
    zoomControl:        true,
    attributionControl: false,
    minZoom:            ASSAM_VIEW.minZoom,
    maxZoom:            ASSAM_VIEW.maxZoom,
    zoomSnap:           0.25,
    zoomDelta:          0.25,
    worldCopyJump:      false,
    maxBounds:          ASSAM_VIEW.bounds,
    maxBoundsViscosity: 1.0,
  }).setView(ASSAM_VIEW.center, ASSAM_VIEW.minZoom);

  // CartoDB Voyager — clean road map, Google Maps aesthetic
  L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png', {
    subdomains: 'abcd',
    maxZoom:    ASSAM_VIEW.maxZoom,
    noWrap:     true,
    bounds:     ASSAM_VIEW.bounds,
  }).addTo(state.map);

  state.map.zoomControl.setPosition('bottomleft');
  state.map.on('drag', () => state.map.panInsideBounds(ASSAM_VIEW.bounds, { animate: false }));

  // Marker group — easier to clear and re-add on refresh
  state.markerLayer = L.layerGroup().addTo(state.map);
}

// ── Data ──────────────────────────────────────────────────────────────────────

async function loadDashboardData(options = {}) {
  if (state.loading) return;
  console.log('[FloodSense] loadDashboardData triggered', options);
  state.loading = true;
  setLoadingState(true);
  showOverlay('Loading district risk data…');

  try {
    const payload = await fetchDistricts();
    state.districtData = Array.isArray(payload.districts) ? payload.districts : [];

    renderStats();
    renderTimestamp();
    renderPins();
    syncSelectedDistrict();
    hideOverlay();
  } catch (err) {
    console.error('loadDashboardData error:', err);
    hideOverlay();
  } finally {
    state.loading = false;
    setLoadingState(false);
  }
}

// ── Pin rendering ─────────────────────────────────────────────────────────────

function renderPins() {
  state.markerLayer.clearLayers();
  state.markerIndex.clear();

  state.districtData.forEach(record => {
    const latlng = CENTROIDS[record.district];
    if (!latlng) return;

    const riskMeta  = RISK[record.risk] || RISK.low;
    const isSelected = normalizeDistrict(record.district) === normalizeDistrict(state.selectedDistrict || '');
    const marker    = L.marker(latlng, {
      icon:        buildPinIcon(riskMeta, record, isSelected),
      zIndexOffset: record.risk === 'high' ? 1000 : record.risk === 'moderate' ? 500 : 0,
    });

    // Tooltip on hover
    marker.bindTooltip(
      `<div class="tt-name">${escapeHtml(record.district)}</div>
       <div class="tt-risk">Flood risk: <strong>${Number(record.risk_percent || 0).toFixed(0)}%</strong></div>
       <div class="tt-river">${formatRiverName(record.river)} River</div>`,
      { direction: 'top', offset: [0, -44], className: 'district-tooltip' }
    );

    marker.on('click', () => openPanel(record));

    state.markerLayer.addLayer(marker);
    state.markerIndex.set(normalizeDistrict(record.district), marker);
  });
}

// Build a Google Maps-style teardrop pin as an SVG DivIcon
function buildPinIcon(riskMeta, record, selected) {
  const c     = riskMeta.color;
  const pct   = Number(record.risk_percent || 0).toFixed(0);
  const size  = selected ? 52 : 44;        // selected pin is bigger
  const ring  = selected ? 3 : 2;           // ring thickness
  const pulse = record.risk === 'high';     // pulsing ring for high risk

  const svgPin = `
    <svg xmlns="http://www.w3.org/2000/svg" width="${size}" height="${size * 1.35}"
         viewBox="0 0 44 59" style="filter:drop-shadow(0 3px 6px ${riskMeta.shadow});">
      <!-- Teardrop body -->
      <path d="M22 2 C12 2 4 10 4 20 C4 32 22 56 22 56 C22 56 40 32 40 20 C40 10 32 2 22 2 Z"
            fill="${c}" stroke="${selected ? '#fff' : 'rgba(0,0,0,0.18)'}"
            stroke-width="${ring}"/>
      <!-- Inner circle -->
      <circle cx="22" cy="20" r="10" fill="rgba(255,255,255,0.92)"/>
      <!-- Risk % text -->
      <text x="22" y="24" text-anchor="middle"
            font-family="'Manrope',sans-serif" font-size="9" font-weight="800"
            fill="${c}">${pct}%</text>
    </svg>
    ${pulse ? `<div class="pin-pulse" style="background:${c}"></div>` : ''}
  `;

  return L.divIcon({
    html:        svgPin,
    className:   'district-pin-icon',
    iconSize:    [size, size * 1.35],
    iconAnchor:  [size / 2, size * 1.35],   // tip of teardrop anchors to lat/lng
    tooltipAnchor: [0, -(size * 1.35)],
    popupAnchor: [0, -(size * 1.35)],
  });
}

// Re-render all pins (called on selection change to update selected pin size)
function refreshPins() {
  renderPins();
}

// ── Panel ─────────────────────────────────────────────────────────────────────

function openPanel(record) {
  state.selectedDistrict = record.district;
  document.body.classList.add('panel-open');
  document.getElementById('sidePanel').setAttribute('aria-hidden', 'false');
  document.getElementById('panelEmpty').hidden = true;
  document.getElementById('panelBody').hidden  = false;

  const riskMeta = RISK[record.risk] || RISK.low;
  const lastRain = record.last_rainfall || {};

  document.getElementById('panelDistrictName').textContent = record.district;
  document.getElementById('panelSubtitle').textContent     = `${riskMeta.label} flood risk · district-level monitoring`;

  const pill = document.getElementById('panelRiskLabel');
  pill.textContent = riskMeta.label;
  pill.className   = `risk-pill ${record.risk}`;

  document.getElementById('panelRiskPercent').textContent = `${Number(record.risk_percent || 0).toFixed(0)}% flood risk`;
  document.getElementById('panelRiverMeta').textContent   = `Primary river: ${formatRiverName(record.river)}`;

  updatePointer('currentPointer',  record.current_level_pct);
  updatePointer('forecastPointer', record.predicted_level_3d_pct);
  document.getElementById('currentLevelValue').textContent  = `${formatMeters(record.river_level)} m`;
  document.getElementById('forecastLevelValue').textContent = `${formatMeters(record.predicted_level_3d)} m`;

  document.getElementById('rainDate').textContent   = lastRain.date      || '-';
  document.getElementById('rainTime').textContent   = lastRain.time      || '-';
  document.getElementById('rainAmount').textContent = `${formatMillimeters(lastRain.amount_mm)} mm`;

  renderForecast(record.forecast || []);

  // Zoom map to district, offset for panel width
  const latlng = CENTROIDS[record.district];
  if (latlng) {
    state.map.flyTo(latlng, 9.5, { duration: 0.6,
      paddingBottomRight: window.innerWidth > 900 ? [420, 0] : [0, 0] });
  }

  // Redraw pins so selected one enlarges
  refreshPins();

  document.getElementById('sidePanel').scrollTop = 0;
  document.getElementById('panelBody').scrollTop = 0;
}

function closePanel() {
  state.selectedDistrict = null;
  document.body.classList.remove('panel-open');
  document.getElementById('sidePanel').setAttribute('aria-hidden', 'true');
  document.getElementById('panelEmpty').hidden = false;
  document.getElementById('panelBody').hidden  = true;

  // Fly back to full Assam view
  state.map.flyTo(ASSAM_VIEW.center, ASSAM_VIEW.minZoom, { duration: 0.5 });
  refreshPins();
}

function syncSelectedDistrict() {
  if (!state.selectedDistrict) return;
  const record = state.districtData.find(d => normalizeDistrict(d.district) === normalizeDistrict(state.selectedDistrict));
  if (!record) { closePanel(); return; }
  openPanel(record);
}

// ── Stats ─────────────────────────────────────────────────────────────────────

function renderStats() {
  const high     = state.districtData.filter(d => d.risk === 'high').length;
  const moderate = state.districtData.filter(d => d.risk === 'moderate').length;
  const low      = state.districtData.filter(d => d.risk === 'low').length;
  document.getElementById('districtCount').textContent     = state.districtData.length || 35;
  document.getElementById('highRiskCount').textContent     = high;
  document.getElementById('moderateRiskCount').textContent = moderate;
  document.getElementById('lowRiskCount').textContent      = low;
}

function renderTimestamp() {
  const first = state.districtData.find(d => d.updated_at);
  const ts = first ? new Date(first.updated_at.replace(' ', 'T')) : new Date();
  document.getElementById('appTimestamp').textContent = ts.toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: true,
  });
}

function renderForecast(forecast) {
  const container = document.getElementById('forecastGrid');
  container.innerHTML = '';
  forecast.forEach(day => {
    const risk = classifyRisk(day.pct);
    const item = document.createElement('div');
    item.className = `forecast-card ${risk}`;
    item.innerHTML = `<span class="forecast-date">${escapeHtml(day.date)}</span>
                      <strong>${formatMeters(day.level)} m</strong>
                      <span class="forecast-risk">${RISK[risk].label}</span>`;
    container.appendChild(item);
  });
}

function updatePointer(id, value) {
  const el = document.getElementById(id);
  if (el) el.style.left = `${Math.max(4, Math.min(96, Number(value || 0)))}%`;
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function normalizeDistrict(name) {
  return String(name || '').toLowerCase().replace(/metropolitan/g, 'metro').replace(/[^a-z0-9]+/g, ' ').trim();
}

function classifyRisk(pct) {
  const v = Number(pct || 0);
  return v > 70 ? 'high' : v > 40 ? 'moderate' : 'low';
}

function formatRiverName(name) {
  const v = String(name || '-').replace(/_/g, ' ');
  return v.charAt(0).toUpperCase() + v.slice(1);
}

function formatMeters(v)      { return Number(v || 0).toFixed(2); }
function formatMillimeters(v) { return Number(v || 0).toFixed(1); }

function escapeHtml(v) {
  return String(v || '').replaceAll('&','&amp;').replaceAll('<','&lt;')
    .replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#39;');
}

function setLoadingState(isLoading) {
  const btn   = document.getElementById('refreshButton');
  const label = document.getElementById('refreshLabel');
  btn.disabled = isLoading;
  btn.classList.toggle('is-loading', isLoading);
  label.textContent = isLoading ? 'Refreshing...' : 'Refresh';
}

function showOverlay(msg) {
  const el = document.getElementById('loadingOverlay');
  if (!el) return;
  const copy = el.querySelector('.loading-copy');
  if (copy && msg) copy.textContent = msg;
  el.removeAttribute('hidden');
  el.style.display = '';
}

function hideOverlay() {
  const el = document.getElementById('loadingOverlay');
  if (!el) return;
  el.setAttribute('hidden', '');
  el.style.display = 'none';
}

async function fetchJson(url) {
  console.log('[FloodSense] fetchJson', url);
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

async function fetchDistricts() {
  let attempts = 0;
  while (true) {
    console.log('[FloodSense] API CALL TRIGGERED /api/districts');
    const res  = await fetch('/api/districts', { cache: 'no-store' });
    const data = await res.json();
    if (res.status === 202 && data.training) {
      attempts++;
      showOverlay(attempts === 1
        ? 'ML models training… ~10 seconds on first load.'
        : `Still training… (${attempts * 3}s elapsed)`);
      await new Promise(r => setTimeout(r, 3000));
      continue;
    }
    return data;
  }
}
