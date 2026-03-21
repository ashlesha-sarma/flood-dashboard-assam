'use strict';

// ── FIX 1: Do NOT call L.latLngBounds() at parse time.
// Leaflet isn't loaded yet when this file is parsed.
// Use plain arrays everywhere — Leaflet accepts [[lat,lng],[lat,lng]] directly.
const ASSAM_VIEW = {
  center:  [26.2006, 92.9376],
  bounds:  [[24.0, 89.45], [28.45, 96.15]], // plain array, NOT L.latLngBounds()
  minZoom: 7,
  maxZoom: 11.5,
};

const RISK_STYLES = {
  low:      { fill: '#2ecc71', stroke: '#157347', text: 'Low' },
  moderate: { fill: '#ffcc00', stroke: '#b38700', text: 'Moderate' },
  high:     { fill: '#e31a1c', stroke: '#991b1b', text: 'High' },
};

const state = {
  map:              null,
  geojsonData:      null,
  geojsonLayer:     null,
  districtData:     [],
  layerIndex:       new Map(),
  selectedDistrict: null,
  loading:          false,
};

document.addEventListener('DOMContentLoaded', () => {
  initMap();
  bindUI();
  loadDashboardData();
});

function bindUI() {
  document.getElementById('refreshButton').addEventListener('click', () => {
    loadDashboardData({ forceRefresh: true });
  });
  document.getElementById('panelClose').addEventListener('click', closePanel);
  document.getElementById('panelScrim').addEventListener('click', closePanel);
}

function initMap() {
  state.map = L.map('map', {
    zoomControl:         true,
    attributionControl:  false,
    minZoom:             ASSAM_VIEW.minZoom,
    maxZoom:             ASSAM_VIEW.maxZoom,
    zoomSnap:            0.25,
    zoomDelta:           0.25,
    worldCopyJump:       false,
    maxBounds:           ASSAM_VIEW.bounds,   // Leaflet accepts plain array here
    maxBoundsViscosity:  1.0,
    preferCanvas:        true,
  }).setView(ASSAM_VIEW.center, ASSAM_VIEW.minZoom);

  L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    subdomains: 'abcd',
    maxZoom:    ASSAM_VIEW.maxZoom,
    noWrap:     true,
    bounds:     ASSAM_VIEW.bounds,
  }).addTo(state.map);

  state.map.zoomControl.setPosition('bottomleft');
  state.map.on('drag', () => {
    state.map.panInsideBounds(ASSAM_VIEW.bounds, { animate: false });
  });
}

async function loadDashboardData(options = {}) {
  if (state.loading) return;

  state.loading = true;
  setLoadingState(true);
  showOverlay('Loading district risk data…');
  setMapStatus(options.forceRefresh ? 'Refreshing district data...' : 'Loading latest district data...');

  try {
    // ── FIX 2: Fetch GeoJSON and ML data in parallel
    const requests = [fetchDistricts()];  // polls until training done
    if (!state.geojsonData) {
      requests.push(fetchJson('/static/data/assam_districts.geojson'));
    }

    const [districtPayload, geojsonPayload] = await Promise.all(requests);
    state.districtData = Array.isArray(districtPayload.districts) ? districtPayload.districts : [];

    if (geojsonPayload) {
      state.geojsonData = geojsonPayload;
    }

    renderStats();
    renderTimestamp();
    renderMapLayer();
    syncSelectedDistrict();

    setMapStatus('Hover a district for quick risk. Click for detailed analytics.');
    hideOverlay();   // ── FIX 3: explicit hide with both hidden + display:none
  } catch (error) {
    console.error('loadDashboardData error:', error);
    hideOverlay();
    setMapStatus('Could not load data. Check the server is running, then click Refresh.');
  } finally {
    state.loading = false;
    setLoadingState(false);
  }
}

// ── FIX 3: Overlay helpers that work regardless of CSS specificity conflicts.
// Using both `hidden` attribute AND explicit style.display so there's no
// CSS override fight (the old code used only `hidden` but CSS had display:grid).
function showOverlay(message) {
  const el = document.getElementById('loadingOverlay');
  if (!el) return;
  const copy = el.querySelector('.loading-copy');
  if (copy && message) copy.textContent = message;
  el.removeAttribute('hidden');
  el.style.display = '';          // let CSS rule (display:grid) take over
}

function hideOverlay() {
  const el = document.getElementById('loadingOverlay');
  if (!el) return;
  el.setAttribute('hidden', '');
  el.style.display = 'none';     // belt-and-suspenders: force hide regardless of CSS
}

function renderStats() {
  const high     = state.districtData.filter(d => d.risk === 'high').length;
  const moderate = state.districtData.filter(d => d.risk === 'moderate').length;
  const low      = state.districtData.filter(d => d.risk === 'low').length;

  document.getElementById('districtCount').textContent      = state.districtData.length || 33;
  document.getElementById('highRiskCount').textContent      = high;
  document.getElementById('moderateRiskCount').textContent  = moderate;
  document.getElementById('lowRiskCount').textContent       = low;
}

function renderTimestamp() {
  const first = state.districtData.find(d => d.updated_at);
  const ts = first ? new Date(first.updated_at.replace(' ', 'T')) : new Date();
  document.getElementById('appTimestamp').textContent = ts.toLocaleString('en-IN', {
    day: '2-digit', month: 'short', year: 'numeric',
    hour: '2-digit', minute: '2-digit', hour12: true,
  });
}

function renderMapLayer() {
  if (!state.geojsonData) return;

  state.layerIndex.clear();
  if (state.geojsonLayer) {
    state.geojsonLayer.remove();
    state.geojsonLayer = null;
  }

  state.geojsonLayer = L.geoJSON(state.geojsonData, {
    style:          feature => districtStyle(getDistrictRecord(feature)),
    onEachFeature:  (feature, layer) => {
      const record       = getDistrictRecord(feature);
      const districtName = record ? record.district : getFeatureName(feature);

      layer.bindTooltip(
        tooltipMarkup(districtName, record ? record.risk_percent : 0),
        { sticky: true, direction: 'top', className: 'district-tooltip' }
      );

      layer.on('mouseover', () => {
        layer.setStyle(hoverDistrictStyle(record, districtName));
        layer.bringToFront();
      });
      layer.on('mouseout', () => {
        layer.setStyle(districtStyle(record, districtName));
      });
      layer.on('click', () => {
        if (!record) return;
        openPanel(record);
        focusDistrict(layer);
      });

      state.layerIndex.set(normalizeDistrictName(districtName), layer);
    },
  }).addTo(state.map);

  const geoBounds = state.geojsonLayer.getBounds().pad(0.03);
  state.map.setMaxBounds(geoBounds);
  state.map.fitBounds(geoBounds, { padding: [18, 18] });
}

function districtStyle(record, fallbackName = '') {
  const riskKey  = record ? record.risk : 'low';
  const palette  = RISK_STYLES[riskKey] || RISK_STYLES.low;
  const selected = normalizeDistrictName(record ? record.district : fallbackName) ===
                   normalizeDistrictName(state.selectedDistrict || '');
  return {
    fillColor:   palette.fill,
    fillOpacity: selected ? 0.62 : 0.5,
    color:       selected ? '#0f172a' : palette.stroke,
    weight:      selected ? 2.8 : 1.15,
    opacity:     0.95,
    lineJoin:    'round',
  };
}

function hoverDistrictStyle(record, fallbackName = '') {
  const base = districtStyle(record, fallbackName);
  return { ...base, fillOpacity: 0.68, weight: Math.max(base.weight, 2.4), color: '#0f172a' };
}

function tooltipMarkup(name, riskPercent) {
  return `<div class="tooltip-name">${escapeHtml(name)}</div>
          <div class="tooltip-risk">Flood risk: ${Number(riskPercent || 0).toFixed(0)}%</div>`;
}

function openPanel(record) {
  state.selectedDistrict = record.district;
  document.body.classList.add('panel-open');
  document.getElementById('sidePanel').setAttribute('aria-hidden', 'false');
  document.getElementById('panelEmpty').hidden = true;
  document.getElementById('panelBody').hidden  = false;

  const riskKey     = record.risk || classifyRisk(record.risk_percent);
  const riskMeta    = RISK_STYLES[riskKey] || RISK_STYLES.low;
  const lastRain    = record.last_rainfall || {};

  document.getElementById('panelDistrictName').textContent  = record.district;
  document.getElementById('panelSubtitle').textContent      = `${riskMeta.text} flood risk · district-level monitoring`;

  const riskLabel = document.getElementById('panelRiskLabel');
  riskLabel.textContent = riskMeta.text;
  riskLabel.className   = `risk-pill ${riskKey}`;

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
  refreshLayerStyles();
  document.getElementById('sidePanel').scrollTop  = 0;
  document.getElementById('panelBody').scrollTop  = 0;
}

function closePanel() {
  state.selectedDistrict = null;
  document.body.classList.remove('panel-open');
  document.getElementById('sidePanel').setAttribute('aria-hidden', 'true');
  document.getElementById('panelEmpty').hidden = false;
  document.getElementById('panelBody').hidden  = true;
  refreshLayerStyles();
  if (state.geojsonLayer) {
    state.map.fitBounds(state.geojsonLayer.getBounds().pad(0.03), { padding: [18, 18] });
  }
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
                      <span class="forecast-risk">${RISK_STYLES[risk].text}</span>`;
    container.appendChild(item);
  });
}

function updatePointer(elementId, value) {
  const pointer = document.getElementById(elementId);
  pointer.style.left = `${Math.max(4, Math.min(96, Number(value || 0)))}%`;
}

function focusDistrict(layer) {
  state.map.fitBounds(layer.getBounds().pad(0.35), {
    paddingTopLeft:     [24, 24],
    paddingBottomRight: window.innerWidth > 900 ? [420, 24] : [24, 24],
    maxZoom:            9.75,
  });
}

function refreshLayerStyles() {
  if (!state.geojsonLayer) return;
  state.geojsonLayer.eachLayer(layer => {
    const record       = getDistrictRecord(layer.feature);
    const districtName = record ? record.district : getFeatureName(layer.feature);
    layer.setStyle(districtStyle(record, districtName));
  });
}

function syncSelectedDistrict() {
  if (!state.selectedDistrict) return;
  const record = findDistrictRecord(state.selectedDistrict);
  if (!record) { closePanel(); return; }
  openPanel(record);
  const layer = state.layerIndex.get(normalizeDistrictName(record.district));
  if (layer) focusDistrict(layer);
}

function getDistrictRecord(feature)  { return findDistrictRecord(getFeatureName(feature)); }
function getFeatureName(feature)      { return feature?.properties?.DISTRICT || feature?.properties?.district || ''; }

function findDistrictRecord(name) {
  const n = normalizeDistrictName(name);
  return state.districtData.find(d => normalizeDistrictName(d.district) === n) || null;
}

function normalizeDistrictName(name) {
  return String(name || '').toLowerCase()
    .replace(/metropolitan/g, 'metro')
    .replace(/[^a-z0-9]+/g, ' ').trim();
}

function classifyRisk(riskPercent) {
  const v = Number(riskPercent || 0);
  return v > 70 ? 'high' : v > 40 ? 'moderate' : 'low';
}

function formatRiverName(name) {
  const v = String(name || '-').replace(/_/g, ' ');
  return v.charAt(0).toUpperCase() + v.slice(1);
}

function formatMeters(value)      { return Number(value || 0).toFixed(2); }
function formatMillimeters(value) { return Number(value || 0).toFixed(1); }

function setLoadingState(isLoading) {
  const button = document.getElementById('refreshButton');
  const label  = document.getElementById('refreshLabel');
  button.disabled = isLoading;
  button.classList.toggle('is-loading', isLoading);
  label.textContent = isLoading ? 'Refreshing...' : 'Refresh';
}

function setMapStatus(text) {
  const el = document.getElementById('mapStatus');
  if (el) el.textContent = text;
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
  return res.json();
}

// ── FIX: poll /api/districts while server is still training (HTTP 202)
async function fetchDistricts() {
  let attempts = 0;
  while (true) {
    const res  = await fetch('/api/districts', { cache: 'no-store' });
    const data = await res.json();
    if (res.status === 202 && data.training) {
      attempts++;
      const msg = attempts === 1
        ? 'ML models training… this takes ~10 seconds on first load.'
        : `Still training… (${attempts * 3}s elapsed)`;
      showOverlay(msg);
      setMapStatus(msg);
      await new Promise(r => setTimeout(r, 3000)); // wait 3s then retry
      continue;
    }
    return data;
  }
}

function escapeHtml(value) {
  return String(value || '')
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;').replaceAll('"', '&quot;').replaceAll("'", '&#39;');
}