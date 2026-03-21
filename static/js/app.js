/* FloodSense Assam — Main Application JS */

'use strict';

// ── State ────────────────────────────────────────────────────────────
const state = {
  districts: [],
  metrics: null,
  historical: null,
  selectedDistrict: null,
  map: null,
  geojsonLayer: null,
  charts: {},
};

const RISK_COLORS = { high: '#e05252', moderate: '#e8973a', low: '#4caf82' };
const RISK_IDX    = { low: 0, moderate: 1, high: 2 };

// ── Boot ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  initTabs();
  loadAll();
});

async function loadAll() {
  setStatus('loading', 'Training models…');
  try {
    const [distRes, metRes, histRes] = await Promise.all([
      fetch('/api/districts'),
      fetch('/api/metrics'),
      fetch('/api/historical'),
    ]);
    state.districts  = (await distRes.json()).districts || [];
    state.metrics    = await metRes.json();
    state.historical = await histRes.json();

    renderMap();
    renderSidebar();
    renderMetrics();
    renderCharts();
    document.getElementById('updateTime').textContent = new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' });
    setStatus('ready', 'Models ready');
  } catch (err) {
    console.error(err);
    setStatus('loading', 'Error — retrying…');
    setTimeout(loadAll, 3000);
  }
}

function setStatus(type, text) {
  const dot  = document.querySelector('.status-dot');
  const span = document.getElementById('statusText');
  dot.className = 'status-dot ' + type;
  span.textContent = text;
}

// ── Leaflet Map ───────────────────────────────────────────────────────
function initMap() {
  state.map = L.map('map', { zoomControl: true, attributionControl: false });
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 11,
  }).addTo(state.map);
}

function renderMap() {
  const riskByDistrict = {};
  state.districts.forEach(d => { riskByDistrict[d.district] = d; });

  fetch('/static/data/assam_districts.geojson')
    .then(r => r.json())
    .then(geojson => {
      if (state.geojsonLayer) state.geojsonLayer.remove();

      state.geojsonLayer = L.geoJSON(geojson, {
        style: feature => {
          const name = feature.properties.district || feature.properties.DISTRICT || feature.properties.NAME_2 || '';
          const d = findDistrict(name, riskByDistrict);
          const risk = d ? d.risk : 'low';
          return {
            fillColor:   RISK_COLORS[risk],
            fillOpacity: risk === 'high' ? 0.55 : risk === 'moderate' ? 0.42 : 0.22,
            color:       RISK_COLORS[risk],
            weight:      1,
            opacity:     0.5,
          };
        },
        onEachFeature: (feature, layer) => {
          const name = feature.properties.district || feature.properties.DISTRICT || feature.properties.NAME_2 || '';
          const d = findDistrict(name, riskByDistrict);

          layer.on({
            mouseover(e) {
              e.target.setStyle({ weight: 2, fillOpacity: 0.75 });
              if (d) {
                e.target.bindTooltip(makeTooltip(d), {
                  className: 'leaflet-tooltip-dark', sticky: true
                }).openTooltip();
              }
            },
            mouseout(e) {
              state.geojsonLayer.resetStyle(e.target);
              e.target.closeTooltip();
            },
            click() {
              if (d) selectDistrict(d);
            },
          });
        },
      }).addTo(state.map);

      state.map.fitBounds(state.geojsonLayer.getBounds(), { padding: [10, 10] });
    });
}

function findDistrict(geoName, riskByDistrict) {
  if (!geoName) return null;
  const clean = geoName.trim();
  if (riskByDistrict[clean]) return riskByDistrict[clean];
  // Fuzzy: try partial match
  for (const key of Object.keys(riskByDistrict)) {
    if (clean.toLowerCase().includes(key.toLowerCase()) ||
        key.toLowerCase().includes(clean.toLowerCase())) {
      return riskByDistrict[key];
    }
  }
  return null;
}

function makeTooltip(d) {
  return `<div class="district-popup">
    <div class="dp-name">${d.district}</div>
    <div class="dp-river">${d.river || '—'} River</div>
    <div class="dp-risk ${d.risk}">${d.risk}</div>
  </div>`;
}

// ── Sidebar ───────────────────────────────────────────────────────────
function renderSidebar() {
  const high = state.districts.filter(d => d.risk === 'high').length;
  const mod  = state.districts.filter(d => d.risk === 'moderate').length;
  const low  = state.districts.filter(d => d.risk === 'low').length;

  document.getElementById('highCount').textContent = high;
  document.getElementById('modCount').textContent  = mod;
  document.getElementById('lowCount').textContent  = low;

  // Alert list: high first, then moderate
  const alerts = state.districts
    .filter(d => d.risk !== 'low')
    .sort((a, b) => RISK_IDX[b.risk] - RISK_IDX[a.risk]);

  const list = document.getElementById('alertList');
  list.innerHTML = '';

  if (alerts.length === 0) {
    list.innerHTML = '<div style="font-size:11px;color:var(--text-3);padding:4px 0">No active alerts</div>';
    return;
  }

  alerts.forEach(d => {
    const el = document.createElement('div');
    el.className = 'alert-item';
    el.innerHTML = `
      <div class="alert-dot ${d.risk}"></div>
      <div class="alert-district">${d.district}</div>
      <div class="alert-pct">${d.pct_to_danger || 0}%</div>
    `;
    el.addEventListener('click', () => selectDistrict(d));
    list.appendChild(el);
  });
}

// ── District detail ───────────────────────────────────────────────────
function selectDistrict(d) {
  state.selectedDistrict = d;

  // Highlight in alert list
  document.querySelectorAll('.alert-item').forEach(el => el.classList.remove('selected'));
  document.querySelectorAll('.alert-item').forEach(el => {
    if (el.querySelector('.alert-district')?.textContent === d.district) {
      el.classList.add('selected');
    }
  });

  document.getElementById('detailEmpty').style.display   = 'none';
  document.getElementById('detailContent').style.display = 'block';

  // Header
  document.getElementById('detailDistrict').textContent = d.district;
  document.getElementById('detailRiver').textContent = `${(d.river || '').replace(/_/g, ' ')} River`;
  const badge = document.getElementById('detailRiskBadge');
  badge.textContent  = d.risk;
  badge.className    = `detail-risk-badge ${d.risk}`;

  // Gauge
  const thr = { min: 0, warning: d.warning_level, danger: d.danger_level };
  const pct = Math.min(100, d.pct_to_danger || 0);
  const fill = document.getElementById('gaugeFill');
  fill.style.width = pct + '%';
  fill.className = `gauge-fill ${d.risk}`;

  // Warning/danger lines
  const span = thr.danger - 0; // relative
  if (thr.warning && thr.danger) {
    const warnPct = ((thr.warning - d.danger_level * 0.92) / (thr.danger - d.danger_level * 0.92) * 100);
    document.getElementById('gaugeWarning').style.left = '75%';
    document.getElementById('gaugeDangerLine').style.left = '92%';
  }

  document.getElementById('gaugeMin').textContent = d.warning_level ? (d.warning_level - 6).toFixed(0) : '—';
  document.getElementById('gaugeCur').textContent = (d.river_level || 0).toFixed(2) + 'm';
  document.getElementById('gaugeMax').textContent = d.danger_level ? (d.danger_level + 1).toFixed(0) : '—';
  document.getElementById('gaugeWarn').textContent = d.warning_level || '—';
  document.getElementById('gaugeDanger').textContent = d.danger_level || '—';

  // Stats
  document.getElementById('statRain3').textContent = (d.rain_3d || 0).toFixed(1);
  const delta = d.level_delta || 0;
  document.getElementById('statDelta').textContent = (delta > 0 ? '+' : '') + delta.toFixed(2) + 'm';
  document.getElementById('statConfidence').textContent = ((d.confidence || 0) * 100).toFixed(0) + '%';

  // SHAP features
  const shapList = document.getElementById('shapList');
  shapList.innerHTML = '';
  (d.shap_features || []).forEach(f => {
    const pctW = Math.min(50, Math.abs(f.contribution) * 200);
    shapList.innerHTML += `
      <div class="shap-item">
        <div class="shap-label" title="${f.label}">${f.label}</div>
        <div class="shap-bar-wrap">
          <div class="shap-bar-bg">
            <div class="shap-bar-fill ${f.direction}" style="width:${pctW}%"></div>
          </div>
        </div>
        <div class="shap-val">${f.direction === 'up' ? '+' : '−'}${Math.abs(f.contribution * 100).toFixed(0)}%</div>
      </div>`;
  });

  // Crop estimate
  const cropSec = document.getElementById('cropSection');
  if (d.crop_ha && d.crop_ha > 0 && d.risk !== 'low') {
    cropSec.style.display = 'block';
    document.getElementById('cropVal').textContent = d.crop_ha.toLocaleString('en-IN');
  } else {
    cropSec.style.display = 'none';
  }

  // Forecast
  const forecastRow = document.getElementById('forecastRow');
  forecastRow.innerHTML = '';
  (d.forecast || []).forEach(f => {
    const pctClass = f.pct >= 90 ? 'high' : f.pct >= 65 ? 'moderate' : 'low';
    forecastRow.innerHTML += `
      <div class="forecast-item">
        <div class="forecast-date">${f.date}</div>
        <div class="forecast-level">${f.level}m</div>
        <div class="forecast-pct ${pctClass}">${f.pct.toFixed(0)}%</div>
      </div>`;
  });

  // Probability bars
  const probList = document.getElementById('probList');
  const proba = d.probability || [1, 0, 0];
  const labels = ['Low', 'Moderate', 'High'];
  const keys   = ['low', 'moderate', 'high'];
  probList.innerHTML = '';
  labels.forEach((label, i) => {
    const pct = (proba[i] * 100).toFixed(1);
    probList.innerHTML += `
      <div class="prob-item">
        <div class="prob-label">${label}</div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill ${keys[i]}" style="width:${pct}%"></div>
        </div>
        <div class="prob-pct">${pct}%</div>
      </div>`;
  });
}

// ── Metrics ───────────────────────────────────────────────────────────
function renderMetrics() {
  const m = state.metrics;
  if (!m) return;
  const gb = m.gradient_boosting || {};
  document.getElementById('perfF1').textContent        = gb.f1_flood || '—';
  document.getElementById('perfPrecision').textContent = gb.precision || '—';
  document.getElementById('perfRecall').textContent    = gb.recall || '—';
  document.getElementById('perfMacro').textContent     = gb.f1_macro || '—';
}

// ── Charts ────────────────────────────────────────────────────────────
function renderCharts() {
  renderFeatureChart();
  renderHistoricalChart();
  renderModelCompare();
}

const CHART_DEFAULTS = {
  color: '#f0ede8',
  borderColor: 'rgba(255,255,255,0.07)',
  gridColor: 'rgba(255,255,255,0.05)',
  font: { family: "'DM Sans', sans-serif", size: 11 },
};

function renderFeatureChart() {
  const fi = (state.metrics?.feature_importance || []).slice(0, 8);
  if (!fi.length) return;

  const ctx = document.getElementById('chartFeatures').getContext('2d');
  if (state.charts.features) state.charts.features.destroy();

  state.charts.features = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: fi.map(f => f.label),
      datasets: [{
        data: fi.map(f => +(f.importance * 100).toFixed(1)),
        backgroundColor: fi.map((_, i) =>
          i === 0 ? '#e8c547' : i === 1 ? '#e8c54799' : 'rgba(232,197,71,0.25)'
        ),
        borderWidth: 0,
        borderRadius: 3,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: {
        backgroundColor: '#1a1c1f',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#f0ede8',
        bodyColor: '#9a9690',
        callbacks: { label: ctx => ` ${ctx.parsed.x.toFixed(1)}% importance` },
      }},
      scales: {
        x: {
          ticks: { color: '#5c5a56', font: CHART_DEFAULTS.font },
          grid: { color: CHART_DEFAULTS.gridColor },
          border: { color: 'transparent' },
        },
        y: {
          ticks: { color: '#9a9690', font: { ...CHART_DEFAULTS.font, size: 10 } },
          grid: { display: false },
          border: { color: 'transparent' },
        },
      },
    },
  });
}

function renderHistoricalChart() {
  const h = state.historical;
  if (!h) return;

  const ctx = document.getElementById('chartHistorical').getContext('2d');
  if (state.charts.historical) state.charts.historical.destroy();

  state.charts.historical = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: h.years,
      datasets: [
        {
          label: 'People affected (M)',
          data: h.people_affected.map(v => +(v / 1e6).toFixed(2)),
          backgroundColor: 'rgba(224,82,82,0.6)',
          borderWidth: 0,
          borderRadius: 3,
          yAxisID: 'y',
        },
        {
          label: 'Crop area (× 10k ha)',
          data: h.crop_area_ha.map(v => +(v / 10000).toFixed(1)),
          backgroundColor: 'rgba(232,151,58,0.5)',
          borderWidth: 0,
          borderRadius: 3,
          yAxisID: 'y2',
          type: 'line',
          borderColor: '#e8973a',
          pointBackgroundColor: '#e8973a',
          pointRadius: 3,
          tension: 0.4,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: 'index' },
      plugins: {
        legend: {
          labels: { color: '#9a9690', font: CHART_DEFAULTS.font, boxWidth: 10 },
          position: 'top',
          align: 'end',
        },
        tooltip: {
          backgroundColor: '#1a1c1f',
          borderColor: 'rgba(255,255,255,0.1)',
          borderWidth: 1,
          titleColor: '#f0ede8',
          bodyColor: '#9a9690',
        },
      },
      scales: {
        x: {
          ticks: { color: '#5c5a56', font: CHART_DEFAULTS.font },
          grid: { color: CHART_DEFAULTS.gridColor },
          border: { color: 'transparent' },
        },
        y: {
          ticks: { color: '#9a9690', font: CHART_DEFAULTS.font },
          grid: { color: CHART_DEFAULTS.gridColor },
          border: { color: 'transparent' },
        },
        y2: {
          position: 'right',
          ticks: { color: '#9a9690', font: CHART_DEFAULTS.font },
          grid: { display: false },
          border: { color: 'transparent' },
        },
      },
    },
  });
}

function renderModelCompare() {
  const m = state.metrics;
  if (!m) return;

  const gb   = m.gradient_boosting || {};
  const base = m.threshold_baseline || {};

  document.getElementById('modelCompare').innerHTML = `
    <div class="compare-card">
      <div class="compare-name">Gradient Boosting</div>
      <div class="compare-metrics">
        <div class="cm-item"><div class="cm-val">${gb.f1_flood || '—'}</div><div class="cm-label">F1 Flood</div></div>
        <div class="cm-item"><div class="cm-val">${gb.f1_macro || '—'}</div><div class="cm-label">F1 Macro</div></div>
        <div class="cm-item"><div class="cm-val">${gb.n_train ? (gb.n_train/1000).toFixed(1)+'k' : '—'}</div><div class="cm-label">Train size</div></div>
      </div>
    </div>
    <div class="compare-card baseline">
      <div class="compare-name">Threshold Baseline</div>
      <div class="compare-metrics">
        <div class="cm-item"><div class="cm-val">${base.f1_flood || '—'}</div><div class="cm-label">F1 Flood</div></div>
        <div class="cm-item"><div class="cm-val">${base.f1_macro || '—'}</div><div class="cm-label">F1 Macro</div></div>
        <div class="cm-item"><div class="cm-val">—</div><div class="cm-label">No training</div></div>
      </div>
    </div>`;
}

// ── Tabs ──────────────────────────────────────────────────────────────
function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + tab).classList.add('active');

      // Re-render chart on tab switch (fixes canvas sizing)
      if (tab === 'features' && state.charts.features) {
        state.charts.features.resize();
      }
      if (tab === 'historical' && state.charts.historical) {
        state.charts.historical.resize();
      }
    });
  });
}
