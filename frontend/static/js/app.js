
/**
 * BigV Safari — Wildlife Spotter for Tourists & Tour Guides
 * Features: Big Five tracker, Field Guide, Guide/Tourist mode,
 * Waze-style community map, GPS, Report to Ranger, animal facts.
 */

'use strict';

// ─── Config ───────────────────────────────────────────────────────────────────
const CONFIG = {
  wsUrl: `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`,
  analyzeInterval: 1500,
  maxSightings: 100,
  maxTickerItems: 5,
  frameQuality: 0.7,
  frameScale: 0.5,
  appName: 'BigV Safari',
  communityRefreshMs: 30000,
  nearbyRadiusKm: 15,
};

// ─── Animal knowledge base (mirrors backend) ──────────────────────────────────
const ANIMAL_DB = {
  lion:        { swahili:'Simba',       emoji:'🦁', bigFive:true,  habitat:'Open savanna, grasslands', diet:'Carnivore — wildebeest, zebra, buffalo', behavior:'Social prides of 10–15. Most active at dawn and dusk.', photoTip:'300mm+ lens. Stay in vehicle. Best at golden hour.', dangerNote:'Do not exit vehicle. Keep windows up if approached.' },
  leopard:     { swahili:'Chui',        emoji:'🐆', bigFive:true,  habitat:'Dense bush, riverine forest, rocky outcrops', diet:'Carnivore — impala, baboon', behavior:'Solitary and nocturnal. Often rests in trees with kills.', photoTip:'Scan tree branches. Patience required — rare sighting!', dangerNote:'Extremely agile. Never approach on foot.' },
  elephant:    { swahili:'Tembo',       emoji:'🐘', bigFive:true,  habitat:'Savanna, forest, bushveld', diet:'Herbivore — up to 300kg vegetation/day', behavior:'Matriarchal herds. Highly intelligent with excellent memory.', photoTip:'Wide angle for herds. Watch for mock charges.', dangerNote:'Back away slowly if ears spread and trunk raised.' },
  rhinoceros:  { swahili:'Kifaru',      emoji:'🦏', bigFive:true,  habitat:'Grassland, savanna, bushveld', diet:'Herbivore — grass (white), leaves (black)', behavior:'Mostly solitary. Poor eyesight but excellent hearing.', photoTip:'Extremely rare — report to ranger immediately.', dangerNote:'Stay 100m+ away. Report GPS location to ranger.' },
  buffalo:     { swahili:'Nyati',       emoji:'🐃', bigFive:true,  habitat:'Savanna, floodplains near water', diet:'Herbivore — grass', behavior:'Large herds. Old bulls ("dagga boys") are unpredictable.', photoTip:'Shoot from vehicle only. Old bulls are most dangerous.', dangerNote:'Never approach lone bulls on foot.' },
  cheetah:     { swahili:'Duma',        emoji:'🐆', bigFive:false, habitat:'Open grassland, semi-arid savanna', diet:'Carnivore — gazelle, impala, hare', behavior:'Diurnal hunter. Fastest land animal (110 km/h).', photoTip:'Best in open plains at dawn. Use fast shutter 1/1000s+.', dangerNote:'Generally not dangerous to vehicles.' },
  giraffe:     { swahili:'Twiga',       emoji:'🦒', bigFive:false, habitat:'Open woodland, savanna', diet:'Herbivore — acacia leaves', behavior:'Tallest animal. Vulnerable when drinking.', photoTip:'Silhouette against sunset. Wide angle for full body.', dangerNote:'Kick can be lethal — maintain distance.' },
  zebra:       { swahili:'Punda Milia', emoji:'🦓', bigFive:false, habitat:'Open grassland, savanna', diet:'Herbivore — grass', behavior:'Highly social herds. Each stripe pattern is unique.', photoTip:'Shoot in shade for best stripe detail.', dangerNote:'Generally safe to observe from vehicle.' },
  hippopotamus:{ swahili:'Kiboko',      emoji:'🦛', bigFive:false, habitat:'Rivers, lakes, wetlands', diet:'Herbivore — grazes at night', behavior:'Most dangerous animal in Africa. Territorial in water.', photoTip:'Dawn/dusk for yawning shots from safe riverbank.', dangerNote:'Never approach water edge on foot. Extremely dangerous.' },
  crocodile:   { swahili:'Mamba',       emoji:'🐊', bigFive:false, habitat:'Rivers, lakes, estuaries', diet:'Carnivore — fish, wildebeest at crossings', behavior:'Ambush predator. Can remain motionless for hours.', photoTip:'Telephoto from bank. River crossings are iconic.', dangerNote:'Never approach water edge. Can move at 17 km/h on land.' },
  wildebeest:  { swahili:'Nyumbu',      emoji:'🐃', bigFive:false, habitat:'Open grassland, savanna', diet:'Herbivore — grass', behavior:'Famous Great Migration (1.5M animals). Calving Jan–Feb.', photoTip:'River crossings — use 1/1000s+ shutter speed.', dangerNote:'Large herds attract predators — scan surroundings.' },
  gorilla:     { swahili:'Gorila',      emoji:'🦍', bigFive:false, habitat:'Mountain and lowland forest', diet:'Herbivore — leaves, stems, fruit', behavior:'Gentle giants. Silverback leads family group.', photoTip:'No flash. Stay 7m minimum. Move slowly.', dangerNote:'Trekking permit required. Follow guide instructions.' },
  chimpanzee:  { swahili:'Sokwe',       emoji:'🐒', bigFive:false, habitat:'Tropical forest', diet:'Omnivore — fruit, insects, small animals', behavior:'Highly intelligent. Tool use observed.', photoTip:'High ISO for forest light. Move slowly and quietly.', dangerNote:'Can be aggressive. Stay with guide at all times.' },
};

const BIG_FIVE_KEYS = ['lion','leopard','elephant','rhinoceros','buffalo'];

// ─── State ────────────────────────────────────────────────────────────────────
const state = {
  ws: null,
  clientId: `bigv-${Math.random().toString(36).slice(2, 9)}`,
  stream: null,
  analyzeTimer: null,
  isRunning: false,
  frameCount: 0,
  totalAnimals: 0,
  criticalAlerts: 0,
  alertCount: 0,
  startTime: null,
  uptimeTimer: null,
  sightings: [],
  lastFpsTime: Date.now(),
  fpsFrames: 0,
  currentFps: 0,
  deferredInstallPrompt: null,
  activeTab: 'camera',
  userLat: null,
  userLng: null,
  watchId: null,
  map: null,
  mapMarkers: [],
  communityTimer: null,
  bigFiveSeen: new Set(),
  mode: 'tourist',  // 'tourist' | 'guide'
  lastAlert: null,
  pinMode: false,
  pendingPinLat: null,
  pendingPinLng: null,
  manualPins: [],
};

// ─── DOM refs ─────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const video         = $('videoFeed');
const canvas        = $('overlayCanvas');
const ctx           = canvas.getContext('2d');
const overlay       = $('cameraOverlay');
const scanLine      = $('scanLine');
const startBtn      = $('startBtn');
const stopBtn       = $('stopBtn');
const uploadBtn     = $('uploadBtn');
const fileInput     = $('fileInput');
const cameraSelect  = $('cameraSelect');
const locationInput = $('locationInput');
const connStatus    = $('connectionStatus');
const alertTicker   = $('alertTicker');
const sightingsLog  = $('sightingsLog');
const agentText     = $('agentText');
const recsBox       = $('recsBox');
const recsList      = $('recsList');
const dangerFill    = $('dangerFill');
const dangerScore   = $('dangerScore');
const currentDet    = $('currentDetection');
const alertModal    = $('alertModal');
const installBtn    = $('installBtn');
const installBanner = $('installBanner');
const cameraDetBadge= $('cameraDetBadge');
const cameraDetText = $('cameraDetText');

// ─── PWA ──────────────────────────────────────────────────────────────────────
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/static/sw.js')
      .then(reg => console.log('[BigV SW] Registered:', reg.scope))
      .catch(err => console.warn('[BigV SW] Failed:', err));
  });
}

window.addEventListener('beforeinstallprompt', e => {
  e.preventDefault();
  state.deferredInstallPrompt = e;
  installBtn.style.display = 'block';
  installBanner.style.display = 'block';
});

window.addEventListener('appinstalled', () => {
  installBtn.style.display = 'none';
  installBanner.style.display = 'none';
  state.deferredInstallPrompt = null;
});

async function triggerInstall() {
  if (!state.deferredInstallPrompt) return;
  state.deferredInstallPrompt.prompt();
  const { outcome } = await state.deferredInstallPrompt.userChoice;
  if (outcome === 'accepted') {
    installBtn.style.display = 'none';
    installBanner.style.display = 'none';
  }
  state.deferredInstallPrompt = null;
}

function dismissInstallBanner() { installBanner.style.display = 'none'; }

// ─── Guide / Tourist Mode ─────────────────────────────────────────────────────
function setMode(mode) {
  state.mode = mode;
  $('btnTourist').classList.toggle('active', mode === 'tourist');
  $('btnGuide').classList.toggle('active', mode === 'guide');
  document.body.classList.toggle('guide-mode', mode === 'guide');
  // Guide mode shows more technical detail
  document.querySelectorAll('.guide-only').forEach(el => {
    el.style.display = mode === 'guide' ? '' : 'none';
  });
  document.querySelectorAll('.tourist-only').forEach(el => {
    el.style.display = mode === 'tourist' ? '' : 'none';
  });
}

// ─── GPS ──────────────────────────────────────────────────────────────────────
function startGPS() {
  if (!('geolocation' in navigator)) return;
  state.watchId = navigator.geolocation.watchPosition(
    pos => {
      state.userLat = pos.coords.latitude;
      state.userLng = pos.coords.longitude;
      const badge = $('gpsBadge');
      if (badge) {
        badge.textContent = `📍 ${state.userLat.toFixed(4)}, ${state.userLng.toFixed(4)}`;
        badge.style.display = 'inline-block';
      }
      if (state.map) centerMapOnUser();
    },
    err => console.warn('[BigV GPS]', err.message),
    { enableHighAccuracy: true, maximumAge: 10000, timeout: 15000 }
  );
}

function haversineKm(lat1, lng1, lat2, lng2) {
  const R = 6371;
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLng = (lng2 - lng1) * Math.PI / 180;
  const a = Math.sin(dLat/2)**2 +
            Math.cos(lat1 * Math.PI/180) * Math.cos(lat2 * Math.PI/180) *
            Math.sin(dLng/2)**2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
}

// ─── Big Five Tracker ─────────────────────────────────────────────────────────
function updateBigFive(seenList) {
  if (!seenList) return;
  let newSighting = false;
  seenList.forEach(animal => {
    const key = animal.toLowerCase();
    if (BIG_FIVE_KEYS.includes(key) && !state.bigFiveSeen.has(key)) {
      newSighting = true;
    }
    state.bigFiveSeen.add(key);
  });

  BIG_FIVE_KEYS.forEach(key => {
    const el = $(`bf-${key}`);
    if (el) {
      el.classList.toggle('seen', state.bigFiveSeen.has(key));
    }
  });

  const count = state.bigFiveSeen.size;
  $('statBigFive').textContent = `${count}/5`;
  const bd = $('statBigFiveD'); if (bd) bd.textContent = count;

  // Big Five complete celebration
  if (count === 5 && newSighting) {
    $('bigFiveModal').style.display = 'flex';
    if ('vibrate' in navigator) navigator.vibrate([200, 100, 200, 100, 200, 100, 500]);
  }
}

// ─── Field Guide ──────────────────────────────────────────────────────────────
function buildFieldGuide() {
  const bigFiveGrid = $('bigFiveGrid');
  const otherGrid = $('otherAnimalsGrid');
  if (!bigFiveGrid || !otherGrid) return;

  Object.entries(ANIMAL_DB).forEach(([key, info]) => {
    const card = createGuideCard(key, info);
    if (info.bigFive) {
      bigFiveGrid.appendChild(card);
    } else {
      otherGrid.appendChild(card);
    }
  });
}

function createGuideCard(key, info) {
  const card = document.createElement('div');
  card.className = `guide-card${info.bigFive ? ' big-five-card' : ''}`;
  card.dataset.animal = key;
  card.innerHTML = `
    <div class="guide-card-header">
      <span class="guide-emoji">${info.emoji}</span>
      <div>
        <div class="guide-name">${key.charAt(0).toUpperCase() + key.slice(1)}</div>
        <div class="guide-swahili">${info.swahili}</div>
      </div>
      ${info.bigFive ? '<span class="big-five-crown">🏆</span>' : ''}
    </div>
    <div class="guide-detail">
      <div class="guide-row"><span class="guide-label">🌍 Habitat</span><span>${info.habitat}</span></div>
      <div class="guide-row"><span class="guide-label">🍖 Diet</span><span>${info.diet}</span></div>
      <div class="guide-row"><span class="guide-label">🧠 Behavior</span><span>${info.behavior}</span></div>
      <div class="guide-row photo-tip"><span class="guide-label">📸 Photo tip</span><span>${info.photoTip}</span></div>
      <div class="guide-row danger-note"><span class="guide-label">⚠️ Safety</span><span>${info.dangerNote}</span></div>
    </div>
  `;
  return card;
}

function filterGuide(query) {
  const q = query.toLowerCase();
  document.querySelectorAll('.guide-card').forEach(card => {
    const animal = card.dataset.animal || '';
    const info = ANIMAL_DB[animal] || {};
    const match = !q ||
      animal.includes(q) ||
      (info.swahili || '').toLowerCase().includes(q) ||
      (info.habitat || '').toLowerCase().includes(q);
    card.style.display = match ? '' : 'none';
  });
}

// ─── Animal Fact Card ─────────────────────────────────────────────────────────
function showAnimalFacts(animalFacts) {
  const box = $('animalFactCard');
  if (!box || !animalFacts || animalFacts.length === 0) {
    if (box) box.style.display = 'none';
    return;
  }

  const fact = animalFacts[0]; // show first detected animal
  const info = ANIMAL_DB[fact.label] || {};
  const emoji = info.emoji || '🐾';
  const isBigFive = BIG_FIVE_KEYS.includes(fact.label);

  box.style.display = 'block';
  box.innerHTML = `
    <div class="fact-card-inner${isBigFive ? ' big-five-fact' : ''}">
      <div class="fact-header">
        <span class="fact-emoji">${emoji}</span>
        <div>
          <div class="fact-name">${fact.label.charAt(0).toUpperCase() + fact.label.slice(1)}
            ${isBigFive ? '<span class="big-five-crown">🏆 Big Five</span>' : ''}
          </div>
          ${fact.swahili ? `<div class="fact-swahili">Swahili: <em>${fact.swahili}</em></div>` : ''}
        </div>
      </div>
      ${fact.behavior ? `<div class="fact-row">🧠 ${fact.behavior}</div>` : ''}
      ${fact.photo_tip ? `<div class="fact-row photo">📸 ${fact.photo_tip}</div>` : ''}
      <button class="btn btn-sm" onclick="switchTab('guide')" style="margin-top:8px">
        📖 Full Field Guide
      </button>
    </div>
  `;
}

// ─── Report to Ranger ─────────────────────────────────────────────────────────
function reportToRanger() {
  const alert = state.lastAlert;
  const location = locationInput.value || 'Unknown location';
  const gps = state.userLat !== null
    ? `GPS: ${state.userLat.toFixed(5)}, ${state.userLng.toFixed(5)}`
    : 'GPS: unavailable';
  const animal = alert ? alert.animal : 'Wildlife';
  const priority = alert ? alert.priority?.toUpperCase() : 'HIGH';
  const time = new Date().toLocaleTimeString();

  const message = `📻 RANGER REPORT\n\nAnimal: ${animal}\nPriority: ${priority}\nLocation: ${location}\n${gps}\nTime: ${time}\n\nPlease respond to this sighting.`;

  // On mobile, try to open SMS or WhatsApp to ranger number
  // For now, show a formatted report dialog
  if (confirm(`Send this report to your ranger?\n\n${message}`)) {
    // Try to open WhatsApp (common in Africa)
    const encoded = encodeURIComponent(message);
    const waUrl = `https://wa.me/?text=${encoded}`;
    window.open(waUrl, '_blank');
  }
}

// ─── Waze-style Nearby Banner ─────────────────────────────────────────────────
function showNearbyAlert(sighting) {
  const banner = $('nearbyBanner');
  if (!banner) return;

  const animals = (sighting.animals || []).map(a => {
    const info = ANIMAL_DB[a.label?.toLowerCase()] || {};
    return `${info.emoji || '🐾'} ${a.label}`;
  }).join(', ');

  const level = sighting.alert_level || 'low';
  const dist = (state.userLat !== null && sighting.lat != null)
    ? ` · ${haversineKm(state.userLat, state.userLng, sighting.lat, sighting.lng).toFixed(1)} km away`
    : '';

  $('nearbyBannerText').innerHTML = `
    <strong>${animals || '🐾 Wildlife'}</strong> spotted by another vehicle${dist}
    <span style="color:var(--text-muted);font-size:0.75rem"> · ${sighting.location || 'Nearby'}</span>
  `;

  banner.className = `nearby-banner ${level}`;
  banner.style.display = 'flex';

  clearTimeout(banner._hideTimer);
  banner._hideTimer = setTimeout(() => { banner.style.display = 'none'; }, 8000);

  if ('vibrate' in navigator) navigator.vibrate([100, 50, 100]);
}

function dismissNearbyBanner() {
  const banner = $('nearbyBanner');
  if (banner) banner.style.display = 'none';
}

// ─── Community Map ────────────────────────────────────────────────────────────
function initMap() {
  if (state.map) return;
  const mapEl = $('communityMap');
  if (!mapEl) return;

  const defaultLat = state.userLat ?? -2.3333;
  const defaultLng = state.userLng ?? 34.8333;

  state.map = L.map('communityMap', { zoomControl: true }).setView([defaultLat, defaultLng], 10);

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 18,
  }).addTo(state.map);

  if (state.userLat !== null) {
    L.circleMarker([state.userLat, state.userLng], {
      radius: 10, color: '#f0883e', fillColor: '#f0883e', fillOpacity: 0.9,
    }).addTo(state.map).bindPopup('🚗 Your vehicle');
  }
}

function centerMapOnUser() {
  if (state.map && state.userLat !== null) {
    state.map.setView([state.userLat, state.userLng], state.map.getZoom());
  }
}

async function loadCommunitySightings() {
  try {
    const resp = await fetch('/community-sightings');
    const data = await resp.json();
    renderMapPins(data.sightings || []);
    updateCommunityCount(data.total || 0);
  } catch (e) {
    console.warn('[BigV Map] Failed:', e);
  }
}

function renderMapPins(sightings) {
  if (!state.map) return;
  state.mapMarkers.forEach(m => m.remove());
  state.mapMarkers = [];

  sightings.forEach(s => {
    if (s.lat == null || s.lng == null) return;
    addLivePinToMap(s);
  });

  if (state.mapMarkers.length > 0 && !state.userLat) {
    const group = L.featureGroup(state.mapMarkers);
    state.map.fitBounds(group.getBounds().pad(0.2));
  }
}

function addLivePinToMap(sighting) {
  if (!state.map || sighting.lat == null || sighting.lng == null) return;

  const level = sighting.alert_level || 'low';
  const color = { critical: '#f85149', high: '#f0883e', medium: '#d29922', low: '#3fb950' }[level] || '#8b949e';
  const animals = (sighting.animals || []).map(a => {
    const info = ANIMAL_DB[a.label?.toLowerCase()] || {};
    return `${info.emoji || '🐾'} ${a.label}`;
  }).join(', ');
  const isOwn = sighting.reporter_id === state.clientId;
  const timeAgo = formatTimeAgo(sighting.timestamp);

  const marker = L.circleMarker([sighting.lat, sighting.lng], {
    radius: level === 'critical' ? 14 : level === 'high' ? 11 : 8,
    color, fillColor: color,
    fillOpacity: isOwn ? 1.0 : 0.65,
    weight: isOwn ? 3 : 1.5,
  }).addTo(state.map);

  marker.bindPopup(`
    <div style="font-family:Inter,sans-serif;min-width:160px">
      <div style="font-weight:700;font-size:0.9rem;margin-bottom:4px">${animals || '🐾 Wildlife'}</div>
      <div style="color:#666;font-size:0.78rem">📍 ${sighting.location || 'Unknown'}</div>
      <div style="color:#666;font-size:0.78rem">⏱ ${timeAgo}</div>
      <div style="margin-top:4px;font-size:0.75rem;color:${color};font-weight:600;text-transform:uppercase">${level} priority</div>
      <div style="font-size:0.7rem;color:#888;margin-top:2px">${isOwn ? '🚗 Your vehicle' : '👥 Another vehicle'}</div>
    </div>
  `);

  state.mapMarkers.push(marker);
}

function updateCommunityCount(count) {

// ─── Manual Pin Location ──────────────────────────────────────────────────────
function togglePinMode() {
  state.pinMode = !state.pinMode;
  const btn = $('pinLocationBtn');
  const banner = $('pinModeBanner');
  
  if (state.pinMode) {
    btn.textContent = '❌ Cancel Pin';
    btn.classList.add('btn-danger');
    btn.classList.remove('btn-primary');
    if (banner) banner.style.display = 'flex';
    
    // Change cursor on map
    if (state.map) {
      state.map.getContainer().style.cursor = 'crosshair';
      state.map.once('click', handleMapClick);
    }
  } else {
    btn.textContent = '📍 Pin Sighting';
    btn.classList.remove('btn-danger');
    btn.classList.add('btn-primary');
    if (banner) banner.style.display = 'none';
    
    if (state.map) {
      state.map.getContainer().style.cursor = '';
      state.map.off('click', handleMapClick);
    }
  }
}

function cancelPinMode() {
  state.pinMode = false;
  const btn = $('pinLocationBtn');
  const banner = $('pinModeBanner');
  
  btn.textContent = '📍 Pin Sighting';
  btn.classList.remove('btn-danger');
  btn.classList.add('btn-primary');
  if (banner) banner.style.display = 'none';
  
  if (state.map) {
    state.map.getContainer().style.cursor = '';
    state.map.off('click', handleMapClick);
  }
}

function handleMapClick(e) {
  state.pendingPinLat = e.latlng.lat;
  state.pendingPinLng = e.latlng.lng;
  
  // Show modal to select animal
  $('pinAnimalModal').style.display = 'flex';
  
  // Reset pin mode
  cancelPinMode();
}

function cancelPinLocation() {
  $('pinAnimalModal').style.display = 'none';
  state.pendingPinLat = null;
  state.pendingPinLng = null;
}

async function confirmPinLocation() {
  const animal = $('pinAnimalSelect').value;
  const notes = $('pinNotes').value.trim();
  const priority = $('pinPriority').value;
  
  if (!animal || state.pendingPinLat === null) {
    alert('Please select an animal type');
    return;
  }
  
  const info = ANIMAL_DB[animal] || {};
  const location = locationInput.value || 'Safari';
  
  // Create manual pin object
  const manualPin = {
    id: `manual-${Date.now()}`,
    lat: state.pendingPinLat,
    lng: state.pendingPinLng,
    animal: animal,
    animals: [{ label: animal, confidence: 1.0 }],
    location: location,
    notes: notes,
    alert_level: priority,
    timestamp: new Date().toISOString(),
    reporter_id: state.clientId,
    manual: true,
  };
  
  // Add to local storage
  state.manualPins.push(manualPin);
  saveManualPins();
  
  // Add pin to map
  addManualPinToMap(manualPin);
  
  // Send to backend (optional - for sharing with other users)
  try {
    await fetch('/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image_b64: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', // 1x1 transparent pixel
        location: location,
        camera_id: state.clientId,
        lat: manualPin.lat,
        lng: manualPin.lng,
        manual_pin: true,
        animal_type: animal,
        notes: notes,
        priority: priority,
      }),
    });
  } catch (e) {
    console.warn('Failed to sync manual pin:', e);
  }
  
  // Close modal
  $('pinAnimalModal').style.display = 'none';
  $('pinNotes').value = '';
  state.pendingPinLat = null;
  state.pendingPinLng = null;
  
  // Show success message
  showToast(`📍 ${info.emoji || '🐾'} ${animal.charAt(0).toUpperCase() + animal.slice(1)} pinned successfully!`);
}

function addManualPinToMap(pin) {
  if (!state.map) return;
  
  const info = ANIMAL_DB[pin.animal] || {};
  const color = { critical: '#f85149', high: '#f0883e', medium: '#d29922', low: '#3fb950' }[pin.alert_level] || '#8b949e';
  
  const marker = L.marker([pin.lat, pin.lng], {
    icon: L.divIcon({
      className: 'manual-pin-icon',
      html: `<div style="font-size:24px;text-shadow:0 0 3px #000">${info.emoji || '📍'}</div>`,
      iconSize: [30, 30],
      iconAnchor: [15, 15],
    })
  }).addTo(state.map);
  
  marker.bindPopup(`
    <div style="font-family:Inter,sans-serif;min-width:180px">
      <div style="font-weight:700;font-size:0.95rem;margin-bottom:6px">
        ${info.emoji || '🐾'} ${pin.animal.charAt(0).toUpperCase() + pin.animal.slice(1)}
        ${info.swahili ? `<br><span style="font-size:0.75rem;color:#888">${info.swahili}</span>` : ''}
      </div>
      ${pin.notes ? `<div style="color:#666;font-size:0.8rem;margin-bottom:4px;font-style:italic">"${pin.notes}"</div>` : ''}
      <div style="color:#666;font-size:0.78rem">📍 ${pin.location}</div>
      <div style="color:#666;font-size:0.78rem">⏱ ${formatTimeAgo(pin.timestamp)}</div>
      <div style="margin-top:6px;font-size:0.75rem;color:${color};font-weight:600;text-transform:uppercase">${pin.alert_level} priority</div>
      <div style="font-size:0.7rem;color:#888;margin-top:4px">📌 Manual pin by you</div>
    </div>
  `);
  
  state.mapMarkers.push(marker);
}

function loadManualPins() {
  try {
    const saved = localStorage.getItem('bigv_manual_pins');
    if (saved) {
      state.manualPins = JSON.parse(saved);
      state.manualPins.forEach(pin => addManualPinToMap(pin));
    }
  } catch (e) {
    console.warn('Failed to load manual pins:', e);
  }
}

function saveManualPins() {
  try {
    localStorage.setItem('bigv_manual_pins', JSON.stringify(state.manualPins));
  } catch (e) {
    console.warn('Failed to save manual pins:', e);
  }
}

function showToast(message) {
  const toast = document.createElement('div');
  toast.className = 'toast-notification';
  toast.textContent = message;
  toast.style.cssText = `
    position: fixed;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    background: var(--bg-secondary);
    color: var(--text-primary);
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    z-index: 10000;
    font-size: 0.9rem;
    font-weight: 500;
    animation: slideUp 0.3s ease;
  `;
  document.body.appendChild(toast);
  setTimeout(() => {
    toast.style.animation = 'slideDown 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}
  const el = $('communityCount'); if (el) el.textContent = count;
  const cd = $('communityCountD'); if (cd) cd.textContent = count;
  const badge = $('mapBadge');
  if (badge && count > 0) { badge.textContent = count; badge.style.display = 'block'; }
}

// ─── Tab Switching ────────────────────────────────────────────────────────────
function switchTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  const panel = $(`tab-${tabName}`);
  if (panel) panel.classList.add('active');
  document.querySelectorAll('.nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.tab === tabName);
  });
  if (tabName === 'alerts') {
    state.alertCount = 0;
    const badge = $('alertBadge');
    if (badge) badge.style.display = 'none';
  }
  if (tabName === 'map') {
    initMap();
    loadCommunitySightings();
  }
}

// ─── WebSocket ────────────────────────────────────────────────────────────────
function connectWS() {
  const url = `${CONFIG.wsUrl}/${state.clientId}`;
  state.ws = new WebSocket(url);

  state.ws.onopen = () => setConnectionStatus('connected', 'Live');
  state.ws.onclose = () => {
    setConnectionStatus('disconnected', 'Offline');
    setTimeout(connectWS, 3000);
  };
  state.ws.onerror = err => console.error('[BigV WS]', err);
  state.ws.onmessage = event => {
    try { handleMessage(JSON.parse(event.data)); }
    catch (e) { console.error('[BigV WS] Parse error:', e); }
  };
}

function handleMessage(msg) {
  switch (msg.type) {
    case 'connected':
      $('totalSightings').textContent = msg.total_sightings || 0;
      if (msg.recent?.length) msg.recent.forEach(addSightingCard);
      fetchModelInfo();
      break;
    case 'analysis':
      renderAnalysis(msg.data);
      break;
    case 'sighting':
      state.sightings.unshift(msg.data);
      if (state.sightings.length > CONFIG.maxSightings) state.sightings.pop();
      addSightingCard(msg.data);
      $('totalSightings').textContent = state.sightings.length;
      state.totalAnimals += msg.data.animals.length;
      updateStats();
      break;
    case 'community_sighting':
      handleCommunitySighting(msg.data);
      break;
    case 'alert':
      handleAlert(msg.data);
      break;
    case 'pong':
      break;
  }
}

function handleCommunitySighting(sighting) {
  const isOwn = sighting.reporter_id === state.clientId;
  addLivePinToMap(sighting);

  const badge = $('mapBadge');
  if (badge) {
    const cur = parseInt(badge.textContent || '0', 10);
    badge.textContent = cur + 1;
    badge.style.display = 'block';
  }

  if (!isOwn) {
    const isNearby = (
      state.userLat !== null && sighting.lat != null &&
      haversineKm(state.userLat, state.userLng, sighting.lat, sighting.lng) <= CONFIG.nearbyRadiusKm
    ) || sighting.lat == null;

    if (isNearby) {
      showNearbyAlert(sighting);
      sendBrowserNotification({
        animal: (sighting.animals || []).map(a => a.label).join(', ') || 'Wildlife',
        location: sighting.location,
        confidence: sighting.confidence,
        priority: sighting.alert_level,
      }, '👥');
    }
  }
}

function sendFrame(imageB64) {
  if (state.ws?.readyState === WebSocket.OPEN) {
    state.ws.send(JSON.stringify({
      type: 'frame',
      image: imageB64,
      location: locationInput.value || 'Safari',
      camera_id: state.clientId,
      lat: state.userLat,
      lng: state.userLng,
    }));
  }
}

// ─── Camera ───────────────────────────────────────────────────────────────────
async function populateCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput');
    cameraSelect.innerHTML = '<option value="">Default Camera</option>';
    cameras.forEach((cam, i) => {
      const opt = document.createElement('option');
      opt.value = cam.deviceId;
      opt.textContent = cam.label || `Camera ${i + 1}`;
      cameraSelect.appendChild(opt);
    });
  } catch (e) { console.warn('Camera enumeration failed:', e); }
}

async function startCamera() {
  try {
    const isMobile = window.innerWidth <= 768;
    const constraints = {
      video: {
        deviceId: cameraSelect.value ? { exact: cameraSelect.value } : undefined,
        width:  { ideal: isMobile ? 1280 : 1920 },
        height: { ideal: isMobile ? 720  : 1080 },
        facingMode: isMobile ? 'environment' : 'user',
      }
    };
    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = state.stream;
    await video.play();
    overlay.classList.add('hidden');
    scanLine.classList.add('active');
    startBtn.disabled = true;
    stopBtn.disabled = false;
    state.isRunning = true;
    state.startTime = Date.now();
    $('cameraId').textContent = `Cam: ${state.clientId.slice(-6)}`;
    startAnalyzeLoop();
    startUptimeTimer();
    if (isMobile) switchTab('camera');
  } catch (e) {
    alert(`Camera error: ${e.message}\n\nPlease allow camera access.`);
  }
}

function stopCamera() {
  if (state.stream) { state.stream.getTracks().forEach(t => t.stop()); state.stream = null; }
  video.srcObject = null;
  overlay.classList.remove('hidden');
  scanLine.classList.remove('active');
  startBtn.disabled = false;
  stopBtn.disabled = true;
  state.isRunning = false;
  clearInterval(state.analyzeTimer);
  clearInterval(state.uptimeTimer);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  cameraDetBadge.style.display = 'none';
}

function startAnalyzeLoop() {
  const offscreen = document.createElement('canvas');
  const offCtx = offscreen.getContext('2d');
  state.analyzeTimer = setInterval(() => {
    if (!state.isRunning || video.readyState < 2) return;
    offscreen.width  = Math.floor(video.videoWidth  * CONFIG.frameScale);
    offscreen.height = Math.floor(video.videoHeight * CONFIG.frameScale);
    offCtx.drawImage(video, 0, 0, offscreen.width, offscreen.height);
    const b64 = offscreen.toDataURL('image/jpeg', CONFIG.frameQuality);
    sendFrame(b64);
    state.fpsFrames++;
    state.frameCount++;
    $('frameCount').textContent = `Frames: ${state.frameCount}`;
    const now = Date.now();
    if (now - state.lastFpsTime >= 1000) {
      state.currentFps = state.fpsFrames;
      state.fpsFrames = 0;
      state.lastFpsTime = now;
      $('fpsLabel').textContent = `FPS: ${state.currentFps}`;
    }
  }, CONFIG.analyzeInterval);
}

// ─── Render analysis ──────────────────────────────────────────────────────────
function renderAnalysis(data) {
  const animals = data.animals || [];
  const dangerPct = Math.round((data.danger_score || 0) * 100);

  dangerFill.style.width = `${dangerPct}%`;
  dangerFill.style.background = dangerColor(data.danger_score || 0);
  dangerScore.textContent = `${dangerPct}%`;
  dangerScore.style.color = dangerColor(data.danger_score || 0);

  agentText.textContent = data.summary || 'No wildlife detected.';

  const recs = data.recommendations || [];
  recsBox.style.display = recs.length > 0 ? 'block' : 'none';
  if (recs.length > 0) recsList.innerHTML = recs.map(r => `<li>${r}</li>`).join('');

  // Animal fact card
  showAnimalFacts(data.animal_facts || []);

  // Big Five update
  if (data.big_five_seen) updateBigFive(data.big_five_seen);

  // Ranger report button
  const rangerBox = $('rangerReportBox');
  if (rangerBox) {
    const showRanger = animals.some(a => a.priority === 'critical' || a.priority === 'high');
    rangerBox.style.display = showRanger ? 'block' : 'none';
  }

  if (animals.length > 0) {
    const labels = animals.map(a => {
      const info = ANIMAL_DB[a.label?.toLowerCase()] || {};
      return `${info.emoji || '🐾'} ${a.label}`;
    }).join(', ');
    cameraDetText.textContent = labels;
    cameraDetBadge.style.display = 'block';
    clearTimeout(cameraDetBadge._hideTimer);
    cameraDetBadge._hideTimer = setTimeout(() => { cameraDetBadge.style.display = 'none'; }, 3000);
  }

  if (animals.length === 0) {
    currentDet.className = 'detection-card';
    currentDet.innerHTML = `
      <div class="no-detection">
        <span>🌿</span>
        <p>No wildlife detected</p>
        <p style="font-size:0.72rem;color:var(--text-muted);margin-top:4px">Keep scanning — try water sources & shade</p>
      </div>`;
  } else {
    const topPriority = animals.reduce((best, a) =>
      priorityRank(a.priority) > priorityRank(best.priority) ? a : best, animals[0]);

    currentDet.className = `detection-card has-animal ${topPriority.priority}`;
    currentDet.innerHTML = `
      <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px">
        ${animals.map(a => {
          const info = ANIMAL_DB[a.label?.toLowerCase()] || {};
          const bf = BIG_FIVE_KEYS.includes(a.label?.toLowerCase()) ? ' 🏆' : '';
          return `<span class="animal-chip ${a.priority}">${info.emoji || '🐾'} ${a.label}${bf}</span>`;
        }).join('')}
      </div>
      ${animals.map(a => `
        <div class="conf-bar-wrap">
          <span style="font-size:0.73rem;min-width:76px;text-transform:capitalize">${a.label}</span>
          <div class="conf-bar"><div class="conf-fill" style="width:${Math.round(a.confidence*100)}%"></div></div>
          <span style="font-size:0.7rem;font-family:monospace;color:var(--text-muted)">${Math.round(a.confidence*100)}%</span>
        </div>`).join('')}
      <div style="font-size:0.7rem;color:var(--text-muted);margin-top:6px">
        Behavior: <strong>${data.behavior || 'unknown'}</strong>
      </div>`;

    drawBBoxes(animals);
  }
}

// ─── Canvas bounding boxes ────────────────────────────────────────────────────
function drawBBoxes(animals) {
  canvas.width  = video.videoWidth  || video.clientWidth;
  canvas.height = video.videoHeight || video.clientHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const scaleX = canvas.width  / (video.videoWidth  * CONFIG.frameScale || canvas.width);
  const scaleY = canvas.height / (video.videoHeight * CONFIG.frameScale || canvas.height);

  animals.forEach(a => {
    if (!a.bbox) return;
    const { x1, y1, x2, y2 } = a.bbox;
    const color = priorityColorHex(a.priority);
    const info = ANIMAL_DB[a.label?.toLowerCase()] || {};
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);
    const label = `${info.emoji || ''} ${a.label} ${Math.round(a.confidence * 100)}%`;
    ctx.font = '13px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = color;
    ctx.fillRect(x1 * scaleX - 1, y1 * scaleY - 22, tw + 10, 20);
    ctx.fillStyle = '#000';
    ctx.fillText(label, x1 * scaleX + 4, y1 * scaleY - 6);
  });
}

// ─── Sighting card ────────────────────────────────────────────────────────────
function addSightingCard(sighting) {
  const animals = sighting.animals || [];
  const time = new Date(sighting.timestamp).toLocaleTimeString();
  const level = sighting.alert_level || 'low';
  const isOwn = sighting.reporter_id === state.clientId || !sighting.reporter_id;
  const sourceLabel = isOwn
    ? '<span class="source-badge own">🚗 Your vehicle</span>'
    : '<span class="source-badge community">👥 Another vehicle</span>';

  const card = document.createElement('div');
  card.className = `sighting-card ${level}`;
  card.innerHTML = `
    <div class="sighting-top">
      <div class="sighting-animals">
        ${animals.map(a => {
          const info = ANIMAL_DB[a.label?.toLowerCase()] || {};
          const bf = BIG_FIVE_KEYS.includes(a.label?.toLowerCase()) ? ' 🏆' : '';
          return `<span class="animal-chip ${a.priority}">${info.emoji || '🐾'} ${a.label}${bf}</span>`;
        }).join('')}
      </div>
      <span class="sighting-time">${time}</span>
    </div>
    <div class="sighting-meta">
      <span>📍 ${sighting.location || 'Unknown'}</span>
      <span>🎯 ${Math.round((sighting.confidence || 0) * 100)}%</span>
      ${sighting.lat != null ? `<span>🗺 ${sighting.lat.toFixed(3)}, ${sighting.lng.toFixed(3)}</span>` : ''}
      ${sourceLabel}
    </div>
    ${sighting.image_b64 ? `<img class="sighting-thumb" src="${sighting.image_b64}" alt="sighting" loading="lazy" />` : ''}
  `;
  card.addEventListener('click', () => showSightingDetail(sighting));
  if (sightingsLog.firstChild) {
    sightingsLog.insertBefore(card, sightingsLog.firstChild);
  } else {
    sightingsLog.appendChild(card);
  }
  while (sightingsLog.children.length > CONFIG.maxSightings) {
    sightingsLog.removeChild(sightingsLog.lastChild);
  }
}

// ─── Alert handling ───────────────────────────────────────────────────────────
function handleAlert(alert) {
  state.lastAlert = alert;
  const priority = alert.priority || 'low';
  const emoji = { critical: '🚨', high: '🔴', medium: '🟡', low: '🟢' }[priority] || '📋';
  const info = ANIMAL_DB[alert.animal?.toLowerCase()] || {};
  const animalEmoji = info.emoji || '🐾';

  const tickerEmpty = alertTicker.querySelector('.ticker-empty');
  if (tickerEmpty) tickerEmpty.remove();

  const item = document.createElement('div');
  item.className = 'ticker-item';
  item.innerHTML = `
    <span>${emoji}</span>
    <span>${animalEmoji}</span>
    <span style="font-weight:600;text-transform:capitalize">${alert.animal}</span>
    ${info.swahili ? `<span style="color:var(--text-muted);font-size:0.72rem">(${info.swahili})</span>` : ''}
    <span style="color:var(--text-muted)">@ ${alert.location}</span>
    <span style="color:var(--text-muted);margin-left:auto;font-family:monospace;font-size:0.68rem">
      ${new Date(alert.timestamp).toLocaleTimeString()}
    </span>
  `;
  alertTicker.insertBefore(item, alertTicker.firstChild);
  while (alertTicker.children.length > CONFIG.maxTickerItems) {
    alertTicker.removeChild(alertTicker.lastChild);
  }

  if (state.activeTab !== 'alerts') {
    state.alertCount++;
    const badge = $('alertBadge');
    if (badge) { badge.textContent = state.alertCount; badge.style.display = 'block'; }
  }

  if (priority === 'critical') state.criticalAlerts++;
  updateStats();

  if (priority === 'critical' || priority === 'high') {
    showAlertModal(alert, emoji, animalEmoji);
    flashPanel();
    if ('vibrate' in navigator) {
      navigator.vibrate(priority === 'critical' ? [300, 100, 300, 100, 300] : [200, 100, 200]);
    }
  }

  sendBrowserNotification(alert, `${emoji}${animalEmoji}`);
}

function showAlertModal(alert, emoji, animalEmoji) {
  const info = ANIMAL_DB[alert.animal?.toLowerCase()] || {};
  const isBigFive = BIG_FIVE_KEYS.includes(alert.animal?.toLowerCase());
  $('alertModalIcon').textContent = `${animalEmoji || emoji}`;
  $('alertModalTitle').innerHTML = `${alert.animal?.toUpperCase()} DETECTED${isBigFive ? ' 🏆' : ''}${info.swahili ? `<br><small style="font-size:0.7rem;color:var(--text-muted)">${info.swahili}</small>` : ''}`;
  $('alertModalBody').textContent = alert.summary || '';
  $('alertModalRecs').innerHTML = (alert.recommendations || []).map(r => `<div>${r}</div>`).join('');

  const rangerBtn = $('modalRangerBtn');
  if (rangerBtn) {
    rangerBtn.style.display = (alert.priority === 'critical' || alert.priority === 'high') ? 'block' : 'none';
  }

  alertModal.style.display = 'flex';
  if (alert.priority !== 'critical') setTimeout(closeAlertModal, 12000);
}

function closeAlertModal() {
  alertModal.style.display = 'none';
}

function flashPanel() {
  document.body.classList.add('flash-alert');
  setTimeout(() => document.body.classList.remove('flash-alert'), 2000);
}

function showSightingDetail(sighting) {
  const animals = (sighting.animals || []).map(a => {
    const info = ANIMAL_DB[a.label?.toLowerCase()] || {};
    return `${info.emoji || '🐾'} ${a.label} (${Math.round(a.confidence*100)}%)`;
  }).join(', ');
  const recs = (sighting.recommendations || []).join('\n');
  const gps = sighting.lat != null ? `\nGPS: ${sighting.lat.toFixed(5)}, ${sighting.lng.toFixed(5)}` : '';
  const bf = (sighting.animals || []).some(a => BIG_FIVE_KEYS.includes(a.label?.toLowerCase())) ? '\n🏆 BIG FIVE SIGHTING!' : '';
  alert(`🦁 BigV Safari Sighting\n\nAnimals: ${animals}${bf}\nLocation: ${sighting.location}${gps}\nTime: ${new Date(sighting.timestamp).toLocaleString()}\nAlert: ${sighting.alert_level?.toUpperCase()}\n\n${sighting.agent_summary || ''}\n\nSafari Advice:\n${recs}`);
}

// ─── Browser notifications ────────────────────────────────────────────────────
async function requestNotificationPermission() {
  if ('Notification' in window && Notification.permission === 'default') {
    await Notification.requestPermission();
  }
}

function sendBrowserNotification(alert, emoji) {
  if ('Notification' in window && Notification.permission === 'granted') {
    const info = ANIMAL_DB[alert.animal?.toLowerCase()] || {};
    new Notification(`${emoji} BigV Safari — ${alert.animal?.toUpperCase()} SPOTTED`, {
      body: `${alert.location} · ${Math.round((alert.confidence || 0) * 100)}% confidence${info.swahili ? ` · ${info.swahili}` : ''}`,
      icon: '/static/img/icon-192.png',
      badge: '/static/img/icon-192.png',
      tag: `bigv-${alert.animal}`,
      vibrate: [200, 100, 200],
    });
  }
}

// ─── Image upload ─────────────────────────────────────────────────────────────
uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async ev => {
    const b64 = ev.target.result;
    overlay.classList.remove('hidden');
    overlay.innerHTML = `<img src="${b64}" style="max-width:100%;max-height:100%;object-fit:contain;border-radius:8px" />`;
    try {
      const resp = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_b64: b64,
          location: locationInput.value || 'Safari',
          camera_id: state.clientId,
          lat: state.userLat,
          lng: state.userLng,
        }),
      });
      const result = await resp.json();
      renderAnalysis(result);
      if (window.innerWidth <= 768) switchTab('detection');
    } catch (err) {
      console.error('Upload error:', err);
      agentText.textContent = `Error: ${err.message}`;
    }
  };
  reader.readAsDataURL(file);
  fileInput.value = '';
});

// ─── Stats ────────────────────────────────────────────────────────────────────
function updateStats() {
  $('statAnimals').textContent  = state.totalAnimals;
  $('statFrames').textContent   = state.frameCount;
  const ad = $('statAnimalsD'); if (ad) ad.textContent = state.totalAnimals;
  const fd = $('statFramesD');  if (fd) fd.textContent = state.frameCount;
}

function startUptimeTimer() {
  state.uptimeTimer = setInterval(() => {
    const elapsed = Math.floor((Date.now() - state.startTime) / 1000);
    const h = String(Math.floor(elapsed / 3600)).padStart(2, '0');
    const m = String(Math.floor((elapsed % 3600) / 60)).padStart(2, '0');
    const s = String(elapsed % 60).padStart(2, '0');
    $('statUptime').textContent = `${m}:${s}`;
    const ud = $('statUptimeD'); if (ud) ud.textContent = `${h}:${m}:${s}`;
  }, 1000);
}

async function fetchModelInfo() {
  try {
    const resp = await fetch('/health');
    const data = await resp.json();
    const model = $('statModel'); if (model) model.textContent = data.detector || 'Unknown';
    $('totalSightings').textContent = data.total_sightings || 0;
  } catch (e) {
    const model = $('statModel'); if (model) model.textContent = 'Offline';
  }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function setConnectionStatus(st, text) {
  connStatus.className = `status-badge ${st}`;
  connStatus.innerHTML = `<span class="dot"></span> ${text}`;
}

function dangerColor(score) {
  if (score >= 0.8) return '#f85149';
  if (score >= 0.6) return '#f0883e';
  if (score >= 0.4) return '#d29922';
  return '#3fb950';
}

function priorityColorHex(priority) {
  return { critical: '#f85149', high: '#f0883e', medium: '#d29922', low: '#3fb950' }[priority] || '#8b949e';
}

function priorityRank(p) {
  return { low: 0, medium: 1, high: 2, critical: 3 }[p] || 0;
}

function formatTimeAgo(timestamp) {
  const diff = Math.floor((Date.now() - new Date(timestamp).getTime()) / 1000);
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff/60)}m ago`;
  return `${Math.floor(diff/3600)}h ago`;
}

// ─── Event listeners ──────────────────────────────────────────────────────────
startBtn.addEventListener('click', startCamera);
stopBtn.addEventListener('click', stopCamera);
installBtn.addEventListener('click', triggerInstall);
$('installBannerBtn')?.addEventListener('click', triggerInstall);

$('clearLogBtn').addEventListener('click', () => {
  sightingsLog.innerHTML = '';
  alertTicker.innerHTML = '<div class="ticker-empty">No alerts yet — monitoring active</div>';
  state.sightings = [];
  state.alertCount = 0;
  $('totalSightings').textContent = 0;
  const badge = $('alertBadge');
  if (badge) badge.style.display = 'none';
});

alertModal.addEventListener('click', e => {
  if (e.target === alertModal) closeAlertModal();
});

document.addEventListener('keydown', e => {
  if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
    e.preventDefault();
    state.isRunning ? stopCamera() : startCamera();
  }
  if (e.code === 'Escape') closeAlertModal();
});

// ─── Init ─────────────────────────────────────────────────────────────────────
(async function init() {
  connectWS();
  await populateCameras();
  await requestNotificationPermission();
  startGPS();
  buildFieldGuide();
  setMode('tourist');
  fetchModelInfo();
  setInterval(fetchModelInfo, 15000);
  state.communityTimer = setInterval(() => {
    if (state.activeTab === 'map') loadCommunitySightings();
  }, CONFIG.communityRefreshMs);
  
  // Load manual pins from localStorage after map initializes
  setTimeout(() => {
    if (state.map) loadManualPins();
  }, 1000);
  
  console.log(`[${CONFIG.appName}] Ready. Client: ${state.clientId}`);
})();

// Made with Bob
