/**
 * BigV — Service Worker
 * Enables PWA offline support and caching
 */

const CACHE_NAME = 'bigv-v1.0';
const STATIC_ASSETS = [
  '/',
  '/static/css/app.css',
  '/static/js/app.js',
  '/static/manifest.json',
  '/static/img/icon-192.png',
  '/static/img/icon-512.png',
  '/static/img/icon-180.png',
];

// ─── Install: cache static assets ────────────────────────────────────────────
self.addEventListener('install', event => {
  console.log('[SW] Installing BigV service worker...');
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS);
    }).then(() => self.skipWaiting())
  );
});

// ─── Activate: clean old caches ──────────────────────────────────────────────
self.addEventListener('activate', event => {
  console.log('[SW] Activating BigV service worker...');
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys.filter(k => k !== CACHE_NAME).map(k => {
          console.log('[SW] Deleting old cache:', k);
          return caches.delete(k);
        })
      )
    ).then(() => self.clients.claim())
  );
});

// ─── Fetch: serve from cache, fall back to network ───────────────────────────
self.addEventListener('fetch', event => {
  const url = new URL(event.request.url);

  // Skip WebSocket requests and API calls — always go to network
  if (
    event.request.url.includes('/ws/') ||
    event.request.url.includes('/analyze') ||
    event.request.url.includes('/upload') ||
    event.request.url.includes('/sightings') ||
    event.request.url.includes('/health')
  ) {
    return; // Let browser handle normally
  }

  // For static assets: cache-first strategy
  if (event.request.method === 'GET') {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;

        return fetch(event.request).then(response => {
          // Cache successful responses for static assets
          if (response.ok && url.pathname.startsWith('/static/')) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          }
          return response;
        }).catch(() => {
          // Offline fallback for navigation requests
          if (event.request.mode === 'navigate') {
            return caches.match('/');
          }
        });
      })
    );
  }
});

// ─── Push notifications ───────────────────────────────────────────────────────
self.addEventListener('push', event => {
  if (!event.data) return;
  const data = event.data.json();
  const options = {
    body: data.body || 'Wildlife detected!',
    icon: '/static/img/icon-192.png',
    badge: '/static/img/icon-192.png',
    vibrate: [200, 100, 200],
    tag: data.tag || 'bigv-alert',
    data: { url: data.url || '/' },
    actions: [
      { action: 'view', title: '👁 View', icon: '/static/img/icon-192.png' },
      { action: 'dismiss', title: '✕ Dismiss' },
    ],
  };
  event.waitUntil(
    self.registration.showNotification(data.title || '🦁 BigV Alert', options)
  );
});

self.addEventListener('notificationclick', event => {
  event.notification.close();
  if (event.action === 'view') {
    event.waitUntil(clients.openWindow(event.notification.data.url));
  }
});

// Made with Bob
