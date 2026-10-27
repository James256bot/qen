// Development: service worker disabled.
//
// The previous SW versions cached /static/index.html aggressively and
// caused stale-HTML issues during rapid iteration. This version deletes
// all caches, unregisters itself, and reloads every open client.

self.addEventListener("install", (e) => {
  self.skipWaiting();
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map(k => caches.delete(k)));
      await self.registration.unregister();
      const clients = await self.clients.matchAll({ type: "window" });
      for (const c of clients) c.navigate(c.url);
    })()
  );
});
