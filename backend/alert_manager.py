"""
Alert Manager — processes sightings and triggers alerts:
- WebSocket push to all clients
- Console/log alerts
- Optional: Slack webhook, email (configurable via env vars)
"""

import json
import logging
import os
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("wildlife_spotter.alerts")

# Alert cooldown per animal per camera (seconds) — avoid alert spam
ALERT_COOLDOWN = {
    "critical": 30,
    "high":     60,
    "medium":   120,
    "low":      300,
}


class AlertManager:
    def __init__(self, connection_manager):
        self.manager = connection_manager
        self._last_alert: dict[str, float] = {}  # key: "camera_id:label"
        self.slack_webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        self.alert_log: list[dict] = []

    async def process(self, sighting: dict):
        """Process a sighting and fire appropriate alerts."""
        alert_level = sighting.get("alert_level", "low")
        animals = sighting.get("animals", [])
        camera_id = sighting.get("camera_id", "unknown")

        for animal in animals:
            label = animal.get("label", "unknown")
            priority = animal.get("priority", "low")
            cooldown_key = f"{camera_id}:{label}"

            # Check cooldown
            last = self._last_alert.get(cooldown_key, 0)
            cooldown = ALERT_COOLDOWN.get(priority, 300)
            if time.time() - last < cooldown:
                continue

            self._last_alert[cooldown_key] = time.time()

            alert = self._build_alert(sighting, animal)
            self.alert_log.append(alert)

            # Log to console
            self._log_alert(alert)

            # Push alert event to all WebSocket clients
            await self.manager.broadcast(json.dumps({
                "type": "alert",
                "data": alert,
            }))

            # Optional Slack notification
            if self.slack_webhook and priority in ("critical", "high"):
                await self._send_slack(alert)

    def _build_alert(self, sighting: dict, animal: dict) -> dict:
        return {
            "sighting_id": sighting.get("id"),
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "animal": animal.get("label"),
            "confidence": animal.get("confidence"),
            "priority": animal.get("priority"),
            "alert_level": sighting.get("alert_level"),
            "location": sighting.get("location"),
            "camera_id": sighting.get("camera_id"),
            "summary": sighting.get("agent_summary", ""),
            "recommendations": sighting.get("recommendations", []),
        }

    def _log_alert(self, alert: dict):
        priority = alert.get("priority", "low")
        emoji = {"critical": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "📋")
        logger.warning(
            f"{emoji} WILDLIFE ALERT [{priority.upper()}] "
            f"{alert['animal']} @ {alert['location']} "
            f"(cam: {alert['camera_id']}, conf: {alert['confidence']:.0%})"
        )

    async def _send_slack(self, alert: dict):
        """Send Slack notification (requires SLACK_WEBHOOK_URL env var)."""
        try:
            import aiohttp
            priority = alert.get("priority", "low")
            color = {"critical": "#FF0000", "high": "#FF6600", "medium": "#FFCC00", "low": "#00CC00"}.get(priority, "#888888")
            payload = {
                "attachments": [{
                    "color": color,
                    "title": f"🦁 Wildlife Alert — {alert['animal'].title()} Detected",
                    "fields": [
                        {"title": "Location", "value": alert["location"], "short": True},
                        {"title": "Camera", "value": alert["camera_id"], "short": True},
                        {"title": "Confidence", "value": f"{alert['confidence']:.0%}", "short": True},
                        {"title": "Priority", "value": priority.upper(), "short": True},
                        {"title": "Summary", "value": alert["summary"], "short": False},
                    ],
                    "footer": "Wildlife Spotter",
                    "ts": int(time.time()),
                }]
            }
            async with aiohttp.ClientSession() as session:
                await session.post(self.slack_webhook, json=payload)
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")

    def get_recent_alerts(self, limit: int = 20) -> list[dict]:
        return self.alert_log[-limit:]

# Made with Bob
