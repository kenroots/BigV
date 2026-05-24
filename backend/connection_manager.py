"""
WebSocket Connection Manager — tracks all connected clients
and provides broadcast / targeted messaging.
"""

import logging
from fastapi import WebSocket

logger = logging.getLogger("wildlife_spotter.connections")


class ConnectionManager:
    def __init__(self):
        self._connections: dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self._connections[client_id] = websocket
        logger.info(f"[+] Client {client_id} connected. Total: {len(self._connections)}")

    def disconnect(self, client_id: str):
        self._connections.pop(client_id, None)
        logger.info(f"[-] Client {client_id} disconnected. Total: {len(self._connections)}")

    async def send(self, client_id: str, message: str):
        """Send message to a specific client."""
        ws = self._connections.get(client_id)
        if ws:
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Send to {client_id} failed: {e}")
                self.disconnect(client_id)

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        dead = []
        for client_id, ws in self._connections.items():
            try:
                await ws.send_text(message)
            except Exception as e:
                logger.warning(f"Broadcast to {client_id} failed: {e}")
                dead.append(client_id)
        for client_id in dead:
            self.disconnect(client_id)

    def count(self) -> int:
        return len(self._connections)

    def client_ids(self) -> list[str]:
        return list(self._connections.keys())

# Made with Bob
