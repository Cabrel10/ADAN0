
import websocket
import threading
import json
import time
from queue import Queue
from typing import List, Dict, Callable, Any

from adan_trading_bot.common.utils import get_logger

logger = get_logger(__name__)

class WebSocketManager:
    """
    Manages WebSocket connections for real-time data streams from exchanges.
    """

    def __init__(self, ws_url: str, subscriptions: List[str]):
        """
        Initializes the WebSocketManager.

        Args:
            ws_url: The WebSocket URL to connect to.
            subscriptions: A list of streams to subscribe to.
        """
        self.ws_url = ws_url
        self.subscriptions = subscriptions
        self.ws = None
        self.thread = None
        self.data_queue = Queue()
        self.stop_event = threading.Event()

    def _on_message(self, ws, message):
        """Callback for when a message is received."""
        try:
            data = json.loads(message)
            self.data_queue.put(data)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode WebSocket message: {message}")

    def _on_error(self, ws, error):
        """Callback for when an error occurs."""
        logger.error(f"WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        """Callback for when the connection is closed."""
        logger.info(f"WebSocket connection closed: {close_status_code} {close_msg}")
        if not self.stop_event.is_set():
            logger.info("Reconnecting WebSocket...")
            self.start()

    def _on_open(self, ws):
        """Callback for when the connection is opened."""
        logger.info(f"WebSocket connection opened to {self.ws_url}")
        self.subscribe()

    def subscribe(self):
        """Subscribes to the specified streams."""
        if self.ws:
            sub_payload = {
                "method": "SUBSCRIBE",
                "params": self.subscriptions,
                "id": 1
            }
            self.ws.send(json.dumps(sub_payload))
            logger.info(f"Subscribed to streams: {self.subscriptions}")

    def start(self):
        """Starts the WebSocket connection in a new thread."""
        if self.thread and self.thread.is_alive():
            logger.warning("WebSocket manager is already running.")
            return

        self.stop_event.clear()
        self.ws = websocket.WebSocketApp(
            self.ws_url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.thread.start()
        logger.info("WebSocket manager started.")

    def stop(self):
        """Stops the WebSocket connection."""
        if self.thread and self.thread.is_alive():
            self.stop_event.set()
            if self.ws:
                self.ws.close()
            self.thread.join(timeout=5)
            logger.info("WebSocket manager stopped.")

    def get_data(self, block: bool = True, timeout: float = None) -> Any:
        """
        Retrieves data from the queue.

        Args:
            block: Whether to block until an item is available.
            timeout: Maximum time to block.

        Returns:
            The next item from the queue, or None if the queue is empty and not blocking.
        """
        if not self.is_alive():
            logger.warning("Attempted to get data from a stopped WebSocket manager.")
            return None
        try:
            return self.data_queue.get(block=block, timeout=timeout)
        except self.data_queue.empty:
            return None

    def is_alive(self) -> bool:
        """Checks if the WebSocket connection thread is alive."""
        return self.thread and self.thread.is_alive()

