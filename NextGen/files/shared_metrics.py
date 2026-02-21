"""
Shared Metrics System for SYNAPSE

Enables communication between training process and API server.
Uses JSON file for cross-process metrics sharing.
"""

import json
import time
import threading
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class SharedMetrics:
    """
    Thread-safe metrics sharing between processes.

    Training process writes metrics, API server reads them.
    Uses atomic file writes to prevent corruption.
    """

    def __init__(self, metrics_file: str = "shared_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self._lock = threading.Lock()
        self._cache: Dict = {}
        self._last_update = 0

        # Initialize empty metrics
        self._write_metrics({
            "state": "idle",
            "step": 0,
            "total_samples": 0,
            "elapsed_seconds": 0,
            "phi": 0.0,
            "agency": 0.0,
            "integration_score": 0.0,
            "meta_confidence": 0.0,
            "neurons": 0,
            "growth_events": 0,
            "samples_per_second": 0.0,
            "timestamp": datetime.now().isoformat(),
            "neurochemistry": {
                "dopamine": 0.5,
                "serotonin": 0.5,
                "oxytocin": 0.5,
                "cortisol": 0.3,
                "norepinephrine": 0.4
            }
        })

    def update(self,
               state: str = "running",
               step: int = 0,
               total_samples: int = 0,
               elapsed_seconds: float = 0.0,
               metrics: Optional[Dict] = None,
               neurochemistry: Optional[Dict] = None,
               neurons: int = 0,
               growth_events: int = 0):
        """
        Update shared metrics from training process.

        Call this after each training step.
        """
        with self._lock:
            data = {
                "state": state,
                "step": step,
                "total_samples": total_samples,
                "elapsed_seconds": elapsed_seconds,
                "phi": metrics.get("phi", 0.0) if metrics else 0.0,
                "agency": metrics.get("mean_agency", 0.0) if metrics else 0.0,
                "integration_score": metrics.get("integration_score", 0.0) if metrics else 0.0,
                "meta_confidence": metrics.get("meta_confidence", 0.0) if metrics else 0.0,
                "meta_uncertainty": metrics.get("meta_uncertainty", 0.0) if metrics else 0.0,
                "self_confidence": metrics.get("self_confidence", 0.0) if metrics else 0.0,
                "neurons": neurons,
                "growth_events": growth_events,
                "samples_per_second": total_samples / elapsed_seconds if elapsed_seconds > 0 else 0.0,
                "timestamp": datetime.now().isoformat(),
                "neurochemistry": neurochemistry or {}
            }
            self._write_metrics(data)

    def _write_metrics(self, data: Dict):
        """Atomic write to file"""
        temp_file = self.metrics_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            # Atomic rename
            temp_file.replace(self.metrics_file)
            self._cache = data
            self._last_update = time.time()
        except Exception as e:
            print(f"[SharedMetrics] Write error: {e}")
            if temp_file.exists():
                temp_file.unlink()

    def read(self) -> Dict:
        """
        Read current metrics.

        Returns cached data if file is locked or unreadable.
        """
        # Use cache if recently updated
        if time.time() - self._last_update < 0.1:
            return self._cache.copy()

        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._cache = data
                    self._last_update = time.time()
                    return data
        except Exception as e:
            pass

        return self._cache.copy()

    def set_state(self, state: str):
        """Update only state"""
        current = self.read()
        current["state"] = state
        current["timestamp"] = datetime.now().isoformat()
        self._write_metrics(current)


# Global instance
_shared_metrics: Optional[SharedMetrics] = None


def get_shared_metrics(metrics_file: str = "shared_metrics.json") -> SharedMetrics:
    """Get or create shared metrics instance"""
    global _shared_metrics
    if _shared_metrics is None:
        _shared_metrics = SharedMetrics(metrics_file)
    return _shared_metrics
