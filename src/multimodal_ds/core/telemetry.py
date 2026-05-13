"""
Session telemetry — lightweight structured logging for production observability.
Writes JSONL to agentic_output/{session_id}/telemetry.jsonl
"""
from __future__ import annotations
import json
import logging
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SessionTelemetry:
    def __init__(self, session_id: str, output_dir: Path):
        self.session_id = session_id
        self.log_path = output_dir / session_id / "telemetry.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._metrics: Dict[str, Any] = {
            "session_id": session_id,
            "node_durations_s": {},
            "llm_calls": 0,
            "llm_failures": 0,
            "files_created": [],
            "pii_blocks": 0,
            "retry_count": 0,
            "tasks_attempted": 0,
            "tasks_succeeded": 0,
        }

    def record_event(self, event_type: str, data: Dict[str, Any]) -> None:
        entry = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "session_id": self.session_id,
            "event": event_type,
            **data,
        }
        with self._lock:
            try:
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception as e:
                logger.debug(f"[Telemetry] Write failed: {e}")

    def increment(self, metric: str, by: int = 1) -> None:
        with self._lock:
            if metric in self._metrics:
                self._metrics[metric] += by

    def record_node_duration(self, node: str, duration_s: float) -> None:
        with self._lock:
            self._metrics["node_durations_s"][node] = round(duration_s, 3)
        self.record_event("node_complete", {"node": node, "duration_s": round(duration_s, 3)})

    @contextmanager
    def timed_node(self, node_name: str):
        t0 = time.time()
        try:
            yield self
        finally:
            self.record_node_duration(node_name, time.time() - t0)

    def flush_summary(self) -> Dict[str, Any]:
        with self._lock:
            summary = dict(self._metrics)
        self.record_event("session_summary", summary)
        return summary


_telemetry: Dict[str, SessionTelemetry] = {}
_tel_lock = threading.Lock()


def get_telemetry(session_id: str, output_dir: Optional[Path] = None) -> SessionTelemetry:
    with _tel_lock:
        if session_id not in _telemetry:
            from multimodal_ds.config import OUTPUT_DIR
            _telemetry[session_id] = SessionTelemetry(session_id, output_dir or OUTPUT_DIR)
        return _telemetry[session_id]
