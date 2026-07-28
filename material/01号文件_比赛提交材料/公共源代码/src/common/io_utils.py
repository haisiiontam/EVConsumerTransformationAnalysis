from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (bool, int, float, str)) or obj is None:
        return obj
    try:
        import numpy as np

        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    atomic_write_text(
        path, json.dumps(data, ensure_ascii=False, indent=2, default=_json_default)
    )


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        os.close(fd)
        df.to_csv(tmp, index=False, encoding="utf-8-sig")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def config_hash(obj: Any) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def setup_file_logging(
    log_path: Path,
    run_id: str,
    *,
    logger_name: str = "run",
    handler_attr: str = "_shared_file",
) -> logging.Logger:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in list(root.handlers):
        if getattr(h, handler_attr, False) or getattr(h, "_task1_file", False):
            root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    class FlushFileHandler(logging.FileHandler):
        def emit(self, record: logging.LogRecord) -> None:
            super().emit(record)
            self.flush()

    ffh = FlushFileHandler(log_path, encoding="utf-8")
    ffh.setLevel(logging.INFO)
    ffh.setFormatter(fmt)
    setattr(ffh, handler_attr, True)
    ffh._task1_file = True
    root.addHandler(ffh)

    if not any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root.handlers
    ):
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        root.addHandler(sh)

    log = logging.getLogger(logger_name)
    log.info("file logging -> %s | run_id=%s", log_path, run_id)
    return log


class RunState:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.data: Dict[str, Any] = {
            "run_id": run_id,
            "updated_at": now_iso(),
            "stages": {},
            "exit_status": "running",
        }
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
                if "stages" not in self.data:
                    self.data["stages"] = {}
            except Exception:
                pass

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        atomic_write_json(self.path, self.data)

    def begin(
        self,
        stage: str,
        expected_artifacts: Optional[list] = None,
        configuration_hash: str = "",
    ) -> None:
        self.data["stages"][stage] = {
            "stage": stage,
            "status": "running",
            "started_at": now_iso(),
            "completed_at": None,
            "run_id": self.run_id,
            "expected_artifacts": expected_artifacts or [],
            "completed_folds": [],
            "completed_trials": [],
            "configuration_hash": configuration_hash,
            "error_message": None,
        }
        self.save()

    def complete(self, stage: str, **extra: Any) -> None:
        st = self.data["stages"].setdefault(stage, {"stage": stage})
        st["status"] = "completed"
        st["completed_at"] = now_iso()
        st.update(extra)
        self.save()

    def fail(self, stage: str, error: str) -> None:
        st = self.data["stages"].setdefault(stage, {"stage": stage})
        st["status"] = "failed"
        st["completed_at"] = now_iso()
        st["error_message"] = error
        self.data["exit_status"] = "failed"
        self.save()

    def partial(self, stage: str, error: str = "", **extra: Any) -> None:
        st = self.data["stages"].setdefault(stage, {"stage": stage})
        st["status"] = "partial"
        st["completed_at"] = now_iso()
        if error:
            st["error_message"] = error
        st.update(extra)
        self.save()

    def status_of(self, stage: str) -> str:
        return str(self.data.get("stages", {}).get(stage, {}).get("status", "pending"))
