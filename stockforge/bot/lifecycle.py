"""Process lifecycle, single-instance advisory locks, polling heartbeats, and safe callbacks (SF-003)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("stockforge.bot.lifecycle")

def get_bot_heartbeat_file() -> str:
    return os.environ.get("BOT_HEARTBEAT_FILE", "/tmp/rupika_bot_heartbeat")


def get_bot_lock_file() -> str:
    return os.environ.get("BOT_LOCK_FILE", "/tmp/rupika_bot.lock")


def get_worker_heartbeat_file() -> str:
    return os.environ.get("WORKER_HEARTBEAT_FILE", "/tmp/rupika_worker_heartbeat")


# Backward compatible aliases
BOT_HEARTBEAT_FILE = "/tmp/rupika_bot_heartbeat"
BOT_LOCK_FILE = "/tmp/rupika_bot.lock"
WORKER_HEARTBEAT_FILE = "/tmp/rupika_worker_heartbeat"


class BotProcessLock:
    """Advisory file lock preventing multiple polling bot instances on the same host."""

    def __init__(self, lock_path: Optional[str] = None) -> None:
        self.lock_path = lock_path or get_bot_lock_file()
        self._fd: Optional[int] = None

    def acquire(self) -> bool:
        """Attempt non-blocking acquisition of process lock."""
        import fcntl
        try:
            Path(self.lock_path).parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(self._fd, 0)
            os.write(self._fd, f"{os.getpid()}\n".encode("utf-8"))
            os.fsync(self._fd)
            logger.info("Bot process lock berhasil diperoleh (PID %d, path %s)", os.getpid(), self.lock_path)
            return True
        except (BlockingIOError, OSError) as exc:
            logger.warning(
                "Tidak dapat memperoleh lock bot di %s (mungkin instance lain sedang berjalan): %s",
                self.lock_path,
                exc,
            )
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except Exception:
                    pass
                self._fd = None
            return False

    def release(self) -> None:
        """Release advisory process lock and unlink lock file."""
        import fcntl
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
                os.close(self._fd)
            except Exception:
                pass
            self._fd = None
            try:
                if os.path.exists(self.lock_path):
                    os.unlink(self.lock_path)
            except Exception:
                pass
            logger.info("Bot process lock dilepaskan (%s)", self.lock_path)


def touch_bot_heartbeat(status: str = "healthy", details: Optional[dict[str, Any]] = None) -> None:
    """Record liveness timestamp for Docker and orchestration healthchecks."""
    target_file = get_bot_heartbeat_file()
    try:
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)
        tmp_file = f"{target_file}.tmp.{os.getpid()}"
        payload = {
            "pid": os.getpid(),
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {},
        }
        with open(tmp_file, "w") as f:
            f.write(json.dumps(payload))
        os.replace(tmp_file, target_file)
    except Exception as exc:
        logger.debug("Gagal memperbarui bot heartbeat: %s", exc)


def remove_bot_heartbeat() -> None:
    """Clean up bot heartbeat file on shutdown."""
    target_file = get_bot_heartbeat_file()
    try:
        if os.path.exists(target_file):
            os.unlink(target_file)
    except Exception:
        pass


def touch_worker_heartbeat(status: str = "running", job_id: Optional[int] = None) -> None:
    """Record worker heartbeat timestamp for Docker healthchecks."""
    target_file = get_worker_heartbeat_file()
    try:
        Path(target_file).parent.mkdir(parents=True, exist_ok=True)
        tmp_file = f"{target_file}.tmp.{os.getpid()}"
        payload = {
            "pid": os.getpid(),
            "status": status,
            "job_id": job_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(tmp_file, "w") as f:
            f.write(json.dumps(payload))
        os.replace(tmp_file, target_file)
    except Exception as exc:
        logger.debug("Gagal memperbarui worker heartbeat: %s", exc)


def remove_worker_heartbeat() -> None:
    """Clean up worker heartbeat file on shutdown."""
    target_file = get_worker_heartbeat_file()
    try:
        if os.path.exists(target_file):
            os.unlink(target_file)
    except Exception:
        pass


async def safe_answer_callback(
    query: Any,
    text: Optional[str] = None,
    show_alert: bool = False,
    cache_time: Optional[int] = None,
) -> bool:
    """Safely answer Telegram callback queries without crashing on expired or invalid queries."""
    if not query:
        return False
    try:
        kwargs: dict[str, Any] = {}
        if text is not None:
            kwargs["text"] = text
        if show_alert:
            kwargs["show_alert"] = show_alert
        if cache_time is not None:
            kwargs["cache_time"] = cache_time

        await query.answer(**kwargs)
        return True
    except Exception as exc:
        err_str = str(exc).lower()
        if "query is too old" in err_str or "query id is invalid" in err_str:
            logger.debug("Callback query kadaluarsa atau ID tidak valid (diabaikan dengan aman): %s", exc)
        else:
            logger.warning("Gagal menjawab callback query: %s", exc)
        return False
