"""
Lilim Scheduler — Task Scheduling via systemd-run or APScheduler

Allows natural-language-triggered scheduling:
  "Remind me to take my meds in 30 minutes"
  "Remind me every day at 9am to review anatomy notes"

Implementation strategy (in order of preference):
  1. systemd-run --on-active or --on-calendar (zero extra deps, most reliable)
  2. APScheduler in-process (if apscheduler is installed)
  3. Fallback: simple threading.Timer for in-process one-shot reminders

Notifications are sent via:
  - notify-send (desktop pop-up — requires libnotify-bin)
  - Fallback: write to /tmp/lilim_reminders.log

DB: stores scheduled tasks in ~/.local/share/lilim/schedules.db
"""

import re
import sqlite3
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

SCHEDULE_DB = Path.home() / ".local" / "share" / "lilim" / "schedules.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message     TEXT NOT NULL,
    trigger     TEXT NOT NULL,     -- ISO timestamp (one-shot) or cron expr (recurring)
    schedule_type TEXT NOT NULL,   -- 'once' | 'recurring'
    backend     TEXT NOT NULL,     -- 'systemd' | 'apscheduler' | 'threading'
    job_id      TEXT,              -- external job ID (systemd unit name, etc.)
    created_at  TEXT NOT NULL,
    fired       INTEGER DEFAULT 0,
    active      INTEGER DEFAULT 1
);
"""

# Simple natural-language time parsing
DURATION_PATTERNS = [
    (r"(\d+)\s*second", "seconds"),
    (r"(\d+)\s*minute", "minutes"),
    (r"(\d+)\s*hour", "hours"),
    (r"(\d+)\s*day", "days"),
]

RECURRING_PATTERNS = [
    (r"every\s+day\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", "daily"),
    (r"every\s+morning", "daily_9am"),
    (r"every\s+evening", "daily_6pm"),
    (r"every\s+(\d+)\s+minutes?", "interval_minutes"),
    (r"every\s+hour", "hourly"),
    (r"every\s+week(?:ly)?", "weekly"),
]


class Scheduler:
    """Task scheduler for Lilim reminders."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or SCHEDULE_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._timers: dict[int, threading.Timer] = {}

    # ── Public API ────────────────────────────────────────────

    def schedule_once(self, message: str, natural_time: str) -> dict:
        """Schedule a one-time reminder.

        Args:
            message:      The reminder message to show.
            natural_time: e.g. "in 30 minutes", "in 2 hours", "in 1 day"

        Returns:
            dict with id, message, trigger_at, backend, error
        """
        delta = self._parse_duration(natural_time)
        if delta is None:
            return {
                "error": f"Could not parse time: '{natural_time}'. "
                         "Try 'in 30 minutes', 'in 2 hours', etc.",
                "id": None,
            }

        trigger_at = datetime.utcnow() + delta
        trigger_iso = trigger_at.isoformat()

        # Try backends in order
        backend, job_id = self._schedule_once_systemd(message, delta)
        if not backend:
            backend, job_id = self._schedule_once_threading(message, delta)

        row_id = self._save_schedule(
            message=message,
            trigger=trigger_iso,
            schedule_type="once",
            backend=backend,
            job_id=job_id,
        )

        return {
            "id": row_id,
            "message": message,
            "trigger_at": trigger_iso,
            "backend": backend,
            "error": None,
        }

    def schedule_recurring(self, message: str, natural_time: str) -> dict:
        """Schedule a recurring reminder.

        Args:
            message:      The reminder message.
            natural_time: e.g. "every day at 9am", "every 30 minutes", "every morning"

        Returns:
            dict with id, message, trigger, backend, error
        """
        cron_expr, human = self._parse_recurring(natural_time)
        if cron_expr is None:
            return {
                "error": f"Could not parse recurring time: '{natural_time}'. "
                         "Try 'every day at 9am', 'every morning', 'every 30 minutes'.",
                "id": None,
            }

        backend, job_id = self._schedule_recurring_systemd(message, cron_expr)

        row_id = self._save_schedule(
            message=message,
            trigger=cron_expr,
            schedule_type="recurring",
            backend=backend,
            job_id=job_id,
        )

        return {
            "id": row_id,
            "message": message,
            "trigger": cron_expr,
            "human_readable": human,
            "backend": backend,
            "error": None,
        }

    def list_schedules(self, active_only: bool = True) -> list[dict]:
        """List all scheduled tasks."""
        with self._conn() as conn:
            query = "SELECT id, message, trigger, schedule_type, backend, created_at, active FROM schedules"
            if active_only:
                query += " WHERE active = 1"
            query += " ORDER BY created_at DESC LIMIT 50"
            rows = conn.execute(query).fetchall()

        return [
            {
                "id": r[0], "message": r[1], "trigger": r[2],
                "type": r[3], "backend": r[4],
                "created_at": r[5], "active": bool(r[6]),
            }
            for r in rows
        ]

    def cancel(self, schedule_id: int) -> dict:
        """Cancel a scheduled task by ID."""
        # Cancel threading timer if running
        if schedule_id in self._timers:
            self._timers[schedule_id].cancel()
            del self._timers[schedule_id]

        # Fetch job_id for systemd cancel
        with self._conn() as conn:
            row = conn.execute(
                "SELECT job_id, backend FROM schedules WHERE id = ?", (schedule_id,)
            ).fetchone()

        if row:
            job_id, backend = row
            if backend == "systemd" and job_id:
                try:
                    subprocess.run(
                        ["systemctl", "--user", "stop", job_id],
                        timeout=5, capture_output=True
                    )
                    subprocess.run(
                        ["systemctl", "--user", "disable", job_id],
                        timeout=5, capture_output=True
                    )
                except Exception:
                    pass

        with self._conn() as conn:
            conn.execute("UPDATE schedules SET active = 0 WHERE id = ?", (schedule_id,))

        return {"id": schedule_id, "cancelled": True}

    # ── Notification ──────────────────────────────────────────

    @staticmethod
    def send_notification(message: str, title: str = "Lilim Reminder"):
        """Send a desktop notification."""
        try:
            subprocess.run(
                ["notify-send", "-a", "Lilim", title, message],
                timeout=5,
                capture_output=True,
            )
        except Exception:
            # Fallback: write to log
            try:
                log = Path("/tmp/lilim_reminders.log")
                with open(log, "a") as f:
                    f.write(f"{datetime.now().isoformat()} — {title}: {message}\n")
            except Exception:
                pass

    # ── Backend: systemd-run ──────────────────────────────────

    def _schedule_once_systemd(self, message: str, delta: timedelta) -> tuple[str, Optional[str]]:
        """Try to schedule via systemd-run --on-active."""
        if not self._systemd_available():
            return "", None

        total_seconds = int(delta.total_seconds())
        unit_name = f"lilim-reminder-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        notify_cmd = self._notify_command(message)

        try:
            result = subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    f"--on-active={total_seconds}s",
                    f"--unit={unit_name}",
                    "/bin/bash", "-c", notify_cmd,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return "systemd", unit_name
        except Exception:
            pass

        return "", None

    def _schedule_recurring_systemd(self, message: str, cron_expr: str) -> tuple[str, Optional[str]]:
        """Try to schedule recurring via systemd timer (OnCalendar)."""
        if not self._systemd_available():
            return "none", None

        # Map cron-like to systemd OnCalendar format
        calendar_map = {
            "0 9 * * *": "09:00",
            "0 18 * * *": "18:00",
            "0 * * * *": "hourly",
            "* * * * *": "minutely",
        }
        on_calendar = calendar_map.get(cron_expr, cron_expr)
        unit_name = f"lilim-recurring-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        notify_cmd = self._notify_command(message)

        try:
            result = subprocess.run(
                [
                    "systemd-run",
                    "--user",
                    f"--on-calendar={on_calendar}",
                    f"--unit={unit_name}",
                    "/bin/bash", "-c", notify_cmd,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                return "systemd", unit_name
        except Exception:
            pass

        return "none", None

    # ── Backend: threading.Timer ──────────────────────────────

    def _schedule_once_threading(self, message: str, delta: timedelta) -> tuple[str, Optional[str]]:
        """Fallback: use threading.Timer (dies with process, but better than nothing)."""
        def fire(msg):
            self.send_notification(msg)

        timer = threading.Timer(delta.total_seconds(), fire, args=[message])
        timer.daemon = True
        timer.start()
        # Store with a temporary key; we'll update after DB insert
        placeholder_id = id(timer)
        self._timers[placeholder_id] = timer
        return "threading", str(placeholder_id)

    # ── Time parsing ──────────────────────────────────────────

    def _parse_duration(self, text: str) -> Optional[timedelta]:
        """Parse 'in X minutes/hours/days' → timedelta."""
        text_lower = text.lower()
        for pattern, unit in DURATION_PATTERNS:
            m = re.search(pattern, text_lower)
            if m:
                amount = int(m.group(1))
                return timedelta(**{unit: amount})
        return None

    def _parse_recurring(self, text: str) -> tuple[Optional[str], str]:
        """Parse recurring schedule text → (cron_expr, human_readable)."""
        text_lower = text.lower()

        # "every day at 9am" / "every day at 9:30am"
        m = re.search(r"every\s+day\s+at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text_lower)
        if m:
            hour = int(m.group(1))
            minute = int(m.group(2) or 0)
            period = m.group(3) or "am"
            if period == "pm" and hour < 12:
                hour += 12
            if period == "am" and hour == 12:
                hour = 0
            cron = f"{minute} {hour} * * *"
            return cron, f"Every day at {hour:02d}:{minute:02d}"

        if "every morning" in text_lower:
            return "0 9 * * *", "Every morning at 09:00"

        if "every evening" in text_lower:
            return "0 18 * * *", "Every evening at 18:00"

        if "every hour" in text_lower or "hourly" in text_lower:
            return "0 * * * *", "Every hour"

        m = re.search(r"every\s+(\d+)\s+minutes?", text_lower)
        if m:
            n = int(m.group(1))
            return f"*/{n} * * * *", f"Every {n} minutes"

        if "every week" in text_lower or "weekly" in text_lower:
            return "0 9 * * 1", "Every Monday at 09:00"

        return None, ""

    # ── Helpers ───────────────────────────────────────────────

    def _systemd_available(self) -> bool:
        try:
            result = subprocess.run(
                ["systemctl", "--user", "status"],
                capture_output=True,
                timeout=3,
            )
            return result.returncode in (0, 3)  # 3 = "no units running" but daemon OK
        except Exception:
            return False

    def _notify_command(self, message: str) -> str:
        safe_msg = message.replace("'", "'\\''")
        return f"notify-send -a Lilim 'Lilim Reminder' '{safe_msg}' || echo '{safe_msg}' >> /tmp/lilim_reminders.log"

    def _save_schedule(self, message: str, trigger: str, schedule_type: str,
                        backend: str, job_id: Optional[str]) -> int:
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO schedules (message, trigger, schedule_type, backend, job_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (message, trigger, schedule_type, backend, job_id, now),
            )
            return cur.lastrowid

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))
