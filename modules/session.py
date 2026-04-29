"""Per-conversation session tracking with idle-based expiry.

The chat is otherwise stateless. This module gives every browser
conversation a server-minted UUID, records each turn against it, and
fires a single "session ended" log whenever the conversation has been
idle for ``SESSION_IDLE_TIMEOUT_SECONDS`` (default 60s) or the user
closes the tab (handled via the /api/chat/end endpoint, which calls
:meth:`SessionManager.end_session`).

Behaviour summary:

* First turn with no live session -> mint a new ``user_id`` (uuid4).
* Subsequent turns with the same ``user_id`` -> append, refresh activity.
* Stale ``user_id`` (already expired by the background sweeper) -> mint
  a fresh id; the client adopts whatever id the server returns. This
  matches the requirement that "if the same user rejoins after the
  session expires, the new conversation gets a different user id".
* Idle expiry / explicit end -> log_session_end() is called once with
  the full HTML transcript. The flow's ``sessionEnd: true`` branch
  fires the summary email.

Sessions live in process memory only. If the server is restarted
mid-conversation the in-flight session is lost (no email is sent for
it) - acceptable for this single-instance deployment.
"""

from __future__ import annotations

import asyncio
import html
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Iterable

from modules import logger as flow_logger

_log = logging.getLogger(__name__)

# How long a session can be idle before it is auto-ended. Configurable
# via env so deployments can dial it up if 60s feels aggressive.
DEFAULT_IDLE_TIMEOUT_SECONDS = 60
# How often the background sweeper wakes up to look for expired sessions.
DEFAULT_SWEEP_INTERVAL_SECONDS = 10


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class _Turn:
    user_input: str
    bot_output: str
    at: datetime
    language: str = "en"


@dataclass
class _Session:
    user_id: str
    started_at: datetime
    last_activity: datetime
    turns: list[_Turn] = field(default_factory=list)


class SessionManager:
    """In-memory session store with a background expiry sweeper.

    Thread-safe for synchronous mutations (chat handler runs in the
    starlette threadpool). The sweeper runs as an asyncio task on the
    main loop and uses ``run_in_executor`` for any blocking work it
    needs to dispatch (currently none - logger calls are themselves
    fire-and-forget).
    """

    def __init__(
        self,
        idle_timeout_seconds: int | None = None,
        sweep_interval_seconds: int | None = None,
    ) -> None:
        self._lock = RLock()
        self._sessions: dict[str, _Session] = {}

        if idle_timeout_seconds is None:
            idle_timeout_seconds = int(
                os.getenv(
                    "SESSION_IDLE_TIMEOUT_SECONDS",
                    str(DEFAULT_IDLE_TIMEOUT_SECONDS),
                )
            )
        if sweep_interval_seconds is None:
            sweep_interval_seconds = int(
                os.getenv(
                    "SESSION_SWEEP_INTERVAL_SECONDS",
                    str(DEFAULT_SWEEP_INTERVAL_SECONDS),
                )
            )
        self.idle_timeout = max(5, idle_timeout_seconds)
        self.sweep_interval = max(2, sweep_interval_seconds)

        self._sweeper_task: asyncio.Task | None = None

    # ── Public API ────────────────────────────────────────────────

    def get_or_create_user_id(self, supplied_id: str | None) -> str:
        """Return the active user id for this turn, minting one if needed.

        - If ``supplied_id`` matches a live session, reuse it.
        - Otherwise mint a fresh uuid4 and create a new session. This
          covers both first-time visitors (no id supplied) and stale
          ids whose session was already swept.
        """

        with self._lock:
            if supplied_id and supplied_id in self._sessions:
                return supplied_id

            new_id = str(uuid.uuid4())
            now = _now_utc()
            self._sessions[new_id] = _Session(
                user_id=new_id,
                started_at=now,
                last_activity=now,
            )
            _log.info("Session started: %s", new_id)
            return new_id

    def record_turn(
        self,
        user_id: str,
        user_input: str,
        bot_output: str,
        language: str = "en",
    ) -> None:
        """Append a completed turn to the session and emit a row payload.

        Silently no-ops if the session has already been ended (e.g. the
        user's tab-close beacon arrived between request start and
        response completion). Per-turn POST goes to the Power Automate
        flow's "add row" branch.
        """

        with self._lock:
            session = self._sessions.get(user_id)
            if session is None:
                _log.warning(
                    "record_turn called for unknown session %s; skipping",
                    user_id,
                )
                return
            now = _now_utc()
            session.turns.append(
                _Turn(
                    user_input=user_input,
                    bot_output=bot_output,
                    at=now,
                    language=language,
                )
            )
            session.last_activity = now

        flow_logger.log_turn(
            user_id=user_id,
            user_input=user_input,
            bot_output=bot_output,
            timestamp=_iso(now),
        )

    def end_session(self, user_id: str, *, reason: str = "explicit") -> None:
        """End a session immediately and fire the summary-email payload.

        Idempotent: a second call for the same user_id is a no-op.
        ``reason`` is logged for diagnostics ("idle", "explicit",
        "shutdown") but is not sent to the flow.
        """

        with self._lock:
            session = self._sessions.pop(user_id, None)
        if session is None:
            return

        if not session.turns:
            _log.info(
                "Session %s ended (%s) with no turns; skipping email",
                user_id,
                reason,
            )
            return

        transcript_html = _render_transcript_html(session)
        flow_logger.log_session_end(
            user_id=user_id,
            transcript_html=transcript_html,
            started_at=_iso(session.started_at),
            ended_at=_iso(_now_utc()),
            turn_count=len(session.turns),
        )
        _log.info(
            "Session %s ended (%s) with %d turn(s)",
            user_id,
            reason,
            len(session.turns),
        )

    def end_all(self, *, reason: str = "shutdown") -> None:
        """End every active session - used during graceful shutdown."""

        with self._lock:
            user_ids = list(self._sessions.keys())
        for uid in user_ids:
            self.end_session(uid, reason=reason)

    # ── Background sweeper ────────────────────────────────────────

    def start_sweeper(self) -> None:
        """Launch the background expiry task on the running event loop."""

        if self._sweeper_task and not self._sweeper_task.done():
            return
        loop = asyncio.get_event_loop()
        self._sweeper_task = loop.create_task(
            self._sweeper_loop(), name="session-sweeper"
        )
        _log.info(
            "Session sweeper started (idle=%ss, sweep=%ss)",
            self.idle_timeout,
            self.sweep_interval,
        )

    async def stop_sweeper(self) -> None:
        if not self._sweeper_task:
            return
        self._sweeper_task.cancel()
        try:
            await self._sweeper_task
        except (asyncio.CancelledError, Exception):  # pragma: no cover
            pass
        self._sweeper_task = None

    async def _sweeper_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.sweep_interval)
                self._sweep_once()
        except asyncio.CancelledError:
            raise
        except Exception:  # pragma: no cover
            _log.exception("session sweeper crashed; restarting in 10s")
            await asyncio.sleep(10)
            asyncio.create_task(self._sweeper_loop())

    def _sweep_once(self) -> None:
        cutoff = _now_utc().timestamp() - self.idle_timeout
        with self._lock:
            expired: list[str] = [
                uid
                for uid, sess in self._sessions.items()
                if sess.last_activity.timestamp() < cutoff
            ]
        for uid in expired:
            self.end_session(uid, reason="idle")

    # ── Test helpers ──────────────────────────────────────────────

    def _active_user_ids(self) -> Iterable[str]:
        with self._lock:
            return list(self._sessions.keys())


# ── HTML rendering ────────────────────────────────────────────────


def _render_transcript_html(session: _Session) -> str:
    """Render a session's turns as a self-contained HTML fragment.

    Used as the email body. Output is escaped: any user/bot text that
    contains HTML metacharacters is rendered as plain text, never as
    live markup.
    """

    parts: list[str] = []
    parts.append(
        f"<p><strong>UserID:</strong> {html.escape(session.user_id)}<br/>"
        f"<strong>Started:</strong> {html.escape(_iso(session.started_at))}<br/>"
        f"<strong>Ended:</strong> {html.escape(_iso(_now_utc()))}<br/>"
        f"<strong>Turns:</strong> {len(session.turns)}</p>"
        "<hr/>"
    )
    for idx, turn in enumerate(session.turns, start=1):
        parts.append(
            f"<p><strong>Turn {idx}</strong> "
            f"<em>({html.escape(_iso(turn.at))})</em></p>"
            f"<p><strong>User:</strong><br/>"
            f"{html.escape(turn.user_input).replace(chr(10), '<br/>')}</p>"
            f"<p><strong>Bot:</strong><br/>"
            f"{html.escape(turn.bot_output).replace(chr(10), '<br/>')}</p>"
            "<hr/>"
        )
    return "".join(parts)
