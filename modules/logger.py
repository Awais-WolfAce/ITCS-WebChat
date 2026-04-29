"""Power Automate flow logger for chat sessions.

Posts JSON payloads to a single Power Automate flow URL configured via the
``POWER_AUTOMATE_LOG_URL`` environment variable. The same flow handles two
payload shapes, distinguished by ``sessionEnd``:

* ``sessionEnd: false`` -> flow appends one row to the Excel ``ChatLog``
  table (one row per user/bot turn).
* ``sessionEnd: true``  -> flow sends one summary email containing the full
  HTML transcript of the session.

All sends are best-effort: failures are logged but never propagated, so a
broken flow URL or transient network error cannot break the user-facing
chat. Posts run in a daemon thread so they do not block the event loop.
"""

from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# How long to wait on the flow before giving up on a single attempt.
_REQUEST_TIMEOUT_SECONDS = 10.0
# Total retry attempts (including the first try) for transient errors.
_MAX_ATTEMPTS = 3
# Backoff between attempts, in seconds.
_RETRY_BACKOFF_SECONDS = 1.5


def _flow_url() -> str | None:
    """Read the flow URL fresh from the env each time.

    Resolving on every call (rather than caching at import) makes it easy
    for tests to override the URL via ``monkeypatch.setenv`` without
    re-importing the module.
    """

    url = os.getenv("POWER_AUTOMATE_LOG_URL", "").strip()
    return url or None


def _post(payload: dict[str, Any]) -> None:
    """Synchronously POST ``payload`` to the flow URL with retries.

    Runs on a worker thread spawned by the public helpers; never raises.
    """

    url = _flow_url()
    if not url:
        logger.debug("POWER_AUTOMATE_LOG_URL not set; skipping log: %s",
                     payload.get("userId"))
        return

    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            with httpx.Client(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
                response = client.post(url, json=payload)
            # Power Automate returns 202 Accepted on success.
            if 200 <= response.status_code < 300:
                return
            # 4xx - permanent client error, no point retrying.
            if 400 <= response.status_code < 500:
                logger.warning(
                    "Flow rejected payload (HTTP %s) for userId=%s: %s",
                    response.status_code,
                    payload.get("userId"),
                    response.text[:300],
                )
                return
            # 5xx and others fall through to retry.
            last_exc = RuntimeError(
                f"flow returned HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )
        except httpx.HTTPError as exc:
            last_exc = exc

        if attempt < _MAX_ATTEMPTS:
            threading.Event().wait(_RETRY_BACKOFF_SECONDS * attempt)

    logger.error(
        "Flow POST failed after %d attempts for userId=%s: %s",
        _MAX_ATTEMPTS,
        payload.get("userId"),
        last_exc,
    )


def _post_async(payload: dict[str, Any]) -> None:
    """Fire-and-forget: dispatch the POST on a daemon thread."""

    if not _flow_url():
        # Avoid spawning threads when logging is disabled entirely.
        return
    threading.Thread(
        target=_post,
        args=(payload,),
        name="flow-logger",
        daemon=True,
    ).start()


def _utc_iso() -> str:
    """Current UTC timestamp in ISO-8601 / RFC-3339 form (with 'Z')."""

    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def log_turn(
    user_id: str,
    user_input: str,
    bot_output: str,
    *,
    timestamp: str | None = None,
) -> None:
    """Send a single per-turn payload to the flow (Excel row branch).

    Args:
        user_id: Stable session id chosen by the SessionManager.
        user_input: Latest user message (post-translation, English).
        bot_output: Final bot response (post-translation, user's language).
        timestamp: Override for the row timestamp. Defaults to "now" UTC.
    """

    _post_async(
        {
            "userId": user_id,
            "input": user_input,
            "output": bot_output,
            "timestamp": timestamp or _utc_iso(),
            "sessionEnd": False,
        }
    )


def log_session_end(
    user_id: str,
    transcript_html: str,
    *,
    started_at: str | None = None,
    ended_at: str | None = None,
    turn_count: int | None = None,
) -> None:
    """Send the session-end payload to the flow (email branch).

    The ``transcript_html`` is rendered straight into the email body by
    the flow, so it should already be safe HTML.
    """

    _post_async(
        {
            "userId": user_id,
            # Flow's "Add a row" branch is gated on sessionEnd=false, so
            # these two are unused for this payload but kept in the schema
            # to satisfy the trigger's expected JSON shape.
            "input": "",
            "output": "",
            "timestamp": ended_at or _utc_iso(),
            "sessionEnd": True,
            "transcript": transcript_html,
            "startedAt": started_at or "",
            "endedAt": ended_at or _utc_iso(),
            "turnCount": turn_count if turn_count is not None else 0,
        }
    )
