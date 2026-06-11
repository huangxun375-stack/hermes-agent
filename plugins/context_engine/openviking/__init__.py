"""OpenViking full context lifecycle engine for Hermes.

Implements the ContextEngine lifecycle contract against a real OpenViking
server, mirroring the behavior validated by the OpenViking OpenClaw plugin:

- ``on_turn_complete``  — incremental turn ingest (watermark-idempotent) +
  threshold-triggered background commit (archive + memory extraction)
- ``prepare_request_messages`` — archive-summary + recall injection merged
  into the current user message (request-only view; returns ``None`` when
  there is nothing to inject so the prompt-cache prefix stays byte-stable)
- ``should_compress``/``compress`` — commit-based compaction: archive the
  session on the server, rebuild the live window as summary + recent tail
- ``on_pre_compress`` — flush not-yet-ingested messages before compaction
- ``ov_search`` tool — agent-invoked memory search

Configuration via environment variables (config.yaml integration can come
later):

    OPENVIKING_ENDPOINT   default http://127.0.0.1:2936
    OPENVIKING_API_KEY    default empty (local dev)
    OPENVIKING_ACCOUNT / OPENVIKING_USER / OPENVIKING_AGENT
    OPENVIKING_COMMIT_TOKEN_THRESHOLD   default 20000
    OPENVIKING_KEEP_RECENT_COUNT        default 10

Select with ``context.engine: openviking`` in config.yaml.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from typing import Any, Dict, List, Optional

from agent.context_engine import (
    ContextEngine,
    ContextEngineCapabilities,
    RequestContext,
    TurnInfo,
)

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:2936"
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
_BAD_PATH_SEGMENT = re.compile(r'[:<>"\\/|?\x00-\x1f]')

SUMMARY_HEADER = "[Session History Summary]"
RECALL_HEADER = "[Relevant Memories]"

OV_SEARCH_SCHEMA = {
    "name": "ov_search",
    "description": (
        "Search the OpenViking long-term memory (archived conversations and "
        "extracted memories) for facts not present in the current context. "
        "Use when the history summary mentions a topic but lacks the exact "
        "detail you need."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Natural-language search query"},
            "limit": {"type": "integer", "description": "Max results (default 5)"},
        },
        "required": ["query"],
    },
}


def _safe_ov_session_id(session_id: str) -> str:
    """Map a Hermes session id to a path-safe OpenViking session id.

    Same policy as the OpenViking OpenClaw plugin: UUIDs pass through
    lowercased; anything with Windows-hostile characters becomes a stable
    sha256.
    """
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("empty session id")
    if _UUID_RE.match(sid):
        return sid.lower()
    if _BAD_PATH_SEGMENT.search(sid):
        return hashlib.sha256(sid.encode("utf-8")).hexdigest()
    return sid


def _message_text(msg: Dict[str, Any]) -> str:
    """Flatten a Hermes message to plain text for OV ingest."""
    content = msg.get("content")
    parts: List[str] = []
    if isinstance(content, str) and content.strip():
        parts.append(content.strip())
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        fn = (tc or {}).get("function", {}) if isinstance(tc, dict) else {}
        name = fn.get("name", "unknown")
        args = fn.get("arguments", "")
        parts.append(f"[tool call: {name}({str(args)[:300]})]")
    if msg.get("role") == "tool":
        name = msg.get("name", "tool")
        parts.insert(0, f"[tool result: {name}]")
    return "\n".join(parts).strip()


class _OVClient:
    """Minimal OpenViking REST client (self-contained, fail-open callers)."""

    def __init__(self, endpoint: str, api_key: str = "",
                 account: str = "", user: str = "", agent: str = ""):
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._account = account
        self._user = user
        self._agent = agent

    def _headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if self._account:
            headers["X-OpenViking-Account"] = self._account
        if self._user:
            headers["X-OpenViking-User"] = self._user
        if self._agent:
            headers["X-OpenViking-Agent"] = self._agent
        return headers

    def _request(self, method: str, path: str, payload: dict = None,
                 params: dict = None, timeout: float = 30.0) -> dict:
        import httpx

        resp = httpx.request(
            method, f"{self._endpoint}{path}",
            json=payload, params=params,
            headers=self._headers(), timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "error":
            err = data.get("error") or {}
            raise RuntimeError(f"OV error {err.get('code')}: {err.get('message')}")
        if isinstance(data, dict) and "result" in data:
            return data["result"] or {}
        return data if isinstance(data, dict) else {}

    def health(self) -> bool:
        try:
            import httpx
            resp = httpx.get(f"{self._endpoint}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    def add_message(self, ov_sid: str, role: str, content: str) -> None:
        self._request("POST", f"/api/v1/sessions/{ov_sid}/messages",
                      {"role": role, "content": content})

    def get_session(self, ov_sid: str, auto_create: bool = True) -> dict:
        return self._request("GET", f"/api/v1/sessions/{ov_sid}",
                             params={"auto_create": str(auto_create).lower()})

    def get_context(self, ov_sid: str, token_budget: int) -> dict:
        return self._request("GET", f"/api/v1/sessions/{ov_sid}/context",
                             params={"token_budget": max(int(token_budget), 0)},
                             timeout=60.0)

    def commit(self, ov_sid: str, keep_recent_count: int = 0) -> dict:
        return self._request("POST", f"/api/v1/sessions/{ov_sid}/commit",
                             {"keep_recent_count": keep_recent_count},
                             timeout=180.0)

    def get_task(self, task_id: str) -> dict:
        return self._request("GET", f"/api/v1/tasks/{task_id}")

    def search(self, query: str, limit: int = 5) -> dict:
        return self._request("POST", "/api/v1/search/find",
                             {"query": query, "limit": limit}, timeout=60.0)


class OpenVikingContextEngine(ContextEngine):
    """Full-lifecycle OpenViking engine (observation + assembly + compaction)."""

    def __init__(self):
        self._endpoint = os.environ.get("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT)
        self._client = _OVClient(
            self._endpoint,
            api_key=os.environ.get("OPENVIKING_API_KEY", ""),
            account=os.environ.get("OPENVIKING_ACCOUNT", ""),
            user=os.environ.get("OPENVIKING_USER", ""),
            agent=os.environ.get("OPENVIKING_AGENT", ""),
        )
        self._commit_threshold = int(
            os.environ.get("OPENVIKING_COMMIT_TOKEN_THRESHOLD", "20000")
        )
        self._keep_recent = int(os.environ.get("OPENVIKING_KEEP_RECENT_COUNT", "10"))
        self._session_id: str = ""
        self._ov_sid: str = ""
        self._lineage: Dict[str, Any] = {}
        # Ingest watermark: count of messages already pushed to OV for the
        # bound session. Re-entry at the same watermark is a no-op.
        self._ingested_count = 0
        # One-shot marker: session id whose live content compress() just
        # archived. The host's compression-rotation transition calls
        # on_session_end(old_sid, OLD-window messages) right after
        # compress() returns, but by then compress() has already flushed
        # (via on_pre_compress) and committed that content AND rebased the
        # watermark into the REBUILT window's index space. Flushing the OLD
        # list against the rebased watermark would re-push the old tail
        # (duplicates in OV), commit a second archive, and leave the
        # watermark above the new window length — silently freezing
        # per-turn ingest. on_session_end consumes this marker and skips.
        self._just_compacted_session: str = ""
        self._bg_lock = threading.Lock()
        self._bg_thread: Optional[threading.Thread] = None
        # Circuit breaker: when the OV server is unreachable, hooks must
        # degrade FAST (pass through) instead of stalling every dispatch on
        # HTTP timeouts. Opens after _CB_THRESHOLD consecutive failures,
        # half-opens after _CB_COOLDOWN_S.
        self._cb_failures = 0
        self._cb_open_until = 0.0
        # Host abort contract: compress() must never raise; on failure it
        # returns the input unchanged and sets this flag (the host then skips
        # session rotation — see conversation_compression.py).
        self._last_compress_aborted = False
        # Assembly cache: the tool loop re-runs request assembly on every
        # provider dispatch; the archive overview only changes on ingest /
        # commit. Cache it per short TTL and invalidate on writes.
        self._ctx_cache: Optional[tuple] = None  # (expires_at, overview)

    _CB_THRESHOLD = 3
    _CB_COOLDOWN_S = 60.0
    _CTX_CACHE_TTL_S = 30.0

    # ── circuit breaker ──────────────────────────────────────────────────

    def _cb_allow(self) -> bool:
        import time as _time
        return _time.time() >= self._cb_open_until

    def _cb_ok(self) -> None:
        self._cb_failures = 0

    def _cb_fail(self) -> None:
        import time as _time
        self._cb_failures += 1
        if self._cb_failures >= self._CB_THRESHOLD:
            self._cb_open_until = _time.time() + self._CB_COOLDOWN_S
            self._cb_failures = 0
            logger.warning(
                "openviking: circuit breaker OPEN for %.0fs (server unreachable at %s)",
                self._CB_COOLDOWN_S, self._endpoint,
            )

    def _invalidate_ctx_cache(self) -> None:
        self._ctx_cache = None

    # ── identity / capabilities ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "openviking"

    def is_available(self) -> bool:
        return self._client.health()

    def capabilities(self) -> ContextEngineCapabilities:
        return ContextEngineCapabilities(
            observation=True,
            request_assembly=True,
            owns_compaction=True,
            lossless_snapshot=True,
            tools=True,
        )

    # ── token bookkeeping (standard) ─────────────────────────────────────

    def update_from_response(self, usage: Dict[str, Any]) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0) or 0
        self.last_completion_tokens = usage.get("completion_tokens", 0) or 0
        self.last_total_tokens = usage.get("total_tokens", 0) or 0

    # ── session lifecycle ────────────────────────────────────────────────

    def on_session_start(self, session_id: str, **kwargs) -> None:
        """Bind the OV session to the conversation's lineage root.

        Hermes rotates ``session_id`` on compaction (continuation of the SAME
        logical conversation) and on /new and /reset (a NEW conversation).
        The host disambiguates via lineage kwargs: ``lineage_root_id`` is
        propagated unchanged across compression rotations and regenerated on
        reset. Binding the OV session to the root therefore gives:

        - compaction rotation -> same OV session (archives accumulate,
          context stays continuous, ingest watermark survives)
        - /new and /reset      -> fresh OV session (clean slate, as intended)
        """
        # Quiesce the background ingest worker BEFORE rebinding: a worker
        # started for the previous binding advances the shared watermark with
        # OLD-list indices; letting it race past a rebind would poison the
        # fresh session's watermark.
        with self._bg_lock:
            t = self._bg_thread
        if t and t.is_alive():
            t.join(timeout=10.0)

        self._session_id = session_id or ""
        # A rebind means any pending compression transition has completed (or
        # the conversation changed) — a lingering marker is stale by now.
        self._just_compacted_session = ""
        anchor = kwargs.get("lineage_root_id") or session_id
        new_ov_sid = _safe_ov_session_id(anchor) if anchor else ""
        if new_ov_sid != self._ov_sid:
            # New logical conversation (or first bind): start a fresh watermark.
            self._ingested_count = 0
        # Same OV session (e.g. compression rotation): keep the watermark so
        # the rebuilt window's already-ingested tail is not pushed twice.
        self._ov_sid = new_ov_sid
        self._lineage = {
            "boundary_reason": kwargs.get("boundary_reason"),
            "lineage_root_id": kwargs.get("lineage_root_id"),
            "old_session_id": kwargs.get("old_session_id"),
        }
        logger.info(
            "openviking engine bound session=%s ov_sid=%s reason=%s root=%s",
            session_id, self._ov_sid,
            self._lineage.get("boundary_reason"),
            self._lineage.get("lineage_root_id"),
        )

    def on_session_end(self, session_id: str, messages: List[Dict[str, Any]]) -> None:
        """Flush remaining messages and commit at real session boundaries."""
        if session_id and session_id == self._just_compacted_session:
            # Compression-rotation boundary: compress() already archived this
            # session's content (on_pre_compress flushed, compress committed)
            # and rebased the watermark to the rebuilt window. Flushing the
            # OLD window here would duplicate messages, double-commit, and
            # freeze per-turn ingest (watermark > new window length).
            self._just_compacted_session = ""
            logger.info(
                "openviking: session %s end follows in-engine compaction — "
                "already flushed+committed, skipping end-flush", session_id)
            return
        try:
            self._ingest_new(messages)
            if self._ingested_count > 0 and self._ov_sid:
                self._client.commit(self._ov_sid, keep_recent_count=0)
                logger.info("openviking: session %s committed at session end", self._ov_sid)
        except Exception as e:
            logger.warning("openviking on_session_end failed: %s", e)

    def carry_over_new_session_context(self, old_session_id: str,
                                       new_session_id: str) -> None:
        logger.info("openviking: carry-over %s -> %s (same lineage)",
                    old_session_id, new_session_id)

    # ── observation / ingest ─────────────────────────────────────────────

    def _ingest_new(self, messages: List[Dict[str, Any]]) -> int:
        """Push messages beyond the watermark to OV. Returns count pushed.

        The watermark advances per message (not in bulk at the end), so a
        mid-batch failure leaves it at the exact failure point — the retry
        on the next turn resumes there with no duplicates and no gaps.
        """
        if not self._ov_sid:
            return 0
        start = min(self._ingested_count, len(messages))
        pushed = 0
        for i in range(start, len(messages)):
            msg = messages[i]
            role = msg.get("role")
            text = _message_text(msg) if role in ("user", "assistant", "tool") else ""
            if not text:
                self._ingested_count = i + 1  # nothing to push for this entry
                continue
            # OV session messages use user/assistant roles; tool results ride
            # along as user-role context (same policy as the OpenClaw plugin).
            ov_role = "assistant" if role == "assistant" else "user"
            self._client.add_message(self._ov_sid, ov_role, text[:8000])
            self._ingested_count = i + 1
            pushed += 1
        return pushed

    def on_turn_complete(self, messages: List[Dict[str, Any]], turn: TurnInfo) -> None:
        # Defensive rebind: if the host rotated the session without a full
        # transition notification (e.g. a bare reset path), detect it here.
        if turn.session_id and turn.session_id != self._session_id:
            self.on_session_start(turn.session_id)
        snapshot = list(messages)  # engine-side copy; never mutate host list

        if not self._cb_allow():
            return  # breaker open: skip fast; watermark self-heals next turn

        def _work():
            try:
                pushed = self._ingest_new(snapshot)
                self._cb_ok()
                if pushed == 0:
                    return
                self._invalidate_ctx_cache()
                info = self._client.get_session(self._ov_sid, auto_create=True)
                pending = int(info.get("pending_tokens") or 0)
                if pending >= self._commit_threshold:
                    result = self._client.commit(
                        self._ov_sid, keep_recent_count=self._keep_recent
                    )
                    self._invalidate_ctx_cache()
                    logger.info(
                        "openviking: threshold commit session=%s pending=%d task=%s",
                        self._ov_sid, pending, result.get("task_id"),
                    )
            except Exception as e:
                self._cb_fail()
                logger.warning("openviking on_turn_complete failed: %s", e)

        with self._bg_lock:
            if self._bg_thread and self._bg_thread.is_alive():
                self._bg_thread.join(timeout=10.0)
            self._bg_thread = threading.Thread(
                target=_work, daemon=True, name="openviking-turn-ingest"
            )
            self._bg_thread.start()

    # ── request assembly ─────────────────────────────────────────────────

    def _get_overview_cached(self, budget: int) -> str:
        """Archive overview with per-TTL cache, circuit-breaker guarded.

        The tool loop calls request assembly on EVERY provider dispatch; the
        overview only changes on ingest/commit (cache invalidated there).
        """
        import time as _time

        if self._ctx_cache and _time.time() < self._ctx_cache[0]:
            return self._ctx_cache[1]
        if not self._cb_allow():
            return self._ctx_cache[1] if self._ctx_cache else ""
        try:
            ov_ctx = self._client.get_context(self._ov_sid, budget)
            overview = (ov_ctx.get("latest_archive_overview") or "").strip()
            self._cb_ok()
            self._ctx_cache = (_time.time() + self._CTX_CACHE_TTL_S, overview)
            return overview
        except Exception as e:
            self._cb_fail()
            logger.debug("openviking: context fetch skipped: %s", e)
            return ""

    def prepare_request_messages(
        self, messages: List[Dict[str, Any]], ctx: RequestContext
    ) -> Optional[List[Dict[str, Any]]]:
        if not self._ov_sid:
            return None
        injections: List[str] = []
        budget = ctx.budget_tokens or self.threshold_tokens or 64_000
        overview = self._get_overview_cached(budget)
        if overview:
            injections.append(f"{SUMMARY_HEADER}\n{overview}")
        if ctx.prefetch_context:
            recall = "\n".join(str(x) for x in ctx.prefetch_context if x)
            if recall.strip():
                injections.append(f"{RECALL_HEADER}\n{recall.strip()}")
        if not injections:
            return None  # nothing to add → byte-stable prefix

        view = [dict(m) for m in messages]
        for m in reversed(view):
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                m["content"] = "\n\n".join(injections) + "\n\n" + m["content"]
                return view
        return None

    # ── compaction ───────────────────────────────────────────────────────

    def should_compress(self, prompt_tokens: int = None) -> bool:
        tokens = prompt_tokens if prompt_tokens else self.last_prompt_tokens
        return bool(self.threshold_tokens) and tokens >= self.threshold_tokens

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> None:
        # Salvage: make sure everything is in OV before the live window shrinks.
        try:
            self._ingest_new(list(messages))
        except Exception as e:
            logger.warning("openviking on_pre_compress flush failed: %s", e)

    def _wait_commit_task(self, task_id: str, timeout_s: float = 180.0,
                          interval_s: float = 2.0) -> str:
        """Client-side poll of the server commit task until it terminates.

        The OV commit endpoint is async (returns ``accepted`` + ``task_id``;
        the archive overview is produced by that server-side task). Same
        client-side-poll semantics the OpenClaw plugin implements for
        ``commitSession(wait=true)`` — there is no server wait parameter.
        Polling the task (not the overview) gives a deterministic terminal
        signal: ``failed`` stops immediately instead of waiting out a timeout.
        """
        import time as _time

        deadline = _time.time() + timeout_s
        while _time.time() < deadline:
            try:
                task = self._client.get_task(task_id)
                status = (task.get("status") or "").lower()
                if status in ("completed", "failed"):
                    if status == "failed":
                        logger.warning("openviking: commit task %s failed: %s",
                                       task_id, task.get("error"))
                    return status
            except Exception as e:
                logger.debug("openviking: task poll error: %s", e)
            _time.sleep(interval_s)
        logger.warning("openviking: commit task %s still running after %ss",
                       task_id, timeout_s)
        return "timeout"

    def compress(self, messages: List[Dict[str, Any]],
                 current_tokens: int = None, focus_topic: str = None
                 ) -> List[Dict[str, Any]]:
        # Failure semantics: the host treats a raised exception as fatal for
        # the turn (compress_context releases the lock and RE-RAISES), while
        # "return input unchanged + _last_compress_aborted" is the sanctioned
        # abort path that skips session rotation cleanly. Never raise here.
        self._last_compress_aborted = False
        self._just_compacted_session = ""  # fresh attempt; set only on success
        if not self._ov_sid:
            self._last_compress_aborted = True
            return messages
        self._invalidate_ctx_cache()
        try:
            result = self._client.commit(self._ov_sid, keep_recent_count=0)
            self._cb_ok()
        except Exception as e:
            self._cb_fail()
            logger.warning("openviking: compress commit failed — aborting "
                           "compaction (no rotation): %s", e)
            self._last_compress_aborted = True
            return messages
        # Compaction must consume the commit's product (the archive overview),
        # so wait for the server task to terminate — rare, already-slow path.
        task_id = result.get("task_id")
        if task_id:
            self._wait_commit_task(str(task_id))
        overview = ""
        try:
            ov_ctx = self._client.get_context(self._ov_sid, self.threshold_tokens or 64_000)
            overview = (ov_ctx.get("latest_archive_overview") or "").strip()
        except Exception as e:
            logger.warning("openviking: post-commit context fetch failed: %s", e)
        if not overview:
            logger.warning("openviking: archive overview unavailable — degraded rebuild (tail only)")
        self._invalidate_ctx_cache()  # next assembly reads the fresh post-commit state
        tail = [m for m in messages if m.get("role") in ("user", "assistant")]
        tail = tail[-max(self.protect_last_n, 2):]
        # Drop a leading assistant message to keep user-first alternation.
        while tail and tail[0].get("role") != "user":
            tail.pop(0)
        rebuilt: List[Dict[str, Any]] = []
        if overview:
            rebuilt.append({"role": "user", "content": f"{SUMMARY_HEADER}\n{overview}"})
            rebuilt.append({"role": "assistant",
                            "content": "Understood — I have the session history summary."})
        rebuilt.extend(dict(m) for m in tail)
        if not rebuilt:
            self._last_compress_aborted = True
            return messages
        self.compression_count += 1
        self.last_prompt_tokens = -1  # match built-in sentinel until next usage
        # The rebuilt window's contents (summary text + already-ingested tail)
        # are all known to the OV session — set the watermark past them so the
        # next turn only ingests genuinely new messages (no duplicates). The
        # upcoming compression-rotation on_session_start keeps this watermark
        # because the lineage root (and thus the OV binding) is unchanged.
        self._ingested_count = len(rebuilt)
        # Arm the one-shot marker: the imminent compression-rotation
        # transition will call on_session_end(this session) — that end-flush
        # must be skipped (content already archived; watermark already
        # rebased to the rebuilt window's index space).
        self._just_compacted_session = self._session_id
        return rebuilt

    # ── tools ────────────────────────────────────────────────────────────

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [OV_SEARCH_SCHEMA]

    def handle_tool_call(self, name: str, args: Dict[str, Any], **kwargs) -> str:
        if name != "ov_search":
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            result = self._client.search(
                str(args.get("query", "")), int(args.get("limit", 5) or 5)
            )
            return json.dumps(result, ensure_ascii=False, default=str)[:20_000]
        except Exception as e:
            return json.dumps({"error": str(e)})

    # ── status ───────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        status = super().get_status()
        status.update({
            "engine": "openviking",
            "endpoint": self._endpoint,
            "ov_session": self._ov_sid,
            "ingested_messages": self._ingested_count,
        })
        return status


def register(ctx) -> None:
    ctx.register_context_engine(OpenVikingContextEngine())
