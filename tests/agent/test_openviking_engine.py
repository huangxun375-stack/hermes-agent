"""Production-behavior tests for the OpenViking context engine plugin.

Covers the test-plan matrix items B3/B4/D1/D2/D4/E4/F1/F2/F3/F4/C6 with a
mocked OV client (no server needed):

- watermark idempotency + self-healing across skipped turns (B3, F3)
- per-turn assembly cache (F1) and circuit breaker degradation (F2/A9)
- lineage-root OV binding across compaction rotation vs reset (D1/D2/D4)
- commit-task failure -> degraded rebuild (C6)
- session-end flush (B4); ov_search error containment (F4)
"""

import json
import time
from unittest.mock import MagicMock

import pytest

from agent.context_engine import RequestContext, TurnInfo
from plugins.context_engine.openviking import (
    SUMMARY_HEADER,
    OpenVikingContextEngine,
    _safe_ov_session_id,
)


def make_engine(client=None):
    eng = OpenVikingContextEngine()
    eng._client = client if client is not None else MagicMock()
    eng.context_length = 100_000
    eng.threshold_tokens = 50_000
    return eng


def run_turn(eng, messages, session_id="sess-a"):
    """Invoke on_turn_complete and wait for the background worker."""
    eng.on_turn_complete(messages, TurnInfo(session_id=session_id))
    t = eng._bg_thread
    if t:
        t.join(timeout=10)


def msgs(*texts, roles=None):
    roles = roles or ["user", "assistant"] * len(texts)
    return [{"role": roles[i], "content": t} for i, t in enumerate(texts)]


# ── watermark: idempotency and self-healing ──────────────────────────────────


class TestWatermark:
    def test_no_duplicate_on_reentry(self):
        eng = make_engine()
        eng.on_session_start("sess-a")
        m = msgs("hello", "hi there")
        run_turn(eng, m)
        assert eng._client.add_message.call_count == 2
        run_turn(eng, m)  # same list again — nothing new
        assert eng._client.add_message.call_count == 2

    def test_skipped_turn_self_heals(self):
        """B3: a turn that bypassed finalize_turn is recovered next turn."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        run_turn(eng, msgs("t1 user", "t1 reply"))
        assert eng._ingested_count == 2
        # Turn 2 bypassed observation (early return); turn 3 sees ALL of it.
        full = msgs("t1 user", "t1 reply", "t2 user", "t2 reply",
                    "t3 user", "t3 reply",
                    roles=["user", "assistant"] * 3)
        run_turn(eng, full)
        assert eng._ingested_count == 6
        assert eng._client.add_message.call_count == 6

    def test_midbatch_failure_resumes_without_duplicates(self):
        """F3: watermark advances per message, so a failure resumes exactly."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        calls = []

        def flaky(sid, role, text):
            calls.append(text)
            if len(calls) == 2:
                raise RuntimeError("network blip")

        eng._client.add_message.side_effect = flaky
        m = msgs("a", "b", "c", "d", roles=["user", "assistant", "user", "assistant"])
        run_turn(eng, m)            # fails on message 2
        assert eng._ingested_count == 1
        eng._client.add_message.side_effect = None
        eng._cb_open_until = 0      # bypass cooldown for the retry
        run_turn(eng, m)            # resumes from message 2
        pushed = [c.args[2] for c in eng._client.add_message.call_args_list]
        assert pushed == ["a", "b", "b", "c", "d"][:len(pushed)]
        assert eng._ingested_count == 4

    def test_session_end_flushes_and_commits(self):
        """B4: real session boundary flushes the tail and commits."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.commit.return_value = {"task_id": "t1"}
        eng.on_session_end("sess-a", msgs("u", "a"))
        assert eng._client.add_message.call_count == 2
        eng._client.commit.assert_called_once()


# ── lineage-root binding across rotations ────────────────────────────────────


class TestRotationBinding:
    def test_compaction_rotation_keeps_ov_session_and_watermark(self):
        """D1: same lineage root -> same OV session, watermark preserved."""
        eng = make_engine()
        eng.on_session_start("root-1", boundary_reason="new",
                             lineage_root_id="root-1")
        run_turn(eng, msgs("u", "a"), session_id="root-1")
        assert eng._ingested_count == 2
        ov_before = eng._ov_sid
        # Host rotates the session id on compaction; root unchanged.
        eng.on_session_start("rotated-2", boundary_reason="compression",
                             old_session_id="root-1", lineage_root_id="root-1")
        assert eng._ov_sid == ov_before
        assert eng._ingested_count == 2  # watermark survives

    def test_reset_rebinds_and_clears_watermark(self):
        """D2: regenerated root -> fresh OV session + fresh watermark."""
        eng = make_engine()
        eng.on_session_start("root-1", boundary_reason="new",
                             lineage_root_id="root-1")
        run_turn(eng, msgs("u", "a"), session_id="root-1")
        eng.on_session_start("fresh-9", boundary_reason="reset",
                             lineage_root_id="fresh-9")
        assert eng._ov_sid == _safe_ov_session_id("fresh-9")
        assert eng._ingested_count == 0

    def test_defensive_rebind_on_session_drift(self):
        """D4: rotation without a transition notification is detected."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        run_turn(eng, msgs("u", "a"), session_id="drifted-b")
        assert eng._session_id == "drifted-b"
        assert eng._ov_sid == _safe_ov_session_id("drifted-b")

    def test_windows_hostile_session_id_is_hashed(self):
        assert "/" not in _safe_ov_session_id("agent:main/sub|x")
        uuid = "0e4b3c1a-2f3d-4e5f-8a9b-0c1d2e3f4a5b"
        assert _safe_ov_session_id(uuid.upper()) == uuid


# ── assembly: cache + circuit breaker ────────────────────────────────────────


class TestAssembly:
    def test_overview_cached_within_turn(self):
        """F1: multiple dispatches in one turn fetch the context once."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.get_context.return_value = {
            "latest_archive_overview": "summary text"}
        m = msgs("question")
        ctx = RequestContext(budget_tokens=1000)
        for _ in range(5):  # tool loop: 5 dispatches
            view = eng.prepare_request_messages(m, ctx)
            assert SUMMARY_HEADER in view[0]["content"]
        assert eng._client.get_context.call_count == 1

    def test_cache_invalidated_after_ingest(self):
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.get_context.return_value = {
            "latest_archive_overview": "v1"}
        eng._client.get_session.return_value = {"pending_tokens": 0}
        eng.prepare_request_messages(msgs("q"), RequestContext())
        run_turn(eng, msgs("u", "a"))  # ingest -> invalidate
        eng.prepare_request_messages(msgs("q2"), RequestContext())
        assert eng._client.get_context.call_count == 2

    def test_no_injection_returns_none(self):
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.get_context.return_value = {"latest_archive_overview": ""}
        assert eng.prepare_request_messages(msgs("q"), RequestContext()) is None

    def test_circuit_breaker_degrades_fast(self):
        """F2/A9: repeated failures open the breaker; hooks skip the network."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.get_context.side_effect = RuntimeError("conn refused")
        for _ in range(eng._CB_THRESHOLD):
            assert eng.prepare_request_messages(msgs("q"), RequestContext()) is None
        assert not eng._cb_allow()
        # Breaker open: no further network calls.
        before = eng._client.get_context.call_count
        eng.prepare_request_messages(msgs("q"), RequestContext())
        assert eng._client.get_context.call_count == before
        # Observation also skips fast while open.
        eng.on_turn_complete(msgs("u", "a"), TurnInfo(session_id="sess-a"))
        assert eng._client.add_message.call_count == 0

    def test_breaker_recovers_after_cooldown(self):
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.get_context.side_effect = RuntimeError("down")
        for _ in range(eng._CB_THRESHOLD):
            eng.prepare_request_messages(msgs("q"), RequestContext())
        eng._cb_open_until = time.time() - 1  # cooldown elapsed
        eng._client.get_context.side_effect = None
        eng._client.get_context.return_value = {
            "latest_archive_overview": "back"}
        view = eng.prepare_request_messages(msgs("q"), RequestContext())
        assert view is not None and SUMMARY_HEADER in view[0]["content"]


# ── compaction degradation ───────────────────────────────────────────────────


class TestCompaction:
    def test_commit_task_failure_degrades_to_tail_only(self):
        """C6: failed server task -> tail-only rebuild, no exception."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.commit.return_value = {"task_id": "t-9"}
        eng._client.get_task.return_value = {"status": "failed", "error": "boom"}
        eng._client.get_context.return_value = {"latest_archive_overview": ""}
        m = msgs("u1", "a1", "u2", "a2", roles=["user", "assistant"] * 2)
        rebuilt = eng.compress(m)
        text = json.dumps(rebuilt)
        assert SUMMARY_HEADER not in text          # degraded: no summary
        assert any(x.get("content") == "u1" for x in rebuilt)  # tail kept
        assert eng.compression_count == 1

    def test_compress_rebuild_sets_watermark_past_window(self):
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.commit.return_value = {"task_id": "t-1"}
        eng._client.get_task.return_value = {"status": "completed"}
        eng._client.get_context.return_value = {
            "latest_archive_overview": "the summary"}
        rebuilt = eng.compress(msgs("u1", "a1"))
        assert eng._ingested_count == len(rebuilt)
        assert SUMMARY_HEADER in rebuilt[0]["content"]


# ── tools ────────────────────────────────────────────────────────────────────


class TestTools:
    def test_ov_search_error_returns_json_error(self):
        """F4: tool failures surface as JSON, never raise into the loop."""
        eng = make_engine()
        eng._client.search.side_effect = RuntimeError("down")
        out = json.loads(eng.handle_tool_call("ov_search", {"query": "x"}))
        assert "error" in out

    def test_unknown_tool_contained(self):
        eng = make_engine()
        out = json.loads(eng.handle_tool_call("nope", {}))
        assert "error" in out
