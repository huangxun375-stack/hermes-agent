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


# ── rotation quiesce: background worker must not poison the new binding ─────


class TestRotationQuiesce:
    def test_session_start_joins_inflight_worker(self):
        """A slow background ingest started under the OLD binding must finish
        (or be joined) before a rebind — otherwise its old-list watermark
        writes would poison the fresh session's watermark."""
        import threading

        eng = make_engine()
        eng.on_session_start("old-root", lineage_root_id="old-root")
        release = threading.Event()
        started = threading.Event()

        def slow_add(sid, role, text):
            started.set()
            assert release.wait(timeout=5), "test deadlock"

        eng._client.add_message.side_effect = slow_add
        eng.on_turn_complete(msgs("u", "a"), TurnInfo(session_id="old-root"))
        assert started.wait(timeout=5)
        # Rebind while the worker is mid-flight; release it from another
        # thread so on_session_start's join can complete.
        threading.Timer(0.2, release.set).start()
        eng.on_session_start("new-root", boundary_reason="reset",
                             lineage_root_id="new-root")
        # Fresh binding has a clean watermark despite the in-flight worker.
        assert eng._ingested_count == 0
        assert eng._ov_sid == _safe_ov_session_id("new-root")


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

    def test_commit_exception_aborts_without_raising(self):
        """Host contract: compress() must NEVER raise (compress_context
        re-raises engine exceptions as fatal). OV-down -> sanctioned abort:
        input returned unchanged + _last_compress_aborted set, which makes
        the host skip session rotation cleanly."""
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._client.commit.side_effect = RuntimeError("connection refused")
        m = msgs("u1", "a1")
        out = eng.compress(m)
        assert out is m  # unchanged input object = host abort signal
        assert eng._last_compress_aborted is True
        assert eng.compression_count == 0

    def test_successful_compress_clears_abort_flag(self):
        eng = make_engine()
        eng.on_session_start("sess-a")
        eng._last_compress_aborted = True  # stale from a previous abort
        eng._client.commit.return_value = {"task_id": "t-1"}
        eng._client.get_task.return_value = {"status": "completed"}
        eng._client.get_context.return_value = {
            "latest_archive_overview": "the summary"}
        eng.compress(msgs("u1", "a1"))
        assert eng._last_compress_aborted is False

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


# ── compression-rotation interplay: end-flush must not duplicate ─────────────


class TestCompressionRotationInterplay:
    """Regression for the watermark index-space collision.

    compress() rebases the watermark into the REBUILT window's index space;
    the host's compression transition then calls on_session_end with the OLD
    window. Without the one-shot marker this re-pushed the old tail
    (duplicates), committed a second archive, and left the watermark above
    the new window length — silently freezing per-turn ingest.
    """

    def _compress_ready_engine(self):
        eng = make_engine()
        eng.on_session_start("root-1", boundary_reason="new",
                             lineage_root_id="root-1")
        eng._client.commit.return_value = {"task_id": "t-1"}
        eng._client.get_task.return_value = {"status": "completed"}
        eng._client.get_context.return_value = {
            "latest_archive_overview": "the summary"}
        return eng

    @staticmethod
    def _long_window(pairs=6):
        texts = []
        for i in range(pairs):
            texts += [f"u{i}", f"a{i}"]
        return msgs(*texts, roles=["user", "assistant"] * pairs)

    def test_session_end_after_compress_skips_flush_and_commit(self):
        eng = self._compress_ready_engine()
        # Old window LONGER than the rebuilt window (summary pair + tail) so
        # the unfixed index-space collision would actually re-push.
        old = self._long_window(pairs=6)            # 12 messages
        run_turn(eng, old, session_id="root-1")
        pushes_before = eng._client.add_message.call_count
        rebuilt = eng.compress(old)                  # rebased watermark
        assert len(rebuilt) < len(old)
        assert eng._client.commit.call_count == 1
        # Host compression transition: end(old) -> start(new, same root).
        eng.on_session_end("root-1", old)
        assert eng._client.add_message.call_count == pushes_before  # no re-push
        assert eng._client.commit.call_count == 1                   # no 2nd commit
        eng.on_session_start("rot-2", boundary_reason="compression",
                             lineage_root_id="root-1",
                             old_session_id="root-1")
        # Fingerprint assertion: the turn after rotation must ingest its NEW
        # messages (wm_delta > 0), exactly and only them.
        window = rebuilt + msgs("u-new", "a-new")
        run_turn(eng, window, session_id="rot-2")
        assert eng._ingested_count - len(rebuilt) == 2
        assert eng._client.add_message.call_count == pushes_before + 2

    def test_real_session_end_still_flushes_and_commits(self):
        """Guard: a session end NOT preceded by compress keeps full behavior."""
        eng = self._compress_ready_engine()
        eng.on_session_end("root-1", msgs("u", "a"))
        assert eng._client.add_message.call_count == 2
        eng._client.commit.assert_called_once()

    def test_marker_is_one_shot(self):
        """A later REAL end of the same session must flush+commit again."""
        eng = self._compress_ready_engine()
        old = self._long_window(pairs=6)
        run_turn(eng, old, session_id="root-1")
        eng.compress(old)
        eng.on_session_end("root-1", old)            # consumed: skip
        commits_after_skip = eng._client.commit.call_count
        eng.on_session_end("root-1", old)            # real end later: commit
        assert eng._client.commit.call_count == commits_after_skip + 1

    def test_aborted_compress_does_not_arm_marker(self):
        """Abort paths must leave session-end flush semantics untouched."""
        eng = self._compress_ready_engine()
        eng._client.commit.side_effect = RuntimeError("server down")
        old = self._long_window(pairs=2)
        out = eng.compress(old)
        assert out is old and eng._last_compress_aborted
        assert eng._just_compacted_session == ""
        eng._client.commit.side_effect = None
        eng._cb_open_until = 0
        eng.on_session_end("root-1", old)            # normal flush+commit
        assert eng._client.commit.call_count >= 1


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
