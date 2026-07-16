"""Transcript miner — bounded reader + state, recurrence, error-recovery.

Runner: python3 -m unittest tests.test_transcript_miner -v

Feature 260713-p2c-transcript-miner (Tasks 1-3). The miner is a SECOND, FULLY
OFFLINE prefer-queue producer over the operator's OWN corpus. Proven here:
  * Task 1 — bounded incremental read + the miner's OWN watermark/state: two
    consecutive incremental mines scan each session/turn EXACTLY once (no
    re-walk), a newly-appended session is picked up next mine, `--backfill`
    respects the per-run session cap (a 3rd session past a cap of 2 defers), an
    un-settled session is re-mined without double-emit, and the persistent
    recurrence tally survives across runs (cross-window accumulation, F1);
  * Task 2 — recurrence: a body across >=N DISTINCT sessions emits ONE candidate
    (recurrence_count == distinct sessions), split-across-windows still emits,
    twice-in-one-session counts once (F4), whitespace/case near-dups collapse,
    once-asked yields none, an emitted key does not re-emit as it climbs;
  * Task 3 — error-recovery: an ERR->ok pair on the same tool+target emits ONE
    factual, byte-stable candidate stored as inert data; a bare ERR yields none.

Feature 260714-miner-operationalize (Task 1) adds TestEmitOnceLedger — review
F-A: emit-once must survive LRU tally eviction via the bounded persistent
`emitted_keys` ledger (round-trips across reloads, stays <= MINER_EMITTED_CAP,
and a legacy state file without the key loads clean).

Hermetic: a temp KB (synthetic index + seeded logs), never the live KB (R-LOC-03).
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

try:
    from tests._fixtures import (load_cli, build_synthetic_kb,
                                 snapshot_corpus_kb, seed_miner_state_at_zero)
except ImportError:
    from _fixtures import (load_cli, build_synthetic_kb,
                           snapshot_corpus_kb, seed_miner_state_at_zero)


def _rec(ts, sid, cwd, origin, turn, prompt):
    """One prompts.log record in the exact log-prompt.sh format."""
    return ("[%s] [session %s] [cwd %s] [origin=%s] [turn=%s]\n%s\n\n"
            % (ts, sid, cwd, origin, turn, prompt))


class _MinerBase(unittest.TestCase):
    def setUp(self):
        self.kdir = tempfile.mkdtemp(prefix="aw_miner_")
        self.addCleanup(shutil.rmtree, self.kdir, True)
        build_synthetic_kb(self.kdir)
        self.mod = load_cli()
        os.makedirs(os.path.join(self.kdir, "logs", "sessions"), exist_ok=True)
        self.logp = os.path.join(self.kdir, "logs", "prompts.log")
        # HOME is the transcript root (_transcript_claude_path). Patch it to an
        # EMPTY temp home for every miner test so the reconciler can never read
        # the operator's real ~/.claude/projects (R-LOC-03) — without this, a
        # synthetic sid that happened to exist there would leak into a fixture.
        self.home = tempfile.mkdtemp(prefix="aw_home_")
        self.addCleanup(shutil.rmtree, self.home, True)
        p = mock.patch.dict(os.environ, {"HOME": self.home})
        p.start()
        self.addCleanup(p.stop)

    # --- corpus writers ---------------------------------------------------
    def _write_prompts(self, records):
        with open(self.logp, "w", encoding="utf-8") as f:
            f.write("".join(records))

    def _append_prompts(self, records):
        with open(self.logp, "a", encoding="utf-8") as f:
            f.write("".join(records))

    def _write_session(self, sid, calls, settled=True):
        sdir = os.path.join(self.kdir, "logs", "sessions", sid)
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "live.jsonl"), "w", encoding="utf-8") as f:
            for c in calls:
                f.write(json.dumps(c) + "\n")
        if settled:                       # main.jsonl present => Stop completed
            with open(os.path.join(sdir, "main.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"type":"user"}\n')

    def _write_transcript(self, sid, calls, proj="-Users-x-proj"):
        """Write a durable Claude session transcript for <sid> under the patched
        HOME, in the REAL shape verified live 2026-07-14: an assistant message
        carrying a tool_use block (id/name/input), then a user message carrying
        the matching tool_result (tool_use_id/is_error/content).
        calls: [(tool_name, input_obj, is_error, content, ts)]"""
        d = os.path.join(self.home, ".claude", "projects", proj)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, sid + ".jsonl"), "w", encoding="utf-8") as f:
            for i, (name, inp, is_err, content, ts) in enumerate(calls):
                tuid = "toolu_%s_%d" % (sid, i)
                f.write(json.dumps({
                    "type": "assistant", "sessionId": sid, "timestamp": ts,
                    "message": {"content": [{"type": "tool_use", "id": tuid,
                                             "name": name, "input": inp}]}}) + "\n")
                f.write(json.dumps({
                    "type": "user", "sessionId": sid, "timestamp": ts,
                    "message": {"content": [{"type": "tool_result",
                                             "tool_use_id": tuid,
                                             "is_error": is_err,
                                             "content": content}]}}) + "\n")

    def _call(self, tool, status, target=None, response="", input_override=None):
        inp = input_override if input_override is not None else (
            json.dumps({"file_path": target}) if target else "{}")
        return {"ts": "2026-07-13T00:00:00Z", "tool": tool, "status": status,
                "input": inp, "response": response}

    # --- state control ----------------------------------------------------
    def _seed_zero(self, **over):
        """Write a state that reads prompts from offset 0 and scans all sessions
        incrementally (empty processed/pending) — the primary detector setup."""
        st = {"offsets": {"prompts": 0, "metrics": 0},
              "processed_sessions": [], "backfill_pending": [], "tally": {}}
        st.update(over)
        self._write_state(st)

    def _write_state(self, st):
        p = os.path.join(self.kdir, self.mod.MINER_STATE_REL)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st, f)

    def _state(self):
        p = os.path.join(self.kdir, self.mod.MINER_STATE_REL)
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    def _mine(self, backfill=False, cap=None):
        return self.mod._transcript_mine(self.kdir, backfill=backfill,
                                         session_cap=cap)

    def _queue(self):
        q = os.path.join(self.kdir, self.mod.PREFER_QUEUE_REL)
        if not os.path.isfile(q):
            return []
        out = []
        with open(q, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out


class TestSeedAndWatermark(_MinerBase):
    def test_first_run_seeds_tail_and_mines_nothing(self):
        # A populated corpus present at first run is SEEDED (offsets=tail, all
        # sessions processed) so incremental mines only NEW material thereafter.
        self._write_prompts([_rec("t", "A", "/x", "human", "1",
                                  "Always pin versions.")])
        self._write_session("A", [self._call("Read", "ok", "/f")])
        self.assertEqual(self._mine(), 0, "first run seeds; mines nothing")
        st = self._state()
        self.assertEqual(st["offsets"]["prompts"], os.path.getsize(self.logp))
        self.assertIn("A", st["processed_sessions"])
        self.assertIn("A", st["backfill_pending"])
        self.assertEqual(self._queue(), [])

    def test_two_incremental_mines_scan_each_once(self):
        self._seed_zero()
        self._write_prompts([
            _rec("t", "A", "/x", "human", "1", "Always pin versions."),
            _rec("t", "B", "/x", "human", "1", "Always pin versions."),
        ])
        self._write_session("A", [self._call("Read", "ERR", "/f", "boom"),
                                  self._call("Read", "ok", "/f")])
        n1 = self._mine()
        q1 = len(self._queue())
        self.assertGreaterEqual(n1, 1)
        # Second mine over the SAME corpus: watermark advanced, session settled
        # and processed -> nothing re-walked, nothing re-emitted.
        self.assertEqual(self._mine(), 0, "no re-walk / no re-emit")
        self.assertEqual(len(self._queue()), q1)
        st = self._state()
        self.assertEqual(st["offsets"]["prompts"], os.path.getsize(self.logp))
        self.assertIn("A", st["processed_sessions"])

    def test_newly_appended_session_picked_up_next_mine(self):
        self._seed_zero()
        self._write_session("A", [self._call("Edit", "ERR", "/a", "x"),
                                  self._call("Edit", "ok", "/a")])
        self._mine()
        before = len(self._queue())
        # A genuinely NEW session appears after the first mine.
        self._write_session("B", [self._call("Edit", "ERR", "/b", "y"),
                                  self._call("Edit", "ok", "/b")])
        self.assertEqual(self._mine(), 1, "new session yields its pair")
        self.assertEqual(len(self._queue()), before + 1)

    def test_backfill_respects_session_cap_defers_third(self):
        # 3 settled sessions seeded as backfill-pending; a cap of 2 scans two now,
        # defers the third to the next backfill run.
        for s in ("s1", "s2", "s3"):
            self._write_session(s, [self._call("Bash", "ok", None,
                                               input_override='{"command":"ls"}')])
        self._seed_zero(backfill_pending=["s1", "s2", "s3"])
        self._mine(backfill=True, cap=2)
        st = self._state()
        self.assertEqual(sorted(st["processed_sessions"]), ["s1", "s2"])
        self.assertEqual(st["backfill_pending"], ["s3"], "3rd session deferred")
        self._mine(backfill=True, cap=2)
        st = self._state()
        self.assertEqual(sorted(st["processed_sessions"]), ["s1", "s2", "s3"])
        self.assertEqual(st["backfill_pending"], [])

    def test_unsettled_session_remined_without_double_emit(self):
        self._seed_zero()
        self._write_session("A", [self._call("Edit", "ERR", "/a", "x"),
                                  self._call("Edit", "ok", "/a")], settled=False)
        self.assertEqual(self._mine(), 1)
        self.assertEqual(len(self._queue()), 1)
        self.assertNotIn("A", self._state()["processed_sessions"],
                         "an un-settled session is not finalized")
        # Re-mine: the session is re-scanned (still not processed) but the pair
        # fp-dedups against the queue -> no second candidate.
        self.assertEqual(self._mine(), 0, "fp-dedup guards re-mine double-emit")
        self.assertEqual(len(self._queue()), 1)

    def test_tally_survives_across_runs_cross_window(self):
        self._seed_zero()
        self._append_prompts([_rec("t", "A", "/x", "human", "1",
                                   "Always squash commits before merge.")])
        self.assertEqual(self._mine(), 0, "one session -> below N, no emit")
        tally = self._state()["tally"]
        self.assertEqual(len(tally), 1, "tally persisted across the run")
        # A DIFFERENT session asks the same thing in a LATER window.
        self._append_prompts([_rec("t", "B", "/x", "human", "1",
                                   "Always squash commits before merge.")])
        self.assertEqual(self._mine(), 1, "distinct-session count reaches 2 -> emit")
        r = self._queue()[0]
        self.assertEqual(r["signal"], "recurrence")
        self.assertEqual(r["recurrence_count"], 2)


class TestRecurrence(_MinerBase):
    def test_three_distinct_sessions_one_candidate_count_three(self):
        self._seed_zero()
        body = "From now on, prefer tabs over spaces."
        self._write_prompts([_rec("t", s, "/x", "human", "1", body)
                             for s in ("A", "B", "C")])
        self.assertEqual(self._mine(), 1)
        r = self._queue()[0]
        self.assertEqual(r["signal"], "recurrence")
        self.assertEqual(r["recurrence_count"], 3)
        self.assertEqual(r["producer"], self.mod.PREFER_QUEUE_MINER_PRODUCER)
        self.assertEqual(r["status"], "queued")
        self.assertEqual(sorted(r["source_sessions"]), ["A", "B", "C"])
        # Stable fp: re-running the SAME window yields the same identity.
        fp = self.mod._prefer_cand_fp(r["sid"], r["turn_id"], r["text"])
        self.assertEqual(fp, self.mod._prefer_cand_fp(r["sid"], r["turn_id"],
                                                      r["text"]))

    def test_twice_in_one_session_counts_once(self):
        self._seed_zero()
        body = "Always run the linter."
        self._write_prompts([_rec("t", "A", "/x", "human", "1", body),
                             _rec("t", "A", "/x", "human", "2", body)])
        self.assertEqual(self._mine(), 0, "distinct-SESSION, not raw occurrence (F4)")
        self.assertEqual(self._queue(), [])

    def test_once_asked_yields_none(self):
        self._seed_zero()
        self._write_prompts([_rec("t", "A", "/x", "human", "1",
                                  "Always foo the bar.")])
        self.assertEqual(self._mine(), 0)

    def test_whitespace_case_nearups_collapse(self):
        self._seed_zero()
        self._write_prompts([
            _rec("t", "A", "/x", "human", "1", "Always Pin   Versions."),
            _rec("t", "B", "/x", "human", "1", "always pin versions"),
        ])
        self.assertEqual(self._mine(), 1, "near-dups collapse to one key")
        self.assertEqual(self._queue()[0]["recurrence_count"], 2)

    def test_emitted_key_does_not_reemit_as_count_climbs(self):
        self._seed_zero()
        body = "Never force-push to main."
        self._write_prompts([_rec("t", "A", "/x", "human", "1", body),
                             _rec("t", "B", "/x", "human", "1", body)])
        self.assertEqual(self._mine(), 1)
        # A THIRD distinct session asks it later; the key already emitted.
        self._append_prompts([_rec("t", "C", "/x", "human", "1", body)])
        self.assertEqual(self._mine(), 0, "emitted key never re-emits")
        self.assertEqual(len(self._queue()), 1)


class TestRecurrencePrecisionGate(_MinerBase):
    """Review F-C: _transcript_mine_recurrence emitted for ANY body reaching N
    distinct sessions, so high-frequency non-preference turns flooded the queue.
    Measured on the live corpus (2026-07-14) the ONLY key reaching N>=2 was the
    ack 'continue pls' — i.e. today recurrence emits noise and nothing else.
    The gate is EMISSION-side: the tally still counts, so it stays re-tunable."""

    def _mine_body_across(self, body, sids=("A", "B")):
        self._write_prompts([_rec("t", s, "/x", "human", "1", body)
                             for s in sids])
        return self._mine()

    # --- the measured live-corpus regression lock -------------------------
    def test_live_corpus_ack_continue_pls_is_gated(self):
        # THE regression lock: the one key that reached N>=2 on the real corpus.
        self._seed_zero()
        self.assertEqual(self._mine_body_across("continue pls"), 0,
                         "the measured live-corpus ack must never emit")
        self.assertEqual(self._queue(), [])

    def test_continue_pls_is_gated_by_BOTH_subtests_independently(self):
        # The plan requires each sub-test to gate it ALONE, so neither is
        # load-bearing by itself. Probe the pure predicate directly.
        key = self.mod._transcript_norm_key("continue pls")
        base = self.mod._transcript_recurrence_gate()
        # (a) length alone: disable the ack test (empty stopwords) -> still gated
        len_only = dict(base, stopwords=frozenset())
        self.assertFalse(self.mod._transcript_recurrence_relevant(key, len_only),
                         "length sub-test alone gates it")
        # (b) ack alone: disable the length test (min_len=0) -> still gated
        ack_only = dict(base, min_len=0)
        self.assertFalse(self.mod._transcript_recurrence_relevant(key, ack_only),
                         "ack sub-test alone gates it")

    def test_high_frequency_acks_across_n_sessions_emit_nothing(self):
        for ack in ("continue", "yes", "ok", "go ahead", "run it", "proceed",
                    "thanks", "continue please", "yes ok thanks"):
            with self.subTest(ack=ack):
                self.setUp()          # a clean KB per ack
                self._seed_zero()
                self.assertEqual(self._mine_body_across(ack), 0,
                                 "%r must not emit" % ack)

    def test_cue_bearing_preference_across_n_sessions_emits_exactly_one(self):
        self._seed_zero()
        self.assertEqual(
            self._mine_body_across("Always pin dependency versions."), 1)
        q = self._queue()
        self.assertEqual(len(q), 1)
        self.assertEqual(q[0]["signal"], "recurrence")

    def test_long_non_cue_all_ack_body_is_gated_by_the_blocklist(self):
        # >= min_len (so (a) passes) and carries NO cue, but every token is an
        # ack -> the blocklist is what gates it.
        body = "ok sure thanks that works great"
        key = self.mod._transcript_norm_key(body)
        gate = self.mod._transcript_recurrence_gate()
        self.assertGreaterEqual(len(key), gate["min_len"], "(a) passes")
        self.assertIsNone(self.mod._PREFER_QUEUE_RE.search(key), "no cue")
        self._seed_zero()
        self.assertEqual(self._mine_body_across(body), 0, "gated by blocklist")

    def test_long_non_cue_substantive_body_still_emits(self):
        # The ack test is TOKEN-level, not substring: a body with real content
        # must survive even without an explicit cue.
        self._seed_zero()
        self.assertEqual(
            self._mine_body_across("run the migrations against staging first"),
            1, "substantive non-cue body is not an ack")

    def test_ack_substring_inside_a_real_preference_does_not_gate_it(self):
        # "continue" appears, but the body is not ALL stopwords.
        key = self.mod._transcript_norm_key("do not continue on a failed build")
        gate = self.mod._transcript_recurrence_gate()
        self.assertFalse(self.mod._transcript_is_ack(key, gate["stopwords"]))
        self.assertTrue(self.mod._transcript_recurrence_relevant(key, gate))

    # --- the gate is re-tunable: the tally is unaffected -------------------
    def test_gated_key_is_still_counted_and_never_flagged_or_ledgered(self):
        # The core re-tunability contract (Task 6 tunes thresholds later): a
        # gated key must stay countable, unflagged, and OUT of the ledger.
        self._seed_zero()
        self.assertEqual(self._mine_body_across("continue pls", ("A", "B")), 0)
        st = self._state()
        key = self.mod._transcript_norm_key("continue pls")
        self.assertIn(key, st["tally"], "the tally COUNTS gated keys")
        self.assertEqual(len(st["tally"][key]["sessions"]), 2,
                         "distinct-session count is unaffected by the gate")
        self.assertFalse(st["tally"][key]["emitted"],
                         "a gated key is NOT flagged emitted")
        self.assertNotIn(key, st["emitted_keys"],
                         "a gated key never enters the emit-once ledger")

    def test_retuning_min_len_down_emits_a_previously_gated_key(self):
        # Proves re-tunability end-to-end: same corpus + a looser gate -> the
        # key that was gated now emits, because it was never flagged/ledgered.
        self._seed_zero()
        body = "use tabs"          # 8 chars, no cue, not an ack -> gated by (a)
        self.assertEqual(self._mine_body_across(body, ("A", "B")), 0)
        self._append_prompts([_rec("t", "C", "/x", "human", "1", body)])
        with mock.patch.dict(os.environ,
                             {self.mod.MINER_RECURRENCE_MIN_LEN_KEY: "4"}):
            self.assertEqual(self._mine(), 1, "a looser gate reconsiders it")
        self.assertEqual(self._queue()[0]["recurrence_count"], 3)

    # --- config plumbing ---------------------------------------------------
    def test_min_len_is_honored_from_env(self):
        self._seed_zero()
        with mock.patch.dict(os.environ,
                             {self.mod.MINER_RECURRENCE_MIN_LEN_KEY: "999"}):
            self.assertEqual(
                self._mine_body_across("Always pin dependency versions."), 0,
                "an absurd min_len gates even a cue-bearing preference")

    def test_cue_required_drops_the_not_an_ack_escape(self):
        self._seed_zero()
        body = "run the migrations against staging first"   # substantive, no cue
        with mock.patch.dict(
                os.environ, {self.mod.MINER_RECURRENCE_CUE_REQUIRED_KEY: "1"}):
            self.assertEqual(self._mine_body_across(body), 0,
                             "cue_required => a non-cue body never emits")

    def test_stopwords_are_honored_from_env(self):
        self._seed_zero()
        # "deploy on fridays" is not an ack by default, but becomes one when the
        # operator declares those tokens acks.
        body = "deploy on fridays"
        with mock.patch.dict(
                os.environ,
                {self.mod.MINER_RECURRENCE_STOPWORDS_KEY: "deploy,on,fridays"}):
            self.assertEqual(self._mine_body_across(body), 0,
                             "operator-supplied stopwords gate it")

    def test_empty_stopwords_config_disables_the_ack_subtest(self):
        gate = dict(self.mod._transcript_recurrence_gate(),
                    stopwords=self.mod._parse_miner_stopwords(""))
        self.assertEqual(gate["stopwords"], frozenset())
        key = self.mod._transcript_norm_key("yes ok thanks that works")
        self.assertTrue(self.mod._transcript_recurrence_relevant(key, gate),
                        "an explicitly empty stopword list disables (b)")

    def test_invalid_min_len_config_degrades_to_the_default(self):
        # A hand-edited typo must NOT silently disable the gate.
        for bad in ("", "abc", "-3", "  "):
            with self.subTest(bad=bad):
                self.assertIsNone(self.mod._parse_miner_min_len(bad))
        self.assertEqual(self.mod._parse_miner_min_len("0"), 0,
                         "0 is VALID and means 'no length gate'")
        with mock.patch.dict(os.environ,
                             {self.mod.MINER_RECURRENCE_MIN_LEN_KEY: "abc"}):
            self.assertEqual(self.mod._transcript_recurrence_gate()["min_len"],
                             self.mod.MINER_RECURRENCE_MIN_LEN)

    def test_gate_is_deterministic_across_calls(self):
        # INV-1: same key + same gate => same verdict, no clock/randomness.
        gate = self.mod._transcript_recurrence_gate()
        for key in ("continue pls", "always pin dependency versions", "yes"):
            k = self.mod._transcript_norm_key(key)
            verdicts = {self.mod._transcript_recurrence_relevant(k, gate)
                        for _ in range(5)}
            self.assertEqual(len(verdicts), 1, "byte-stable verdict for %r" % key)

    def test_error_recovery_emission_is_unaffected_by_the_gate(self):
        # The gate applies to RECURRENCE only: a short error-recovery candidate
        # (whose text would fail the length sub-test) must still emit.
        self._seed_zero()
        self._write_session("A", [self._call("Read", "ERR", "/f", "boom"),
                                  self._call("Read", "ok", "/f")])
        self.assertEqual(self._mine(), 1, "error-recovery is not gated")
        self.assertEqual(self._queue()[0]["signal"], "error-recovery")


class TestEmitOnceLedger(_MinerBase):
    """Review F-A: the per-entry `emitted` flag dies with its tally entry under
    LRU eviction, so an evicted-then-recurring key re-emitted a DUPLICATE whose
    fp (keyed on the NEWER occurrence) the archive/queue dedup could not catch.
    The bounded persistent `emitted_keys` ledger outlives the tally entry."""

    def test_evicted_key_recurring_across_new_sessions_does_not_reemit(self):
        # The F-A regression lock. Drives the REAL eviction path (tally past cap),
        # not a hand-written ledger: without the fix this mines a duplicate.
        self._seed_zero()
        x = "Always pin dependency versions, never use latest."
        y = "Prefer table-driven tests for parsers."
        with mock.patch.object(self.mod, "MINER_TALLY_CAP", 1):
            self._write_prompts([_rec("t", s, "/x", "human", "1", x)
                                 for s in ("A", "B")])
            self.assertEqual(self._mine(), 1, "X reaches N -> emits once")
            xkey = self.mod._transcript_norm_key(x)
            self.assertIn(xkey, self._state()["emitted_keys"])

            # A DIFFERENT key reaches N in a later window; the cap-1 prune then
            # LRU-evicts X (untouched this window => at the front of the tally).
            self._append_prompts([_rec("t", s, "/x", "human", "1", y)
                                  for s in ("D", "E")])
            self.assertEqual(self._mine(), 1, "Y emits")
            st = self._state()
            self.assertNotIn(xkey, st["tally"], "X was LRU-evicted from the tally")
            self.assertIn(xkey, st["emitted_keys"], "ledger OUTLIVES the entry")

            # X recurs across two NEW distinct sessions: its tally entry (and its
            # `emitted` flag) is gone, so only the ledger can suppress it.
            self._append_prompts([_rec("t", s, "/x", "human", "1", x)
                                  for s in ("F", "G")])
            self.assertEqual(self._mine(), 0,
                             "evicted-then-recurring key never re-emits (F-A)")
        self.assertEqual(len(self._queue()), 2, "only X and Y, no duplicate")

    def test_ledger_round_trips_across_state_reload(self):
        self._seed_zero()
        body = "Always squash commits before merge."
        self._write_prompts([_rec("t", s, "/x", "human", "1", body)
                             for s in ("A", "B")])
        self.assertEqual(self._mine(), 1)
        key = self.mod._transcript_norm_key(body)
        self.assertEqual(self._state()["emitted_keys"], [key])
        # Reload through the real loader: the ledger survives verbatim.
        st, fresh = self.mod._transcript_load_state(self.kdir)
        self.assertFalse(fresh)
        self.assertEqual(st["emitted_keys"], [key])

    def test_ledger_is_bounded_lru_dropping_oldest(self):
        self._seed_zero()
        bodies = ["Always prefer approach number %d here." % i for i in range(3)]
        with mock.patch.object(self.mod, "MINER_EMITTED_CAP", 2):
            for i, b in enumerate(bodies):
                self._append_prompts([
                    _rec("t", "s%d%s" % (i, s), "/x", "human", "1", b)
                    for s in ("a", "b")])
                self.assertEqual(self._mine(), 1)
            led = self._state()["emitted_keys"]
        self.assertEqual(len(led), 2, "ledger stays <= MINER_EMITTED_CAP")
        self.assertEqual(led, [self.mod._transcript_norm_key(b)
                               for b in bodies[1:]], "oldest-emitted dropped")

    def test_legacy_state_without_emitted_keys_loads_clean(self):
        # The LIVE state file is exactly this shape (written by the old build).
        self._write_state({"offsets": {"prompts": 0, "metrics": 0},
                           "processed_sessions": [], "backfill_pending": [],
                           "tally": {}})
        st, fresh = self.mod._transcript_load_state(self.kdir)
        self.assertFalse(fresh, "a legacy state file is NOT re-seeded at the tail")
        self.assertEqual(st["emitted_keys"], [], "back-compat: absent -> []")
        # And it still mines normally through the ledger path.
        body = "Never commit generated artifacts to the repo."
        self._write_prompts([_rec("t", s, "/x", "human", "1", body)
                             for s in ("A", "B")])
        self.assertEqual(self._mine(), 1)
        self.assertEqual(self._state()["emitted_keys"],
                         [self.mod._transcript_norm_key(body)])

    def test_ledger_normalization_rejects_junk_and_dedups(self):
        norm = self.mod._transcript_norm_emitted
        self.assertEqual(norm(None), [], "a missing ledger normalizes to []")
        self.assertEqual(norm("nope"), [], "a non-list normalizes to []")
        self.assertEqual(norm(["a", None, 3, "", "b", "a"]), ["a", "b"],
                         "junk dropped, order-preserving dedup")
        with mock.patch.object(self.mod, "MINER_EMITTED_CAP", 2):
            self.assertEqual(norm(["a", "b", "c"]), ["b", "c"],
                             "an over-cap ledger tail-caps to the NEWEST keys")

    def test_fresh_seed_state_carries_an_empty_ledger(self):
        st = self.mod._transcript_default_state()
        self.assertEqual(st["emitted_keys"], [])


class TestErrorRecovery(_MinerBase):
    def test_error_then_fix_pair_yields_one_candidate(self):
        self._seed_zero()
        self._write_session("A", [
            self._call("Read", "ERR", "/tmp/missing.md",
                       "No such file or directory"),
            self._call("Read", "ok", "/tmp/missing.md", "contents"),
        ])
        self.assertEqual(self._mine(), 1)
        r = self._queue()[0]
        self.assertEqual(r["signal"], "error-recovery")
        self.assertEqual(r["producer"], self.mod.PREFER_QUEUE_MINER_PRODUCER)
        self.assertIn("error-recovery", r["text"])
        self.assertIn("/tmp/missing.md", r["text"])
        self.assertEqual(r["source_sessions"], ["A"])

    def test_error_recovery_text_byte_stable(self):
        tool, target, sig = "Edit", "/a/b.py", "No such file or directory"
        t1 = self.mod._transcript_error_recovery_text(tool, target, sig)
        t2 = self.mod._transcript_error_recovery_text(tool, target, sig)
        self.assertEqual(t1, t2)
        self.assertNotIn("\n", t1, "derived text is a single inert line")

    def test_bare_unrecovered_error_yields_none(self):
        self._seed_zero()
        self._write_session("A", [self._call("Read", "ERR", "/x", "boom")])
        self.assertEqual(self._mine(), 0)
        self.assertEqual(self._queue(), [])

    def test_error_recovery_text_is_inert_data(self):
        # Injection-looking error text is stored VERBATIM inside the candidate as
        # data — never executed/interpolated (R-SEC-02).
        self._seed_zero()
        evil = "$(rm -rf /) `curl evil` -- boom"
        self._write_session("A", [self._call("Bash", "ERR", None, evil,
                                  input_override='{"command":"deploy.sh"}'),
                                  self._call("Bash", "ok", None,
                                  input_override='{"command":"deploy.sh"}')])
        self.assertEqual(self._mine(), 1)
        r = self._queue()[0]
        self.assertEqual(r["status"], "queued")   # inert; not run
        self.assertIn("deploy.sh", r["text"])


class TestClaudeTranscriptReconciliation(_MinerBase):
    """Feature 260714-miner-operationalize Task 3 / review F-B.

    The PostToolUse hook does NOT fire on a FAILED Claude tool call (spike,
    verified 2026-07-14: `ls <missing>`, `false` and `exit 7` each produced ZERO
    live.jsonl records while every success logged), so error-recovery was blind
    on the majority harness. These lock the reconstruction of the missing ERR
    record from the durable transcript's tool_result `is_error`."""

    TS_ERR = "2026-07-14T10:00:00.123Z"
    TS_FIX = "2026-07-14T10:05:00.456Z"

    def test_failed_claude_bash_is_captured_as_ERR_and_pairs_with_a_later_fix(self):
        # THE headline case: the hook logged only the later SUCCESS; the failure
        # exists solely in the transcript. Pre-fix this mined nothing at all.
        self._seed_zero()
        cmd = "pytest tests/test_api.py"
        self._write_session("A", [
            {"ts": "2026-07-14T10:05:00Z", "tool": "Bash", "status": "ok",
             "input": json.dumps({"command": cmd}), "response": "2 passed"},
        ])
        self._write_transcript("A", [
            ("Bash", {"command": cmd}, True,
             "Exit code 1\nModuleNotFoundError: No module named 'api'",
             self.TS_ERR),
            ("Bash", {"command": cmd}, False, "2 passed", self.TS_FIX),
        ])
        self.assertEqual(self._mine(), 1)
        r = self._queue()[0]
        self.assertEqual(r["signal"], "error-recovery")
        self.assertEqual(r["source_sessions"], ["A"])
        self.assertIn("Bash", r["text"])
        self.assertIn(cmd, r["text"])                     # target, from tool_use
        # The signature is _transcript_err_signature's FIRST non-blank response
        # line — for a Claude failure that is the "Exit code N" preamble, with the
        # informative detail on line 2. Factual + deterministic, but low-signal;
        # recorded as carry-forward rather than changing shared, test-locked
        # signature semantics inside this task's scope.
        self.assertIn("Exit code 1", r["text"])

    def test_reconciler_emits_ERR_records_only_for_is_error_true(self):
        self._write_transcript("A", [
            ("Bash", {"command": "ok-one"}, False, "fine", self.TS_ERR),
            ("Bash", {"command": "bad-one"}, True, "boom", self.TS_FIX),
        ])
        errs = self.mod._transcript_reconcile_errors("A", [])
        self.assertEqual(len(errs), 1, "only the is_error:true result is rebuilt")
        self.assertEqual(errs[0]["status"], "ERR")
        self.assertEqual(errs[0]["tool"], "Bash")
        self.assertEqual(self.mod._transcript_tool_target(errs[0]), "bad-one")

    def test_reconstructed_record_matches_the_hook_schema(self):
        # Byte-compatible with what log-tool.sh would have written, so every
        # downstream reader (detector, signature, target) works unchanged.
        self._write_transcript("A", [
            ("Bash", {"command": "x"}, True, "boom", self.TS_ERR)])
        rec = self.mod._transcript_reconcile_errors("A", [])[0]
        self.assertEqual(sorted(rec), ["agent", "input", "response", "status",
                                       "tool", "ts"])
        self.assertIsInstance(rec["input"], str)      # hook stores JSON strings
        self.assertIsInstance(rec["response"], str)

    def test_transcript_ts_normalized_to_hook_second_granularity(self):
        self._write_transcript("A", [
            ("Bash", {"command": "x"}, True, "boom", self.TS_ERR)])
        rec = self.mod._transcript_reconcile_errors("A", [])[0]
        self.assertEqual(rec["ts"], "2026-07-14T10:00:00Z")

    def test_records_merge_onto_one_ordered_timeline(self):
        self._seed_zero()
        self._write_session("A", [
            {"ts": "2026-07-14T10:05:00Z", "tool": "Bash", "status": "ok",
             "input": json.dumps({"command": "c"}), "response": "fixed"},
        ])
        self._write_transcript("A", [
            ("Bash", {"command": "c"}, True, "boom", self.TS_ERR)])
        sdir = os.path.join(self.kdir, "logs", "sessions", "A")
        recs = self.mod._transcript_read_session(sdir, "A")
        self.assertEqual([r["status"] for r in recs], ["ERR", "ok"],
                         "the 10:00 reconstructed ERR precedes the 10:05 fix")

    def test_error_after_the_fix_does_not_pair_backwards(self):
        # Ordering is load-bearing: a failure AFTER the success is not recovery.
        self._seed_zero()
        self._write_session("A", [
            {"ts": "2026-07-14T09:00:00Z", "tool": "Bash", "status": "ok",
             "input": json.dumps({"command": "c"}), "response": "fine"},
        ])
        self._write_transcript("A", [
            ("Bash", {"command": "c"}, True, "boom", "2026-07-14T11:00:00.0Z")])
        self.assertEqual(self._mine(), 0)

    def test_session_without_a_transcript_is_untouched(self):
        # The codex path (no Claude transcript) keeps its own exit-code ERRs.
        self._seed_zero()
        self._write_session("A", [
            self._call("command_execution", "ERR", None, "boom",
                       input_override="deploy.sh"),
            self._call("command_execution", "ok", None, "done",
                       input_override="deploy.sh"),
        ])
        self.assertEqual(self.mod._transcript_reconcile_errors("A", []), [])
        self.assertEqual(self._mine(), 1, "codex ERR->ok still pairs")

    def test_absent_or_unreadable_transcript_degrades_to_empty(self):
        self.assertEqual(self.mod._transcript_reconcile_errors("nope", []), [])
        self.assertEqual(self.mod._transcript_claude_path("nope"), "")

    def test_malformed_transcript_lines_are_skipped_not_fatal(self):
        d = os.path.join(self.home, ".claude", "projects", "p")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "A.jsonl"), "w", encoding="utf-8") as f:
            f.write("not json\n")
            f.write(json.dumps({"message": {"content": "not-a-list"}}) + "\n")
            f.write(json.dumps({"message": {"content": [
                {"type": "tool_use", "id": "t1", "name": "Bash",
                 "input": {"command": "x"}}]}, "timestamp": self.TS_ERR}) + "\n")
            f.write(json.dumps({"message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": True,
                 "content": "boom"}]}, "timestamp": self.TS_ERR}) + "\n")
        errs = self.mod._transcript_reconcile_errors("A", [])
        self.assertEqual(len(errs), 1)

    def test_unpaired_tool_result_is_ignored(self):
        d = os.path.join(self.home, ".claude", "projects", "p")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "A.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"message": {"content": [
                {"type": "tool_result", "tool_use_id": "orphan",
                 "is_error": True, "content": "boom"}]},
                "timestamp": self.TS_ERR}) + "\n")
        self.assertEqual(self.mod._transcript_reconcile_errors("A", []), [])

    def test_writer_recorded_err_is_not_duplicated(self):
        # A tool whose response DID carry success=false is already in live.jsonl;
        # the reconstruction must not double it.
        live = [{"ts": "2026-07-14T10:00:00Z", "tool": "Bash", "status": "ERR",
                 "input": json.dumps({"command": "x"}), "response": "boom"}]
        self._write_transcript("A", [
            ("Bash", {"command": "x"}, True, "boom", self.TS_ERR)])
        self.assertEqual(self.mod._transcript_reconcile_errors("A", live), [])

    def test_a_second_failure_at_a_different_time_is_still_rebuilt(self):
        # The dedup keys on (tool, target, ts) — never collapsing real repeats.
        live = [{"ts": "2026-07-14T10:00:00Z", "tool": "Bash", "status": "ERR",
                 "input": json.dumps({"command": "x"}), "response": "boom"}]
        self._write_transcript("A", [
            ("Bash", {"command": "x"}, True, "boom", self.TS_ERR),
            ("Bash", {"command": "x"}, True, "boom again", self.TS_FIX)])
        errs = self.mod._transcript_reconcile_errors("A", live)
        self.assertEqual([e["ts"] for e in errs], ["2026-07-14T10:05:00Z"])

    def test_sid_path_traversal_is_refused(self):
        for evil in ("../../etc/passwd", "a/b", "..", ".", "", None, 7):
            self.assertEqual(self.mod._transcript_claude_path(evil), "")

    def test_reconciler_is_read_only_and_deterministic(self):
        self._write_transcript("A", [
            ("Bash", {"command": "x"}, True, "boom", self.TS_ERR)])
        tpath = self.mod._transcript_claude_path("A")
        before = open(tpath, encoding="utf-8").read()
        a = self.mod._transcript_reconcile_errors("A", [])
        b = self.mod._transcript_reconcile_errors("A", [])
        self.assertEqual(a, b, "byte-stable across calls (INV-1)")
        self.assertEqual(open(tpath, encoding="utf-8").read(), before)

    def test_error_text_from_a_transcript_stays_inert(self):
        # R-SEC-02: transcript content is untrusted data, stored verbatim.
        self._seed_zero()
        evil = "$(rm -rf /) `curl evil` -- boom"
        self._write_session("A", [
            {"ts": "2026-07-14T10:05:00Z", "tool": "Bash", "status": "ok",
             "input": json.dumps({"command": "deploy.sh"}), "response": "done"}])
        self._write_transcript("A", [
            ("Bash", {"command": "deploy.sh"}, True, evil, self.TS_ERR),
            ("Bash", {"command": "deploy.sh"}, False, "done", self.TS_FIX)])
        self.assertEqual(self._mine(), 1)
        self.assertEqual(self._queue()[0]["status"], "queued")  # inert, not run


class TestTruncatedInputTargetGuard(_MinerBase):
    """Feature 260714-miner-operationalize Task 3 / review F-D.

    log-tool.sh bounds `input` at MAXLEN=1500 + a marker, so a long call's stored
    input is INVALID JSON. Pre-fix, json.loads raised and the WHOLE blob became
    the target via _transcript_norm_key — wrong, and collision-prone across two
    commands sharing a 1500-char prefix."""

    def _truncated(self, cmd):
        """Exactly what log-tool.sh writes for an over-long tool_input."""
        blob = json.dumps({"command": cmd, "description": "x" * 200})
        return self.mod._transcript_hook_trunc(blob)

    def test_truncated_input_yields_the_command_never_the_blob(self):
        cmd = "deploy --region us-east-1 " + ("--flag " * 400)
        raw = self._truncated(cmd)
        self.assertGreater(len(raw), self.mod.MINER_TRANSCRIPT_MAXLEN)
        target = self.mod._transcript_tool_target({"input": raw})
        self.assertTrue(target.startswith("deploy --region us-east-1"))
        self.assertNotIn("[truncated]", target, "the marker never leaks in")
        self.assertNotIn('"description"', target, "the blob is not the target")
        self.assertLess(len(target), len(raw))

    def test_two_commands_sharing_a_long_prefix_do_not_collide(self):
        # The pre-fix collision: an unrelated success 'recovering' an error.
        base = "run " + ("x" * 1600)
        a = self.mod._transcript_tool_target({"input": self._truncated(base + " --alpha")})
        b = self.mod._transcript_tool_target({"input": self._truncated(base + " --beta")})
        # Both truncate inside the shared prefix, so they legitimately agree;
        # what must hold is that neither is the raw blob.
        self.assertNotIn("[truncated]", a)
        self.assertEqual(a, b)
        short_a = self.mod._transcript_tool_target(
            {"input": json.dumps({"command": "run --alpha"})})
        short_b = self.mod._transcript_tool_target(
            {"input": json.dumps({"command": "run --beta"})})
        self.assertNotEqual(short_a, short_b, "distinct short commands stay distinct")

    def test_truncated_err_and_fix_derive_the_same_target_and_pair(self):
        # The reason parity with the writer's truncation matters at all.
        self._seed_zero()
        cmd = "pytest " + ("tests/test_%d.py " % 1) * 400
        raw = self._truncated(cmd)
        self._write_session("A", [
            {"ts": "2026-07-14T10:00:00Z", "tool": "Bash", "status": "ERR",
             "input": raw, "response": "boom"},
            {"ts": "2026-07-14T10:05:00Z", "tool": "Bash", "status": "ok",
             "input": raw, "response": "passed"},
        ])
        self.assertEqual(self._mine(), 1)

    def test_truncated_precise_target_is_preferred_over_command(self):
        blob = json.dumps({"file_path": "/a/b.py", "command": "z" * 2000})
        raw = self.mod._transcript_hook_trunc(blob)
        self.assertEqual(self.mod._transcript_tool_target({"input": raw}),
                         "/a/b.py")

    def test_valid_json_input_is_unaffected(self):
        self.assertEqual(self.mod._transcript_tool_target(
            {"input": json.dumps({"file_path": "/a/b.py"})}), "/a/b.py")
        self.assertEqual(self.mod._transcript_tool_target(
            {"input": json.dumps({"command": "  ls -la  "})}), "ls -la")

    def test_plain_non_json_string_input_keeps_the_normalized_path(self):
        # codex-stream.py writes the BARE command as `input` — not JSON.
        self.assertEqual(self.mod._transcript_tool_target({"input": "  ls -la "}),
                         "ls -la")

    def test_brace_shell_group_is_not_mistaken_for_truncated_json(self):
        # A codex command may literally start with '{' — it must not be salvaged
        # into '' and lose its target.
        # norm_key strips the surrounding brace/semicolon punctuation.
        self.assertEqual(
            self.mod._transcript_tool_target({"input": "{ echo hi; }"}),
            "echo hi")

    def test_unrecoverable_blob_yields_no_target_not_a_bogus_one(self):
        raw = '{"comm'          # truncation cut before any field value
        self.assertEqual(self.mod._transcript_tool_target({"input": raw}), "")

    def test_field_extractor_handles_escapes_and_severed_escapes(self):
        ex = self.mod._transcript_extract_json_field
        self.assertEqual(ex(r'{"command":"say \"hi\" now"}', "command"),
                         'say "hi" now')
        self.assertEqual(ex(r'{"command":"a\\b"}', "command"), r"a\b")
        self.assertEqual(ex('{"command":"tail-cut', "command"), "tail-cut")
        self.assertEqual(ex('{"command":"bad\\', "command"), "bad")
        self.assertEqual(ex('{"command":"x\\u00', "command"), "x")
        self.assertEqual(ex('{"other":"x"}', "command"), "")

    def test_target_derivation_is_byte_stable(self):
        raw = self._truncated("deploy " + "y" * 2000)
        self.assertEqual(self.mod._transcript_tool_target({"input": raw}),
                         self.mod._transcript_tool_target({"input": raw}))


class TestSnapshotBackfillHarness(unittest.TestCase):
    """Feature 260714-miner-operationalize, Task 5 — the SNAPSHOT-BACKFILL harness.

    Locks the harness the operator's real-corpus measurement runs on (D4: measure
    a COPY, never mutate the live KB). The source corpus here is synthetic and
    live-SHAPED — a preference repeated across 3 distinct sessions, ack noise, and
    an error->fix pair — so the harness contract is proven hermetically while the
    same `snapshot_corpus_kb` + `seed_miner_state_at_zero` code path carries the
    real measurement (one definition, no drift).

    THE CENTRAL LOCK is test_selfseeded_snapshot_mines_nothing_the_p1_trap: it
    proves the pre-seed is LOAD-BEARING. Without it a snapshot self-seeds
    `offsets.prompts` at EOF, `--backfill` does not rewind it, and the harness
    mines ZERO while every other assertion still passes — a vacuous green. That
    test fails loudly if anyone ever deletes the pre-seed as 'redundant'.
    """

    def setUp(self):
        self.mod = load_cli()
        self.src = tempfile.mkdtemp(prefix="aw_src_")
        self.addCleanup(shutil.rmtree, self.src, True)
        self.dst = tempfile.mkdtemp(prefix="aw_snap_")
        self.addCleanup(shutil.rmtree, self.dst, True)
        self.home = tempfile.mkdtemp(prefix="aw_home_")   # empty => no leak
        self.addCleanup(shutil.rmtree, self.home, True)
        p = mock.patch.dict(os.environ, {"HOME": self.home})
        p.start()
        self.addCleanup(p.stop)
        self._build_source_corpus()

    def _build_source_corpus(self):
        """A live-SHAPED source KB: one real preference across 3 distinct
        sessions, high-frequency acks, one-off chatter, and an ERR->ok pair."""
        build_synthetic_kb(self.src)
        logs = os.path.join(self.src, "logs")
        os.makedirs(os.path.join(logs, "sessions"), exist_ok=True)
        pref = "always pin dependency versions, never use latest"
        recs = []
        for i, sid in enumerate(("s-aaa", "s-bbb", "s-ccc")):
            recs.append(_rec("2026-07-1%d" % i, sid, "/w/proj", "human",
                             "t%d" % i, pref))
            recs.append(_rec("2026-07-1%d" % i, sid, "/w/proj", "human",
                             "t%da" % i, "continue"))        # ack noise
            recs.append(_rec("2026-07-1%d" % i, sid, "/w/proj", "machine",
                             "t%db" % i, pref))              # machine => ignored
        recs.append(_rec("2026-07-13", "s-ddd", "/w/proj", "human", "t9",
                         "what does this one-off question do"))
        with open(os.path.join(logs, "prompts.log"), "w", encoding="utf-8") as f:
            f.write("".join(recs))
        with open(os.path.join(logs, "metrics.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"event": "terminal", "feature": "x"}) + "\n")
        for sid in ("s-aaa", "s-bbb", "s-ccc", "s-ddd"):
            sdir = os.path.join(logs, "sessions", sid)
            os.makedirs(sdir, exist_ok=True)
            calls = []
            if sid == "s-aaa":                      # an ERR -> ok pair to drain
                calls = [{"ts": "2026-07-10T00:00:01Z", "tool": "Bash",
                          "status": "ERR", "input": '{"command":"pytest -q"}',
                          "response": "boom"},
                         {"ts": "2026-07-10T00:00:09Z", "tool": "Bash",
                          "status": "ok", "input": '{"command":"pytest -q"}',
                          "response": "passed"}]
            with open(os.path.join(sdir, "live.jsonl"), "w",
                      encoding="utf-8") as f:
                for c in calls:
                    f.write(json.dumps(c) + "\n")
            with open(os.path.join(sdir, "main.jsonl"), "w",
                      encoding="utf-8") as f:      # settled
                f.write("x" * 4096)                # bulk the snapshot must NOT copy

    # --- helpers ----------------------------------------------------------
    def _drain(self, kdir, cap=2):
        """Run `prefer mine --backfill` until backfill_pending drains (bounded,
        exactly as the real measurement does). Returns (total, passes)."""
        total, passes = 0, 0
        while True:
            total += self.mod._transcript_mine(kdir, backfill=True,
                                               session_cap=cap)
            passes += 1
            with open(os.path.join(kdir, self.mod.MINER_STATE_REL),
                      encoding="utf-8") as f:
                st = json.load(f)
            if not st.get("backfill_pending") or passes > 50:
                return total, passes

    def _queue(self, kdir):
        q = os.path.join(kdir, self.mod.PREFER_QUEUE_REL)
        if not os.path.isfile(q):
            return []
        with open(q, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def _state(self, kdir):
        with open(os.path.join(kdir, self.mod.MINER_STATE_REL),
                  encoding="utf-8") as f:
            return json.load(f)

    def _fps(self, kdir):
        """The dedup fingerprints of the queued records. `fp` is DERIVED, never
        a stored field: the queue record carries the producer-shared schema, and
        _transcript_mine recomputes the fp at mine time to dedup against the
        live queue + archive tail. Deriving it the same way here keeps the test
        honest about the on-disk schema."""
        return sorted(self.mod._prefer_cand_fp(r.get("sid"), r.get("turn_id"),
                                               r.get("text"))
                      for r in self._queue(kdir))

    # --- the P1 trap ------------------------------------------------------
    def test_selfseeded_snapshot_mines_no_recurrence_the_p1_trap(self):
        # THE REGRESSION LOCK. Same corpus, NO offset-0 pre-seed: the miner
        # self-seeds offsets.prompts at EOF and --backfill does not rewind it.
        #
        # The trap is RECURRENCE-SPECIFIC, and that is exactly what makes it
        # dangerous. Deleting the state also self-seeds backfill_pending with
        # every on-disk sid, so the SESSION-sourced error-recovery detector
        # still drains and still emits — a harness asserting only "total > 0"
        # goes green on a corpus whose prompts.log was never read at all. The
        # recurrence count is the only assertion that catches it.
        snapshot_corpus_kb(self.src, self.dst)
        os.remove(os.path.join(self.dst, self.mod.MINER_STATE_REL))  # self-seed
        total, _ = self._drain(self.dst)
        sigs = [r.get("signal") for r in self._queue(self.dst)]
        self.assertEqual(sigs.count("recurrence"), 0,
                         "P1 has been fixed upstream (--backfill now rewinds the "
                         "prompts offset?) — re-verify the harness contract")
        self.assertEqual(total, sigs.count("error-recovery"),
                         "the only candidates a self-seeded snapshot can mine "
                         "are session-sourced")
        st = self._state(self.dst)
        size = os.path.getsize(os.path.join(self.dst, "logs", "prompts.log"))
        self.assertEqual(st["offsets"]["prompts"], size)   # parked at the tail

    def test_preseeded_snapshot_mines_a_nonzero_set(self):
        snapshot_corpus_kb(self.src, self.dst)
        total, _ = self._drain(self.dst)
        self.assertGreater(total, 0, "0 candidates == the P1 trap bit again; a "
                                     "snapshot harness must never pass vacuously")
        sigs = [r.get("signal") for r in self._queue(self.dst)]
        self.assertIn("recurrence", sigs)
        self.assertIn("error-recovery", sigs)   # pending pre-seed => sessions drain

    def test_the_repeated_preference_is_the_only_recurrence_candidate(self):
        snapshot_corpus_kb(self.src, self.dst)
        self._drain(self.dst)
        rec = [r for r in self._queue(self.dst) if r.get("signal") == "recurrence"]
        self.assertEqual(len(rec), 1)
        self.assertIn("pin dependency versions", rec[0]["text"])
        self.assertEqual(rec[0]["recurrence_count"], 3)      # distinct sessions
        self.assertEqual(rec[0]["source_sessions"], ["s-aaa", "s-bbb", "s-ccc"])

    # --- integrity --------------------------------------------------------
    def test_a_second_full_drain_mines_nothing_new_and_fps_are_stable(self):
        snapshot_corpus_kb(self.src, self.dst)
        self._drain(self.dst)
        fps = self._fps(self.dst)
        self.assertTrue(fps)
        seed_miner_state_at_zero(self.dst)     # re-arm: re-read the WHOLE corpus
        again, _ = self._drain(self.dst)
        self.assertEqual(again, 0, "re-mining the same corpus double-emitted")
        self.assertEqual(self._fps(self.dst), fps)

    def test_queue_records_carry_the_shared_producer_schema(self):
        snapshot_corpus_kb(self.src, self.dst)
        self._drain(self.dst)
        q = self._queue(self.dst)
        self.assertTrue(q)
        for r in q:
            # the schema the Stop-hook producer writes, field-for-field ...
            for k in ("ts", "sid", "cwd", "project", "turn_id", "text",
                      "matched_pattern", "status", "producer"):
                self.assertIn(k, r, "record missing shared-schema field %r" % k)
            self.assertIn("signal", r)             # ... + the miner's additive
            self.assertNotIn("fp", r, "fp is a DERIVED dedup key, not a stored "
                                      "field — persisting it would fork the "
                                      "producer-shared schema")
            self.assertEqual(r["status"], "queued")        # QUARANTINED
            self.assertEqual(r["producer"], "transcript-miner")

    def test_state_stays_bounded_and_offsets_are_monotonic(self):
        snapshot_corpus_kb(self.src, self.dst)
        before = self._state(self.dst)["offsets"]
        self._drain(self.dst)
        st = self._state(self.dst)
        self.assertLessEqual(len(st["tally"]), self.mod.MINER_TALLY_CAP)
        self.assertLessEqual(len(st["emitted_keys"]), self.mod.MINER_EMITTED_CAP)
        for k in ("prompts", "metrics"):
            self.assertGreaterEqual(st["offsets"][k], before[k])
            self.assertLessEqual(
                st["offsets"][k],
                os.path.getsize(os.path.join(self.dst, "logs",
                                             "prompts.log" if k == "prompts"
                                             else "metrics.jsonl")))
        self.assertEqual(st["backfill_pending"], [])       # fully drained

    # --- snapshot fidelity + write isolation ------------------------------
    def test_snapshot_never_writes_to_the_source_corpus(self):
        def snap(root):
            out = {}
            for dirpath, _dirs, files in os.walk(root):
                for fn in files:
                    p = os.path.join(dirpath, fn)
                    out[os.path.relpath(p, root)] = (os.path.getsize(p),
                                                     os.stat(p).st_mtime_ns)
            return out
        before = snap(self.src)
        snapshot_corpus_kb(self.src, self.dst)
        self._drain(self.dst)
        self.assertEqual(snap(self.src), before, "the measurement mutated live")

    def test_snapshot_copies_the_read_surface_and_stubs_main_jsonl(self):
        def rb(*parts):
            with open(os.path.join(*parts), "rb") as f:
                return f.read()
        info = snapshot_corpus_kb(self.src, self.dst)
        self.assertEqual(info["sessions"], 4)
        self.assertEqual(info["settled_sessions"], 4)
        for name in ("prompts.log", "metrics.jsonl"):
            self.assertEqual(rb(self.dst, "logs", name),
                             rb(self.src, "logs", name))
        d = os.path.join(self.dst, "logs", "sessions", "s-aaa")
        self.assertEqual(rb(d, "live.jsonl"),
                         rb(self.src, "logs", "sessions", "s-aaa", "live.jsonl"))
        # main.jsonl: settled MARKER only — presence is the whole read surface.
        self.assertEqual(os.path.getsize(os.path.join(d, "main.jsonl")), 0)
        self.assertTrue(self.mod._transcript_session_settled(d))

    def test_preseed_state_rewinds_offsets_and_arms_the_backfill(self):
        st = seed_miner_state_at_zero(self.dst, ["s-aaa", "s-bbb"])
        self.assertEqual(st["offsets"], {"prompts": 0, "metrics": 0})
        # pending seeded (NOT empty): --backfill selects `pending & on_disk`, so
        # an empty pending set silently scans ZERO sessions.
        self.assertEqual(st["backfill_pending"], ["s-aaa", "s-bbb"])
        self.assertEqual(st["processed_sessions"], ["s-aaa", "s-bbb"])
        self.assertEqual(st["tally"], {})
        self.assertEqual(st["emitted_keys"], [])
        loaded, fresh = self.mod._transcript_load_state(self.dst)
        self.assertFalse(fresh, "the pre-seed must be read as an EXISTING state")
        self.assertEqual(loaded["offsets"]["prompts"], 0)


if __name__ == "__main__":
    unittest.main()
