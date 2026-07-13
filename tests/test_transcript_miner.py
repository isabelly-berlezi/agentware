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

Hermetic: a temp KB (synthetic index + seeded logs), never the live KB (R-LOC-03).
"""

import json
import os
import shutil
import tempfile
import unittest

try:
    from tests._fixtures import load_cli, build_synthetic_kb
except ImportError:
    from _fixtures import load_cli, build_synthetic_kb


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


if __name__ == "__main__":
    unittest.main()
