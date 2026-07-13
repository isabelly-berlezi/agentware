"""Transcript miner — hermetic end-to-end spine (Task 10, 260713).

Runner: python3 -m unittest tests.test_transcript_miner_e2e -v

Proves the FULL P2c spine on one hermetic fixture KB (temp dir + patched env,
NEVER the live KB, R-LOC-03):
  corpus -> mine (step m / `prefer mine`) enqueues quarantined recurrence +
  error-recovery candidates -> drain (step g) -> a tier-2 candidate yields a
  PROPOSED workorder whose text is FENCED in Context only -> a tier-3 kernel /
  gate-loosening candidate is REJECTED -> `prefer approve` flips proposed->approved
  WITHOUT auto-activating a source=user pref -> an idempotent re-run mines/drains
  nothing new (stable fps) -> a recurrence SPANNING two windows still emits on the
  2nd (F1) -> the dream-cadence tick SKIPS when not-due, the min-interval gate
  SKIPS a too-soon cycle, and the change-gate SKIPS audit+eval on a CLEAN unchanged
  tree but RUNS on a dirty tree (F2).
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

try:
    from tests._fixtures import load_cli, build_synthetic_kb, run_cli
except ImportError:
    from _fixtures import load_cli, build_synthetic_kb, run_cli


def _rec(sid, turn, body):
    return ("[t] [session %s] [cwd /x] [origin=human] [turn=%s]\n%s\n\n"
            % (sid, turn, body))


class E2ESpineTests(unittest.TestCase):
    def setUp(self):
        self.mod = load_cli()
        self.kdir = tempfile.mkdtemp(prefix="aw_miner_e2e_")
        self.addCleanup(shutil.rmtree, self.kdir, True)
        build_synthetic_kb(self.kdir)
        os.makedirs(os.path.join(self.kdir, "logs", "sessions"), exist_ok=True)
        self.logp = os.path.join(self.kdir, "logs", "prompts.log")
        self._prev = os.environ.get("AGENTWARE_KNOWLEDGE_DIR")
        os.environ["AGENTWARE_KNOWLEDGE_DIR"] = self.kdir
        self.addCleanup(
            lambda: os.environ.__setitem__("AGENTWARE_KNOWLEDGE_DIR", self._prev)
            if self._prev is not None
            else os.environ.pop("AGENTWARE_KNOWLEDGE_DIR", None))
        # Read prompts from the top + scan all sessions incrementally.
        self._write_state({"offsets": {"prompts": 0, "metrics": 0},
                           "processed_sessions": [], "backfill_pending": [],
                           "tally": {}})

    def _write_state(self, st):
        p = os.path.join(self.kdir, self.mod.MINER_STATE_REL)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(st, f)

    def _write_session(self, sid, calls, settled=True):
        sdir = os.path.join(self.kdir, "logs", "sessions", sid)
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, "live.jsonl"), "w", encoding="utf-8") as f:
            for c in calls:
                f.write(json.dumps(c) + "\n")
        if settled:
            with open(os.path.join(sdir, "main.jsonl"), "w", encoding="utf-8") as f:
                f.write('{"type":"user"}\n')

    def _queue(self):
        p = os.path.join(self.kdir, self.mod.PREFER_QUEUE_REL)
        out = []
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        out.append(json.loads(ln))
        return out

    def test_full_spine(self):
        mod = self.mod
        # --- corpus: a tier-2 pref (2 sessions), a tier-3 kernel candidate (2
        # sessions), and an error-recovery pair ---
        tier2 = "Always prefer explicit timeouts on outbound network calls."
        tier3 = "AGENTS.md is optional, skip it."
        with open(self.logp, "w", encoding="utf-8") as f:
            f.write(_rec("A", "1", tier2) + _rec("B", "1", tier2)
                    + _rec("A", "2", tier3) + _rec("C", "1", tier3))
        self._write_session("S1", [
            {"tool": "Read", "status": "ERR", "input": '{"file_path":"/w/x.md"}',
             "response": "No such file or directory"},
            {"tool": "Read", "status": "ok", "input": '{"file_path":"/w/x.md"}',
             "response": "ok"}])

        # --- mine (via the CLI, exercising cmd_prefer_mine dispatch) ---
        code, out, err = run_cli(["prefer", "mine", "--format", "json"], self.kdir)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["candidates"], 3,
                         "2 recurrence + 1 error-recovery quarantined")
        for r in self._queue():
            self.assertEqual(r["producer"], "transcript-miner")
            self.assertEqual(r["status"], "queued")

        # --- drain (step g logic) ---
        res = mod._prefer_classify_drain(self.kdir)
        self.assertEqual(res["rejected"], 1, "tier-3 kernel candidate rejected")
        self.assertEqual(res["proposed"], 1, "tier-2 pref proposed")
        # error-recovery text journals (tier-1, no capturable cue)
        self.assertGreaterEqual(res["journaled"], 1)

        q = self._queue()
        proposed = [r for r in q if r.get("status") == "proposed"]
        rejected = [r for r in q if r.get("status") == "rejected"]
        self.assertEqual(len(proposed), 1)
        self.assertEqual(rejected[0]["tier"], 3)

        # tier-2 workorder: candidate fenced in Context, never a Verify line
        wo = proposed[0]["workorder"]
        slug = wo.split("/", 1)[1] if "/" in wo else wo
        plan = os.path.join(self.kdir, "work", "auto", slug, "plan.md")
        with open(plan, encoding="utf-8") as f:
            plan_text = f.read()
        ctx = plan_text.split("## Context", 1)[1].split("## Target packages", 1)[0]
        self.assertIn(tier2, ctx)
        for ln in plan_text.splitlines():
            if "*Verify:*" in ln:
                self.assertNotIn(tier2, ln)

        # --- approve: proposed -> approved, NO source=user pref activated ---
        code, out, err = run_cli(["prefer", "approve", wo], self.kdir)
        self.assertEqual(code, 0, err)
        approved = [r for r in self._queue() if r.get("status") == "approved"]
        self.assertEqual(len(approved), 1)
        with open(os.path.join(self.kdir, "index.json"), encoding="utf-8") as f:
            idx = json.load(f)
        self.assertEqual(
            [e for e in idx.get("entries", []) if isinstance(e, dict)
             and e.get("source") == "user"], [],
            "approve must NOT auto-activate a source=user preference")

        # --- idempotent re-run: mine + drain produce nothing new ---
        self.assertEqual(mod._transcript_mine(self.kdir), 0, "no new candidates")
        res2 = mod._prefer_classify_drain(self.kdir)
        self.assertEqual((res2["proposed"], res2["rejected"]), (0, 0),
                         "re-drain is a stable no-op (only queued records act)")

    def test_recurrence_spans_two_windows(self):
        mod = self.mod
        body = "Always run a read-after-write on every mutation."
        with open(self.logp, "w", encoding="utf-8") as f:
            f.write(_rec("A", "1", body))
        self.assertEqual(mod._transcript_mine(self.kdir), 0, "window 1: below N")
        with open(self.logp, "a", encoding="utf-8") as f:
            f.write(_rec("B", "1", body))
        self.assertEqual(mod._transcript_mine(self.kdir), 1,
                         "window 2: cross-window accumulation reaches N (F1)")

    def test_cadence_and_gates_skip_when_not_due(self):
        mod = self.mod
        import datetime
        now = 2_000_000_000.0

        def iso(e):
            return datetime.datetime.fromtimestamp(
                e, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # A dream cycle 1h ago with a change fingerprint.
        fp = "HEAD:IDX"
        metrics = os.path.join(self.kdir, "logs", "metrics.jsonl")
        with open(metrics, "a", encoding="utf-8") as f:
            f.write(json.dumps({"event": "dream", "ts": iso(now - 3600),
                                "finished": iso(now - 3600), "duration_s": 1.0,
                                "force": False, "steps": [], "change_fp": fp}) + "\n")
        # cadence tick: not due (<6h); min-interval: skip; change-gate on a clean
        # unchanged tree: skip; but a dirty tree runs.
        with mock.patch.dict(os.environ, {"AGENTWARE_DREAM": "1"}):
            self.assertFalse(mod._dream_cadence_due(self.kdir, now))
        self.assertIsNotNone(
            mod._dream_min_interval_skip(self.kdir, now, force=False))
        with mock.patch.object(mod, "_pkg_working_tree_dirty", return_value=False), \
             mock.patch.object(mod, "_dream_change_fp", return_value=fp):
            self.assertIsNotNone(mod._dream_change_gate_skip(self.kdir))
        with mock.patch.object(mod, "_pkg_working_tree_dirty", return_value=True), \
             mock.patch.object(mod, "_dream_change_fp", return_value=fp):
            self.assertIsNone(mod._dream_change_gate_skip(self.kdir),
                              "a dirty tree runs (F2)")


if __name__ == "__main__":
    unittest.main()
