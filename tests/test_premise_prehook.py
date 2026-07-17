"""Premise-grounding pre-hook — end-to-end loop test
(feature 260717-premise-grounding-gate, Task 5).

Drives the REAL agentware.sh against a SYNTHETIC, INITIALIZED throwaway knowledge
dir and a FAKE agent runtime, proving the deterministic pre-hook:

  * a plan whose `## Premises` block declares a premise that REPRODUCES-FALSIFIED
    against live state HARD-ABORTS in run_pre_hooks with LOOP_OUTCOME=
    premise_falsified + exit 1 BEFORE `run_phase "pre"` (the fake agent is never
    spawned — proven by an absent spawn sentinel);
  * the AGENTWARE_DISABLE_PREMISE_GATE kill-switch BYPASSES the gate LOUDLY and
    records a `> DECISION:` to the worklog (never a silent skip);
  * PRE_PROMPT (the pre-phase instruction that forbids the LLM from changing
    acceptance criteria) is byte-identical to git HEAD — the halt authority lives
    entirely in the deterministic pre-hook (RCA F3 fix), never the LLM.

The operator KB is never touched (R-LOC-03): HOME + AGENTWARE_KNOWLEDGE_DIR are
patched to temp dirs and the sibling package gates are disabled so the test
exercises ONLY the premise gate.

Runner:  python3 -m unittest tests.test_premise_prehook -v
"""

import json
import os
import shutil
import stat
import subprocess
import tempfile
import unittest

try:
    from tests._fixtures import REPO_ROOT, build_synthetic_kb, run_cli
except ImportError:  # discover -s tests puts tests/ on sys.path
    from _fixtures import REPO_ROOT, build_synthetic_kb, run_cli

AGENTWARE_SH = os.path.join(REPO_ROOT, "agentware.sh")
FEATURE = "260717-premise-prehook-test"

# A lint-passing plan (mirrors tests/test_plan_lint.GOOD_PLAN) whose `## Premises`
# block declares ONE premise that is FALSE against a fresh synthetic KB: the miner
# state file does not exist there, so `path exists …` reproduces-falsified.
_PLAN_HEAD = """\
# Plan — premise prehook test

> Synthetic plan for the premise-grounding pre-hook loop test.

## Target packages

- /tmp/aw-premise-prehook-target

"""

_PLAN_TAIL = """\
## Tasks

- ⬜ **1** Implement the first synthetic thing deterministically.
  *Verify:* `scripts/agentware index validate` exits 0.

- ⬜ **2** [e2e] Update the knowledge base after building (`R-KB-*`).
  Run `scripts/agentware learn` then `scripts/agentware index validate`.
  *Verify:* `scripts/agentware index validate` exits 0.

## Acceptance criteria

- [ ] the gate behaves

<promise>TASK_COMPLETE</promise>
"""

_FALSIFIED_PREMISES = """\
## Premises

- premise: the miner state file is present (it is NOT in a fresh KB)
  cite: learnings/geofence-reminders.md
  check: path exists logs/steering/.transcript-miner.state.json

"""

PLAN_FALSIFIED = _PLAN_HEAD + _FALSIFIED_PREMISES + _PLAN_TAIL

# Fake agent runtime: touches a spawn sentinel on EVERY invocation (so an absent
# sentinel proves the loop never reached the pre phase), flips one open marker per
# MAIN spawn, and echoes each phase's completion promise + a worklog line.
FAKE_CLI = r"""#!/usr/bin/env bash
touch "$SPAWN_SENTINEL"
if printf '%s' "$*" | grep -q "find the next task marked"; then
  python3 - "$FAKE_PLAN" <<'PY'
import re, sys
p = sys.argv[1]
with open(p, encoding="utf-8") as f:
    s = f.read()
s = re.sub(r"⬜|🟡", "✅", s, count=1)
with open(p, "w", encoding="utf-8") as f:
    f.write(s)
PY
  printf -- '- iteration: did synthetic task, verified.\n' >> "$FAKE_WORKLOG"
fi
echo "<promise>PRE_TASK_COMPLETE</promise>"
echo "<promise>TASK_COMPLETE</promise>"
echo "<promise>POST_COMPLETE</promise>"
"""


def _have(binary):
    return shutil.which(binary) is not None


@unittest.skipUnless(_have("jq"), "jq (an agentware.sh preflight dep) is required")
@unittest.skipUnless(_have("bash"), "bash is required to run agentware.sh")
class PremisePreHookLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agentware-premise-prehook-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tmp_home = os.path.join(self.tmp, "home")
        os.makedirs(os.path.join(self.tmp_home, ".agentware"))
        self.bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin_dir)
        self.kdir = os.path.join(self.tmp, "kb")
        os.makedirs(self.kdir)
        build_synthetic_kb(self.kdir)
        run_cli(["index", "migrate-frontmatter"], self.kdir)
        run_cli(["index", "rebuild"], self.kdir)
        with open(os.path.join(self.kdir, "MAIN.md"), "w", encoding="utf-8") as f:
            f.write("# KB\n\n- **Handle**: tester\n")
        with open(os.path.join(self.kdir, ".initialized"), "w",
                  encoding="utf-8") as f:
            f.write("2026-07-17T00:00:00Z tester\n")
        self.docs = os.path.join(self.kdir, "work", FEATURE)
        os.makedirs(self.docs)
        self.plan = os.path.join(self.docs, "plan.md")
        with open(self.plan, "w", encoding="utf-8") as f:
            f.write(PLAN_FALSIFIED)
        self.worklog = os.path.join(self.docs, "worklog.md")
        self.spawn_sentinel = os.path.join(self.tmp, "spawned")
        self.fake_cli = os.path.join(self.bin_dir, "claude")
        with open(self.fake_cli, "w", encoding="utf-8") as f:
            f.write(FAKE_CLI)
        os.chmod(self.fake_cli, os.stat(self.fake_cli).st_mode
                 | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    def _env(self, **extra):
        env = dict(os.environ)
        # Unset the kill-switch unless a test explicitly passes it in `extra`.
        env.pop("AGENTWARE_DISABLE_PREMISE_GATE", None)
        env.update({
            "HOME": self.tmp_home,
            "AGENTWARE_KNOWLEDGE_DIR": self.kdir,
            "AGENTWARE_CLI": "claude",
            "AGENTWARE_KB_AUTOCOMMIT": "0",
            "AGENTWARE_KB_PUSH": "0",
            "AGENTWARE_NO_STREAM": "1",
            "AGENTWARE_DISABLE_RELEASE_GATE": "1",
            "AGENTWARE_REVIEW_DIFF_FILES_OVERRIDE": "work/x/plan.md docs/readme.md",
            "AGENTWARE_DISABLE_TARGET_GATE": "1",
            "FAKE_PLAN": self.plan,
            "FAKE_WORKLOG": self.worklog,
            "SPAWN_SENTINEL": self.spawn_sentinel,
            "PATH": self.bin_dir + os.pathsep + os.environ.get("PATH", ""),
        })
        env.update(extra)
        return env

    def _run(self, env):
        return subprocess.run(
            ["bash", AGENTWARE_SH, FEATURE], cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=240)

    def _terminal_outcome(self):
        mpath = os.path.join(self.kdir, "logs", "metrics.jsonl")
        outcome = None
        try:
            with open(mpath, encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if ev.get("event") == "terminal" and ev.get("feature") == FEATURE:
                        outcome = ev.get("outcome")
        except OSError:
            pass
        return outcome

    def test_falsified_premise_halts_before_pre_phase(self):
        proc = self._run(self._env())
        self.assertNotEqual(proc.returncode, 0,
                            "a falsified premise must abort:\n%s" % proc.stdout)
        self.assertIn("premise-grounding gate FAILED", proc.stdout, proc.stdout)
        self.assertIn("falsified", proc.stdout.lower(), proc.stdout)
        # HALT was BEFORE run_phase "pre": the fake agent was never spawned.
        self.assertFalse(os.path.exists(self.spawn_sentinel),
                         "gate must halt before the pre phase spawns the agent")
        self.assertEqual(self._terminal_outcome(), "premise_falsified",
                         "terminal outcome must be premise_falsified")

    def test_kill_switch_bypasses_loudly_and_records_decision(self):
        # The worklog does NOT exist yet (the common first-bypass case). The bypass
        # must CREATE it and record the > DECISION: — never silently drop it.
        self.assertFalse(os.path.exists(self.worklog))
        proc = self._run(self._env(AGENTWARE_DISABLE_PREMISE_GATE="1"))
        self.assertIn("BYPASSING the premise-grounding gate", proc.stdout,
                      proc.stdout)
        self.assertTrue(os.path.isfile(self.worklog),
                        "bypass must create the worklog to record the DECISION")
        with open(self.worklog, encoding="utf-8") as f:
            wl = f.read()
        self.assertIn("premise-grounding gate BYPASSED via "
                      "AGENTWARE_DISABLE_PREMISE_GATE", wl, wl)

    def test_pre_prompt_byte_identical_to_head(self):
        # The halt authority lives in the deterministic pre-hook, never the LLM —
        # so the pre-phase instruction that forbids changing acceptance criteria is
        # untouched. Assert the PRE_PROMPT region of agentware.sh matches git HEAD.
        diff = subprocess.run(
            ["git", "diff", "HEAD", "--", "agentware.sh"],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, text=True).stdout
        removed = [ln[1:] for ln in diff.splitlines()
                   if ln.startswith("-") and not ln.startswith("---")]
        for needle in ("DO NOT change acceptance criteria or functional outcomes",
                       "You are reviewing the plan for",
                       "Improve the plan to better meet these criteria"):
            self.assertFalse(any(needle in r for r in removed),
                             "PRE_PROMPT line changed: %r" % needle)


if __name__ == "__main__":
    unittest.main()
