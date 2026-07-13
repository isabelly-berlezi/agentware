"""Self-healing adversarial-review gate — end-to-end loop test
(feature 260713-adversarial-review-gate, Task 5).

Drives the REAL agentware.sh against a SYNTHETIC, INITIALIZED throwaway knowledge
dir and a FAKE agent runtime, proving the Task-5 wiring in `run_selfheal_review`
(AUDIT -> CALL-OUT -> FIX -> RE-AUDIT, bounded, then PROCEED-or-ESCALATE):

  * a self-extension diff with a planted CONFIRMED defect is CAUGHT, auto-fixed,
    RE-AUDITED clean, and only THEN does the run PROCEED to the done-stamp + kb-sync
    ("fully complete", exit 0);
  * an UNFIXABLE planted defect (the fixer names no regression test, so the fix is
    REFUSED) survives every bounded round -> the run is BLOCKED (exit 1, "BLOCKED"),
    and NEVER reaches the done-stamp / kb-sync ("fully complete" absent);
  * a CLEAN self-extension diff proceeds with no remediation;
  * a NON-self-extension diff skips the gate entirely (byte-identical to today).

The gate spawns its Opus fan-out through `scripts/agentware review`/`remediate`,
which resolve the agent binary via AGENTWARE_CLI — so a single FAKE `claude` binary
drives every reviewer / verifier / critic / fixer with NO real Opus. The operator KB
is never touched (R-LOC-03): HOME + AGENTWARE_KNOWLEDGE_DIR are patched to temp dirs.

Runner:  python3 -m unittest tests.test_loop_selfheal_gate -v
"""

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
FEATURE = "260713-selfheal-gate-test"

# Lint-passing synthetic plan: monotonic `- ⬜ **N**` markers, a *Verify:* per task,
# an R7 KB token, one [e2e] task, exactly one <promise>, no banned phrases.
PLAN_MD = """\
# Plan — self-heal gate test

> Synthetic plan for the self-healing adversarial-review gate loop test.

## Target packages

- /tmp/aw-selfheal-gate-target

## Tasks

- ⬜ **1** Implement the first synthetic thing deterministically.
  *Verify:* `scripts/agentware index validate` exits 0.

- ⬜ **2** Update the knowledge base after building (`R-KB-*`).
  Run `scripts/agentware learn` then `scripts/agentware index validate`.
  *Verify:* `scripts/agentware index validate` exits 0.

- ⬜ **3** [e2e] Run the end-to-end regression against the real loop.
  *Verify:* `./agentware.sh selfheal-gate` proceeds/blocks as expected.

## Acceptance criteria

- [ ] the self-heal gate fires on a self-extension diff

<promise>TASK_COMPLETE</promise>
"""

# One FAKE `claude` for every spawn. Dispatch is on the prompt HEAD (the text BEFORE
# the first `<<<` data fence) so the fenced diff/finding DATA — which, for THIS very
# feature, literally contains the reviewer/critic/fixer prompt strings — can never
# leak a dispatch keyword. Review-gate spawns return ONLY JSON; phase spawns (PRE /
# MAIN / POST) flip one plan marker (MAIN only) + write a worklog line and echo the
# phase promises, mirroring tests/test_loop_worklog_gate.
FAKE_CLI = r"""#!/usr/bin/env bash
prompt="${!#}"
head="${prompt%%<<<*}"

# --- Review-gate persona spawns (JSON only; no phase promises) ----------------
if printf '%s' "$head" | grep -q "REMEDIATION fixer"; then
  if [[ "${FAKE_FIXABLE:-1}" == "1" ]]; then
    # A verifiable fix: names a real, fast, passing regression test + signals the
    # RE-AUDIT that the defect is closed by dropping the fix marker.
    touch "$FAKE_FIXMARK"
    echo '{"fixes":[{"index":0,"status":"fixed","regression_test":"tests/test_frontmatter.py","notes":"planted defect fixed"}]}'
  else
    # An UNVERIFIED fix: claims fixed but names NO regression test -> REFUSED
    # (downgraded to unfixable). No marker -> the defect survives every re-audit.
    echo '{"fixes":[{"index":0,"status":"fixed","notes":"claims fixed, names no regression test"}]}'
  fi
  exit 0
fi
if printf '%s' "$head" | grep -q "COMPLETENESS CRITIC"; then
  echo '{"gaps":[],"additional_findings":[]}'
  exit 0
fi
if printf '%s' "$head" | grep -q "REFUTE"; then
  echo '{"verdict":"confirmed","verdict_rationale":"reproduced against the diff"}'
  exit 0
fi
if printf '%s' "$head" | grep -q "dimension reviewer"; then
  if [[ "${FAKE_REVIEW_DIES:-0}" == "1" ]]; then
    exit 1   # simulate the agent runtime unreachable: no output, nonzero exit
  fi
  if [[ "${FAKE_CLEAN:-0}" == "1" ]] || [[ -f "${FAKE_FIXMARK:-/nonexistent}" ]]; then
    echo '{"findings":[]}'
  elif printf '%s' "$head" | grep -q "'correctness' dimension reviewer"; then
    echo '{"findings":[{"dimension":"correctness","severity":"high","file":"agentware.sh","line":1,"summary":"planted correctness defect","failure_scenario":"hostile input crashes the gate"}]}'
  else
    echo '{"findings":[]}'
  fi
  exit 0
fi

# --- Phase spawns (PRE / MAIN / POST) -----------------------------------------
if printf '%s' "$prompt" | grep -q "find the next task marked"; then
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
@unittest.skipUnless(_have("git"), "git is required for the self-extension predicate")
class SelfHealGateLoopTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="agentware-shgate-")
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.tmp_home = os.path.join(self.tmp, "home")
        os.makedirs(self.tmp_home)
        self.bin_dir = os.path.join(self.tmp, "bin")
        os.makedirs(self.bin_dir)
        self.kdir = os.path.join(self.tmp, "kb")
        os.makedirs(self.kdir)
        # Build + initialize a valid KB so the pre/post toolkit gates run for real.
        build_synthetic_kb(self.kdir)
        run_cli(["index", "migrate-frontmatter"], self.kdir)
        run_cli(["index", "rebuild"], self.kdir)
        with open(os.path.join(self.kdir, "MAIN.md"), "w", encoding="utf-8") as f:
            f.write("# KB\n\n- **Handle**: tester\n")
        with open(os.path.join(self.kdir, ".initialized"), "w", encoding="utf-8") as f:
            f.write("2026-07-13T00:00:00Z tester\n")
        self.docs = os.path.join(self.kdir, "work", FEATURE)
        os.makedirs(self.docs)
        self.plan = os.path.join(self.docs, "plan.md")
        with open(self.plan, "w", encoding="utf-8") as f:
            f.write(PLAN_MD)
        self.worklog = os.path.join(self.docs, "worklog.md")
        self.fake_cli = os.path.join(self.bin_dir, "claude")
        with open(self.fake_cli, "w", encoding="utf-8") as f:
            f.write(FAKE_CLI)
        os.chmod(self.fake_cli, os.stat(self.fake_cli).st_mode
                 | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
        self.fixmark = os.path.join(self.tmp, "fixmark")

    def _env(self, **extra):
        env = dict(os.environ)
        env.update({
            "HOME": self.tmp_home,
            "AGENTWARE_KNOWLEDGE_DIR": self.kdir,
            "AGENTWARE_CLI": "claude",
            "AGENTWARE_KB_AUTOCOMMIT": "0",
            "AGENTWARE_KB_PUSH": "0",
            "AGENTWARE_NO_STREAM": "1",
            "AGENTWARE_METRICS_EMIT": "0",
            "AGENTWARE_DISABLE_RELEASE_GATE": "1",   # our own edits show dirty
            "AGENTWARE_DISABLE_TARGET_GATE": "1",    # target path need not exist
            # Keep the audited diff small + deterministic: the fake ignores the diff
            # body (it dispatches on the prompt head), and the self-extension
            # PREDICATE is independent of the range (it reads git status/name-only).
            "AGENTWARE_REVIEW_DIFF_RANGE": "HEAD..HEAD",
            "FAKE_PLAN": self.plan,
            "FAKE_WORKLOG": self.worklog,
            "FAKE_FIXMARK": self.fixmark,
            "PATH": self.bin_dir + os.pathsep + os.environ.get("PATH", ""),
        })
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def _run(self, env):
        return subprocess.run(
            ["bash", AGENTWARE_SH, FEATURE], cwd=REPO_ROOT, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            timeout=300)

    def test_planted_defect_is_fixed_reaudited_clean_then_proceeds(self):
        # Round 1 finds a CONFIRMED high; remediation applies a verified fix +
        # drops the marker; round 2 re-audits CLEAN -> the run PROCEEDS.
        proc = self._run(self._env(FAKE_FIXABLE="1", AGENTWARE_REVIEW_MAX_ROUNDS="3"))
        out = proc.stdout
        self.assertEqual(proc.returncode, 0,
                         "a self-healed run must complete:\n%s" % out)
        self.assertIn("running the adversarial-review gate", out, out)
        self.assertIn("round 1", out, out)
        self.assertIn("auto-remediating", out, out)
        self.assertIn("CLEAN", out, out)          # a later round audits clean
        self.assertIn("fully complete", out, out)  # reached done-stamp + kb-sync
        self.assertTrue(os.path.isfile(self.fixmark), "the fixer must have run")

    def test_unfixable_defect_blocks_done_and_escalates(self):
        # The fixer never provides a regression test -> every fix is REFUSED -> the
        # confirmed high survives every bounded round -> BLOCKED, no done, no commit.
        proc = self._run(self._env(FAKE_FIXABLE="0", AGENTWARE_REVIEW_MAX_ROUNDS="2"))
        out = proc.stdout
        self.assertNotEqual(proc.returncode, 0,
                            "an unfixable defect must block the run:\n%s" % out)
        self.assertIn("BLOCKED", out, out)
        self.assertNotIn("fully complete", out,
                         "a blocked run must NOT reach done-stamp + kb-sync:\n%s" % out)
        self.assertFalse(os.path.isfile(self.fixmark),
                         "the unfixable fixer must not signal a closed defect")

    def test_clean_self_extension_diff_proceeds_without_remediation(self):
        proc = self._run(self._env(FAKE_CLEAN="1"))
        out = proc.stdout
        self.assertEqual(proc.returncode, 0,
                         "a clean self-extension audit must proceed:\n%s" % out)
        self.assertIn("running the adversarial-review gate", out, out)
        self.assertIn("CLEAN", out, out)
        self.assertNotIn("auto-remediating", out,
                         "a clean round must not remediate:\n%s" % out)
        self.assertIn("fully complete", out, out)

    def test_non_self_extension_diff_skips_the_gate(self):
        # Force the changed-file list to a NON-package path: the predicate no-matches
        # and the gate is skipped entirely (no reviewer ever spawns).
        proc = self._run(self._env(
            AGENTWARE_REVIEW_DIFF_FILES_OVERRIDE="work/x/plan.md docs/readme.md"))
        out = proc.stdout
        self.assertEqual(proc.returncode, 0,
                         "a non-self-extension run must complete:\n%s" % out)
        self.assertIn("non-self-extension diff", out, out)
        self.assertNotIn("running the adversarial-review gate", out, out)
        self.assertIn("fully complete", out, out)

    # === Post-audit fail-CLOSED hardening (2026-07-13) =====================
    def test_agent_runtime_unreachable_fails_closed_blocks(self):
        # THE fail-open the audit found: when every reviewer subprocess fails (agent
        # runtime unreachable / nonzero exit), the gate must BLOCK (fail-closed),
        # NEVER silently stamp done + commit an un-audited self-extension diff.
        proc = self._run(self._env(
            FAKE_REVIEW_DIES="1", AGENTWARE_REVIEW_MAX_ROUNDS="2",
            AGENTWARE_REVIEW_DIFF_FILES_OVERRIDE="scripts/agentware"))
        out = proc.stdout
        self.assertNotEqual(proc.returncode, 0,
                            "an un-runnable audit must BLOCK, not proceed:\n%s" % out)
        self.assertIn("COULD NOT RUN", out, out)
        self.assertNotIn("fully complete", out,
                         "a fail-closed block must NOT reach done-stamp/kb-sync:\n%s"
                         % out)

    def test_skip_false_does_NOT_bypass_gate(self):
        # Fix B: AGENTWARE_SKIP_ADVERSARIAL_REVIEW=false means "keep the gate ON".
        # A bare `-n` guard would BYPASS on any non-empty value (inverting intent);
        # the truthy-matched guard must RUN the gate.
        proc = self._run(self._env(
            FAKE_CLEAN="1", AGENTWARE_SKIP_ADVERSARIAL_REVIEW="false",
            AGENTWARE_REVIEW_DIFF_FILES_OVERRIDE="scripts/agentware"))
        out = proc.stdout
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn("running the adversarial-review gate", out,
                      "=false must NOT bypass — the gate must actually run:\n%s" % out)
        self.assertNotIn("BYPASSING", out, out)
        self.assertIn("fully complete", out, out)

    def test_skip_internal_whitespace_does_NOT_bypass(self):
        # Re-audit residual: bash must trim ONLY leading/trailing whitespace (like
        # Python str.strip()), NOT collapse internal whitespace. "o n" must stay
        # non-truthy in BOTH bash and Python so it can never bypass without a DECISION.
        proc = self._run(self._env(
            FAKE_CLEAN="1", AGENTWARE_SKIP_ADVERSARIAL_REVIEW="o n",
            AGENTWARE_REVIEW_DIFF_FILES_OVERRIDE="scripts/agentware"))
        out = proc.stdout
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn("running the adversarial-review gate", out,
                      "internal-space 'o n' must NOT bypass — the gate must run:\n%s"
                      % out)
        self.assertNotIn("BYPASSING", out, out)

    def test_skip_truthy_bypasses_with_recorded_decision(self):
        # A TRUTHY value bypasses — but ONLY with a recorded `> DECISION:` (never
        # silent), and the Python + bash truthy checks must AGREE so the marker lands.
        proc = self._run(self._env(
            FAKE_CLEAN="1", AGENTWARE_SKIP_ADVERSARIAL_REVIEW="1",
            AGENTWARE_REVIEW_DIFF_FILES_OVERRIDE="scripts/agentware"))
        out = proc.stdout
        self.assertEqual(proc.returncode, 0, out)
        self.assertIn("BYPASSING", out, out)
        self.assertNotIn("running the adversarial-review gate", out,
                         "a truthy skip must short-circuit before the round loop:\n%s"
                         % out)
        with open(self.worklog, encoding="utf-8") as f:
            wl = f.read()
        self.assertIn("> DECISION:", wl,
                      "a bypass must RECORD a > DECISION: marker (never silent)")


if __name__ == "__main__":
    unittest.main()
