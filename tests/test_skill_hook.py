"""Task 5 — skills two-tier: END-TO-END roster wiring for BOTH harnesses (R2-10).

(i) scripts/hooks/session-start.sh must inject the per-machine skills roster into
    additionalContext when the lockfile is populated, and be a byte-identical
    no-op when it is empty.
(ii) agentware.sh build_codex_prompt must do the same (codex parity), proven by
    EXECUTING the extracted function — not just `bash -n`.

Hermetic: a tmp kdir (.initialized + MAIN.md) via AGENTWARE_KNOWLEDGE_DIR, a tmp
lockfile via AGENTWARE_SKILLS_LOCK_PATH. Stdlib-only.
"""

import json
import os
import shutil
import subprocess
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOOK = os.path.join(REPO_ROOT, "scripts", "hooks", "session-start.sh")
AGENTWARE_SH = os.path.join(REPO_ROOT, "agentware.sh")


def _extract_bash_func(path, name):
    """Extract a top-level `name() { ... }` bash function body (closing brace at
    column 0). Good enough for the well-formatted agentware.sh functions."""
    with open(path, "r", encoding="utf-8") as _f:
        lines = _f.read().splitlines()
    start = None
    for i, ln in enumerate(lines):
        if ln.startswith(name + "() {") or ln.startswith(name + "(){"):
            start = i
            break
    if start is None:
        raise AssertionError("function %s not found in %s" % (name, path))
    for j in range(start + 1, len(lines)):
        if lines[j] == "}":
            return "\n".join(lines[start:j + 1])
    raise AssertionError("no closing brace for %s" % name)


class SkillHookTest(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="aw-hook-")
        self.kdir = os.path.join(self.root, "kb")
        os.makedirs(self.kdir)
        with open(os.path.join(self.kdir, ".initialized"), "w") as f:
            f.write("2026-07-19 test\n")
        with open(os.path.join(self.kdir, "MAIN.md"), "w") as f:
            f.write("# Test MAIN\nHello.\n")
        self.lock = os.path.join(self.root, "skills.lock.json")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _env(self, lock_path):
        env = dict(os.environ)
        env["AGENTWARE_KNOWLEDGE_DIR"] = self.kdir
        env["AGENTWARE_SKILLS_LOCK_PATH"] = lock_path
        env.pop("AGENTWARE_INVOKED_FROM", None)
        return env

    def _write_lock(self):
        data = {"schema": 1, "skills": {
            "alpha": {"status": "installed", "verdict": "PASS"},
            "beta": {"status": "quarantined", "verdict": "PASS"},
        }}
        with open(self.lock, "w") as f:
            json.dump(data, f)

    def _run_hook(self, lock_path):
        cp = subprocess.run(["bash", HOOK], input="{}", capture_output=True,
                            text=True, env=self._env(lock_path), cwd=REPO_ROOT)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        out = cp.stdout
        try:
            ctx = json.loads(out)["hookSpecificOutput"]["additionalContext"]
        except (ValueError, KeyError):
            ctx = out  # plain-text fallback path
        return ctx

    # --- session-start hook ---------------------------------------------
    def test_hook_populated_lockfile_injects_roster(self):
        self._write_lock()
        ctx = self._run_hook(self.lock)
        self.assertIn("AGENTWARE_SKILLS:", ctx)
        self.assertIn("alpha", ctx)
        self.assertIn("beta", ctx)
        self.assertIn("skill sync --approve", ctx)

    def test_hook_empty_lockfile_is_noop(self):
        # Absent lockfile -> no roster; and the rest of the context is identical
        # to the populated run minus the roster block.
        absent = os.path.join(self.root, "nope.json")
        ctx_empty = self._run_hook(absent)
        self.assertNotIn("AGENTWARE_SKILLS:", ctx_empty)
        self._write_lock()
        ctx_full = self._run_hook(self.lock)
        # Removing the roster block from the full context yields the empty one.
        idx = ctx_full.index("AGENTWARE_SKILLS:")
        # roster is appended (with a leading newline) to CTX; strip it back off.
        trimmed = ctx_full[:idx].rstrip("\n")
        self.assertEqual(trimmed, ctx_empty.rstrip("\n"),
                         "roster must be an ADDITIVE block; nothing else changes")

    # --- codex parity (build_codex_prompt) ------------------------------
    def _run_codex(self, lock_path):
        func = _extract_bash_func(AGENTWARE_SH, "build_codex_prompt")
        # AGENT points at a non-existent persona so strip_frontmatter is skipped.
        harness = (
            'set -euo pipefail\n'
            'AGENT="__no_such_agent__"\n'
            'INITIALIZED=true\n'
            'KDIR="%s"\n'
            'REPO_ROOT="%s"\n'
            '%s\n'
            'build_codex_prompt "PHASE_PROMPT_MARKER"\n'
        ) % (self.kdir, REPO_ROOT, func)
        cp = subprocess.run(["bash", "-c", harness], capture_output=True,
                            text=True, env=self._env(lock_path), cwd=REPO_ROOT)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        return cp.stdout

    def test_codex_populated_injects_roster(self):
        self._write_lock()
        out = self._run_codex(self.lock)
        self.assertIn("AGENTWARE_SKILLS:", out)
        self.assertIn("alpha", out)
        self.assertIn("skill sync --approve", out)
        self.assertIn("PHASE_PROMPT_MARKER", out)

    def test_codex_empty_is_noop(self):
        absent = os.path.join(self.root, "nope.json")
        out = self._run_codex(absent)
        self.assertNotIn("AGENTWARE_SKILLS:", out)
        self.assertIn("PHASE_PROMPT_MARKER", out)


if __name__ == "__main__":
    unittest.main()
