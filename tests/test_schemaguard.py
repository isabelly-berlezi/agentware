"""Central KB-writer registry / schema-guard coverage tests
(feature 260711-emitter-gate-hardening, T3).

Before this feature the older-CLI schema guard covered only index/learn/decide/
ingest — the plan-state writers (cmd_plan_new, cmd_plan_set_status,
cmd_plan_claim — which runs EVERY loop iteration — cmd_plan_set_state,
cmd_plan_state_backfill), cmd_features and the skill writers were UNGUARDED, so
an older CLI could silently mutate a newer (migrated) KB through those verbs.

A single registry (`_kb_writer_funcs`, the version-refusal SUPERSET) now gates
every KB-mutating verb; a separate `_kb_autoapply_funcs` subset (superset minus
the migration-runner verb `cmd_plan_state_backfill`) drives the on-stale-KB
auto-apply so migration 001's backfill never double-runs. `cmd_migrate` and
`cmd_attach` are deliberately NOT registered.

Tests (behavioral, not grep-shaped):
  - COMPLETENESS: walk the argparse set_defaults(func=…) tree and assert the
    registry equals a ground-truth writer set (deliberate-omission canary);
  - per-writer newer-.schema refusal via the real dispatch gate;
  - recall/query stay EXEMPT (not refused);
  - `migrate --list` on an older CLI prints the mismatch honestly and leaves
    `.schema` UNTOUCHED (read-only).

`test_schemaguard_` is unique vs every existing module filename and method.
Runner: python3 -m unittest tests.test_schemaguard -v
"""

import argparse
import contextlib
import importlib.machinery
import importlib.util
import io
import os
import shutil
import tempfile
import unittest

try:
    from tests._fixtures import REPO_ROOT, build_synthetic_kb, run_cli
except ImportError:
    from _fixtures import REPO_ROOT, build_synthetic_kb, run_cli


def _load():
    ld = importlib.machinery.SourceFileLoader(
        "aw_schemaguard", os.path.join(REPO_ROOT, "scripts", "agentware"))
    m = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("aw_schemaguard", ld))
    ld.exec_module(m)
    return m


# The GROUND TRUTH: every KB-mutating command verb. If a new writer verb is added
# without registering it, the completeness test below fails LOUDLY. cmd_migrate /
# cmd_attach are intentionally absent (see registry docstring).
_GROUND_TRUTH_WRITERS = {
    "cmd_index_add", "cmd_index_remove", "cmd_index_sync", "cmd_index_rebuild",
    "cmd_index_migrate",
    "cmd_learn", "cmd_decide", "cmd_ingest",
    "cmd_plan_new", "cmd_plan_add_task", "cmd_plan_set_state",
    "cmd_plan_set_status", "cmd_plan_claim", "cmd_plan_state_backfill",
    "cmd_features", "cmd_skill_add", "cmd_skill_remove",
    # cmd_dream: step-a rebuilds the structured index in-process (registered
    # after the 260711 adversarial review found it unguarded + cron-installable).
    "cmd_dream",
}

# Top-level defs that call save_index()/rebuild_kb() (the structured-index
# rewriters). Discovered by scanning the CLI source in
# test_schemaguard_no_unregistered_index_writer, this PINNED set means a NEW
# index-writing function anywhere in the CLI trips the test — the genuinely
# non-vacuous completeness check that WOULD have caught cmd_dream. Each maps to the
# dispatch COMMAND that reaches it; that command must be registered OR a documented
# exemption (git-sync / fanout-merge / bootstrap / attach — see the DECISIONs in
# the feature worklog: they rebuild only {entries,tags} from migration-invariant
# frontmatter as part of a git/merge/init flow).
_EXPECTED_INDEX_WRITER_DEFS = {
    # def name -> owning dispatch command (None = primitive/helper of below)
    "cmd_init": "cmd_init",                       # bootstrap (pre-initialization)
    "cmd_index_add": "cmd_index_add",
    "cmd_index_remove": "cmd_index_remove",
    "cmd_index_sync": "cmd_index_sync",
    "rebuild_kb": None,                           # the primitive itself
    "cmd_index_rebuild": "cmd_index_rebuild",
    "_attach_migrate_structure": "cmd_attach",   # arbitrary candidate path
    "cmd_learn": "cmd_learn",
    "cmd_decide": "cmd_decide",
    "cmd_ingest": "cmd_ingest",
    "kb_git_push_once": "cmd_kb_git_commit",     # git-sync rebuild-before-commit
    "kb_git_merge_continue": "cmd_kb_git_merge_continue",
    "_mq_integrate_kb": "cmd_fanout_merge_queue",
    "_dream_step_kb_pull": "cmd_dream",
    "_dream_step_index_rebuild": "cmd_dream",
    "cmd_skill_add": "cmd_skill_add",
    "cmd_skill_remove": "cmd_skill_remove",
}
# Commands that legitimately rewrite the index but are EXEMPT from the schema gate
# for the CURRENT schema (documented DECISION in the worklog): git-sync / merge
# rebuilds regenerate only {entries,tags} from migration-001-invariant frontmatter
# during a commit/merge; init is the pre-initialization bootstrap; attach targets
# an arbitrary candidate path with its own conformance flow.
_INDEX_WRITER_EXEMPT_COMMANDS = {
    "cmd_init", "cmd_attach", "cmd_kb_git_commit", "cmd_kb_git_merge_continue",
    "cmd_fanout_merge_queue",
}


def _collect_dispatch_funcs(parser, acc):
    """Walk the argparse tree collecting every set_defaults(func=…)."""
    d = parser._defaults.get("func")
    if d is not None:
        acc.add(d)
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for sub in action.choices.values():
                _collect_dispatch_funcs(sub, acc)
    return acc


class SchemaGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = _load()

    def setUp(self):
        self.kdir = tempfile.mkdtemp(prefix="agentware-schemaguard-")
        self.addCleanup(shutil.rmtree, self.kdir, True)
        build_synthetic_kb(self.kdir)
        # Mark initialized so the predispatch gate engages (it no-ops on an
        # uninitialized KB — the onboarding path).
        with open(os.path.join(self.kdir, ".initialized"), "w",
                  encoding="utf-8") as f:
            f.write("test\n")

    def _write_schema(self, n):
        with open(os.path.join(self.kdir, ".schema"), "w", encoding="utf-8") as f:
            f.write("%d\n" % n)

    def _read_schema_raw(self):
        p = os.path.join(self.kdir, ".schema")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return f.read()

    def _gate(self, func):
        """Run the real dispatch gate for `func` against this KB."""
        prev = os.environ.get("AGENTWARE_KNOWLEDGE_DIR")
        os.environ["AGENTWARE_KNOWLEDGE_DIR"] = self.kdir
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return self.m._kb_predispatch_gate(argparse.Namespace(func=func))
        finally:
            if prev is None:
                os.environ.pop("AGENTWARE_KNOWLEDGE_DIR", None)
            else:
                os.environ["AGENTWARE_KNOWLEDGE_DIR"] = prev

    # --- completeness -------------------------------------------------------
    def test_schemaguard_registry_equals_ground_truth(self):
        registry = {f.__name__ for f in self.m._kb_writer_funcs()}
        self.assertEqual(
            registry, _GROUND_TRUTH_WRITERS,
            "registry drifted from the ground-truth writer set: "
            "missing=%s extra=%s" % (
                sorted(_GROUND_TRUTH_WRITERS - registry),
                sorted(registry - _GROUND_TRUTH_WRITERS)))

    def test_schemaguard_completeness_canary(self):
        # Deliberate-omission canary: proves the completeness assertion is not
        # vacuous — dropping the loop's every-iteration writer MUST be detected.
        registry = {f.__name__ for f in self.m._kb_writer_funcs()}
        self.assertIn("cmd_plan_claim", registry,
                      "cmd_plan_claim runs every loop iteration — the original hole")
        omitted = registry - {"cmd_plan_claim"}
        self.assertNotEqual(omitted, _GROUND_TRUTH_WRITERS,
                            "canary: an omitted writer must break the equality")

    def test_schemaguard_every_registry_func_is_dispatchable(self):
        parser_funcs = {f.__name__
                        for f in _collect_dispatch_funcs(self.m.build_parser(), set())}
        registry = {f.__name__ for f in self.m._kb_writer_funcs()}
        self.assertTrue(registry <= parser_funcs,
                        "registry has non-dispatchable funcs: %s"
                        % sorted(registry - parser_funcs))

    def test_schemaguard_migrate_and_attach_excluded(self):
        registry = {f.__name__ for f in self.m._kb_writer_funcs()}
        self.assertNotIn("cmd_migrate", registry)
        self.assertNotIn("cmd_attach", registry)

    def test_schemaguard_autoapply_excludes_dualmode_and_runner(self):
        # Auto-apply excludes the migration-runner verb (double-run) AND the
        # dual-mode dream (its --dry-run must not write .schema).
        superset = self.m._kb_writer_funcs()
        autoapply = self.m._kb_autoapply_funcs()
        self.assertEqual({f.__name__ for f in (superset - autoapply)},
                         {"cmd_plan_state_backfill", "cmd_dream"})

    def test_schemaguard_dream_registered_but_not_autoapply(self):
        reg = {f.__name__ for f in self.m._kb_writer_funcs()}
        auto = {f.__name__ for f in self.m._kb_autoapply_funcs()}
        self.assertIn("cmd_dream", reg, "cmd_dream must be version-refusal guarded")
        self.assertNotIn("cmd_dream", auto,
                         "cmd_dream --dry-run must not trigger a .schema write")

    def test_schemaguard_dream_refuses_on_newer_kb(self):
        self._write_schema(self.m.KB_SCHEMA_VERSION + 1)
        self.assertEqual(self._gate(self.m.cmd_dream), 1,
                         "dream must REFUSE on a newer KB (rebuild_kb would corrupt it)")

    def test_schemaguard_no_unregistered_index_writer(self):
        # THE non-vacuous completeness check: scan the CLI source for every
        # save_index()/rebuild_kb() call site, map to the enclosing top-level def,
        # and assert (1) the writer-def set matches the pinned expectation (a NEW
        # index-writing function trips this — this is what would have caught
        # cmd_dream), and (2) every writer-def's owning command is registered OR a
        # documented exemption.
        import re
        with open(os.path.join(REPO_ROOT, "scripts", "agentware"),
                  encoding="utf-8") as f:
            lines = f.read().split("\n")
        defs = [(i + 1, re.match(r"^def (\w+)\(", l).group(1))
                for i, l in enumerate(lines) if re.match(r"^def \w+\(", l)]

        def enclosing(ln):
            name = "<module>"
            for dl, dn in defs:
                if dl <= ln:
                    name = dn
                else:
                    break
            return name

        discovered = set()
        for i, l in enumerate(lines):
            if re.search(r"\b(save_index|rebuild_kb)\(", l) and \
                    not re.match(r"^def (save_index|rebuild_kb)\(", l):
                discovered.add(enclosing(i + 1))
        self.assertEqual(
            discovered, set(_EXPECTED_INDEX_WRITER_DEFS),
            "structured-index writer functions changed — classify the new/removed "
            "one (register its command or add a documented exemption): missing=%s "
            "unexpected=%s" % (
                sorted(set(_EXPECTED_INDEX_WRITER_DEFS) - discovered),
                sorted(discovered - set(_EXPECTED_INDEX_WRITER_DEFS))))

        registered = {f.__name__ for f in self.m._kb_writer_funcs()}
        for wdef, command in _EXPECTED_INDEX_WRITER_DEFS.items():
            if command is None:
                continue
            self.assertTrue(
                command in registered or command in _INDEX_WRITER_EXEMPT_COMMANDS,
                "index-writer '%s' -> command '%s' is neither schema-guarded nor a "
                "documented exemption" % (wdef, command))

    # --- newer-.schema refusal for EVERY writer (older CLI on newer KB) ------
    def test_schemaguard_all_writers_refuse_on_newer_kb(self):
        self._write_schema(self.m.KB_SCHEMA_VERSION + 1)
        for func in self.m._kb_writer_funcs():
            self.assertEqual(
                self._gate(func), 1,
                "writer %s must REFUSE (rc 1) on a newer KB" % func.__name__)

    def test_schemaguard_named_writers_refuse(self):
        # The specific verbs the plan calls out (previously unguarded).
        self._write_schema(self.m.KB_SCHEMA_VERSION + 1)
        for name in ("cmd_plan_new", "cmd_plan_set_status", "cmd_plan_claim",
                     "cmd_features", "cmd_skill_remove"):
            self.assertEqual(self._gate(getattr(self.m, name)), 1,
                             "%s must refuse on a newer KB" % name)

    def test_schemaguard_readers_exempt_on_newer_kb(self):
        self._write_schema(self.m.KB_SCHEMA_VERSION + 1)
        for name in ("cmd_recall", "cmd_query", "cmd_config", "cmd_metrics"):
            self.assertIsNone(self._gate(getattr(self.m, name)),
                              "%s is read-only and must NOT be gated" % name)

    # --- CLI-level realism (through main() + argparse) ----------------------
    def test_schemaguard_cli_writer_refuses_reader_works(self):
        self._write_schema(self.m.KB_SCHEMA_VERSION + 1)
        # a writer verb refuses loud...
        code, _o, err = run_cli(["plan", "new", "990101-x", "--title", "t"],
                                self.kdir)
        self.assertEqual(code, 1)
        self.assertIn("older CLI on a newer KB", err)
        # ...features too...
        code, _o, err = run_cli(["features"], self.kdir)
        self.assertEqual(code, 1)
        # ...while a reader still works.
        code, _o, _e = run_cli(["recall", "bm25 deterministic ranking"], self.kdir)
        self.assertEqual(code, 0)

    # --- migrate --list honesty + read-only ---------------------------------
    def test_schemaguard_migrate_list_older_cli(self):
        newer = self.m.KB_SCHEMA_VERSION + 1
        self._write_schema(newer)
        before = self._read_schema_raw()
        code, out, _e = run_cli(["migrate", "--list"], self.kdir)
        self.assertEqual(code, 0)
        self.assertIn("OLDER CLI", out)
        self.assertIn("NEWER KB", out)
        self.assertNotIn("up to date", out)
        # read-only: .schema byte-identical after --list.
        self.assertEqual(self._read_schema_raw(), before,
                         "migrate --list must not touch .schema")

    def test_schemaguard_migrate_list_read_only_when_stale(self):
        # cur < target: prints pending, still never creates/writes .schema here
        # beyond what already exists (bootstrap case: no .schema file).
        code, out, _e = run_cli(["migrate", "--list"], self.kdir)
        self.assertEqual(code, 0)
        self.assertFalse(os.path.exists(os.path.join(self.kdir, ".schema")),
                         "migrate --list must NOT create .schema")


if __name__ == "__main__":
    unittest.main()
