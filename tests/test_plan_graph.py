"""Tests for the PLAN-GRAPH orchestrator (feature 260722-plan-graph-orchestration).

The plan-graph layer is pure orchestration over the shipped `fanout` worktree-
lane substrate + the six-state plan machine. This module grows one TestCase per
plan task:

  TestPlanGraphParsers   — Task 1: the pure read-only parsers + lint rules
                           (plan_depends_on / plan_write_set / infer_write_set /
                           plan_checkpoint) and plan-lint R15/R16.
  TestManifestValidate   — Task 2 (later)
  TestSchedulerPure      — Task 3 (later)
  ...

Stdlib `unittest` only (pytest is NOT installed). The CLI is imported in-process
via tests._fixtures.load_cli so we test the REAL symbols, not a copy.

Runner:  python3 -m unittest tests.test_plan_graph -v
"""

import os
import tempfile
import unittest

try:
    from tests._fixtures import load_cli, REPO_ROOT, SyntheticKBTestCase
except ImportError:  # when run via `discover -s tests`, tests/ is on sys.path
    from _fixtures import load_cli, REPO_ROOT, SyntheticKBTestCase


CLI = load_cli()


# --- plan.md builders --------------------------------------------------------

def _plan(body, *, self_ext=False, target="~/workspace/agentware"):
    """A minimal but lint-valid-enough plan.md carrying the graph blocks in
    `body`. `self_ext` stamps the `> Type:` self-extension header the deterministic
    detector keys on."""
    typ = "> Type: **self-extension (package change)**\n" if self_ext else ""
    return (
        "# Plan — fixture\n\n"
        "> Status: draft\n"
        "%s"
        "\n## Target packages\n\n- %s\n\n"
        "%s\n"
        "## Tasks\n\n- ⬜ **1** do a thing\n  *Verify:* true\n\n"
        "## Acceptance criteria\n\n- [ ] it works\n"
        % (typ, target, body)
    )


class TestPlanGraphParsers(unittest.TestCase):
    # --- plan_depends_on -----------------------------------------------------

    def test_depends_on_default_logical(self):
        text = _plan("## Depends-on:\n\n- 260710-alpha\n- 260711-beta\n")
        self.assertEqual(
            CLI.plan_depends_on(text),
            [("260710-alpha", "logical"), ("260711-beta", "logical")],
        )

    def test_depends_on_explicit_kinds(self):
        text = _plan(
            "## Depends-on:\n\n"
            "- 260710-alpha logical\n"
            "- 260711-beta resource\n"
            "- 260712-gamma (resource)\n"
            "- 260713-delta:resource\n"
        )
        self.assertEqual(
            CLI.plan_depends_on(text),
            [
                ("260710-alpha", "logical"),
                ("260711-beta", "resource"),
                ("260712-gamma", "resource"),
                ("260713-delta", "resource"),
            ],
        )

    def test_depends_on_malformed_sentinel(self):
        text = _plan(
            "## Depends-on:\n\n"
            "- 260710-alpha weird\n"          # out-of-vocab kind
            "- bad slug here\n"               # extra tokens
            "- has/slash logical\n"           # slug not [A-Za-z0-9_-]+
        )
        edges = CLI.plan_depends_on(text)
        self.assertEqual([k for _, k in edges],
                         ["__malformed__", "__malformed__", "__malformed__"])

    def test_depends_on_absent_is_empty(self):
        self.assertEqual(CLI.plan_depends_on(_plan("")), [])

    def test_depends_on_stops_at_next_heading(self):
        text = _plan("## Depends-on:\n\n- 260710-alpha\n\n## Write set\n\n- docs/x.md\n")
        self.assertEqual(CLI.plan_depends_on(text), [("260710-alpha", "logical")])

    # --- plan_write_set ------------------------------------------------------

    def test_write_set_explicit(self):
        text = _plan("## Write set\n\n- docs/a.md\n- `scripts/agentware`\n")
        self.assertEqual(CLI.plan_write_set(text),
                         ["docs/a.md", "scripts/agentware"])

    def test_write_set_serialize_on_prefix(self):
        text = _plan("## Write set\n\n- serialize_on: scripts/agentware\n")
        self.assertEqual(CLI.plan_write_set(text), ["scripts/agentware"])

    def test_write_set_absent_is_empty(self):
        self.assertEqual(CLI.plan_write_set(_plan("")), [])

    def test_write_set_dedup_sorted(self):
        text = _plan("## Write set\n\n- b.md\n- a.md\n- a.md\n")
        self.assertEqual(CLI.plan_write_set(text), ["a.md", "b.md"])

    # --- infer_write_set -----------------------------------------------------

    def test_infer_self_ext_no_named_file_sentinel(self):
        # A self-extension plan naming no concrete kernel file collapses to the
        # {scripts/agentware} sentinel so two such plans serialize.
        text = _plan("A generic self-extension with no path named.",
                     self_ext=True, target="~/workspace/agentware")
        self.assertEqual(CLI.infer_write_set(text, REPO_ROOT),
                         ["scripts/agentware"])

    def test_infer_self_ext_named_kernel_file_narrows(self):
        text = _plan("Edit scripts/agentware and agentware.sh dispatch.",
                     self_ext=True)
        got = CLI.infer_write_set(text, REPO_ROOT)
        self.assertIn("scripts/agentware", got)
        self.assertIn("agentware.sh", got)

    def test_infer_non_self_ext_specific_dir(self):
        text = _plan("A docs-only change.", target=os.path.join(REPO_ROOT, "docs"))
        self.assertEqual(CLI.infer_write_set(text, REPO_ROOT), ["docs"])

    def test_infer_un_narrowable_whole_repo_sentinel(self):
        # Whole-repo target, no kernel mention -> the whole-repo sentinel, which
        # MUST overlap every repo-relative token (the old `<pkg>/**` form used the
        # repo basename as the glob root and overlapped nothing — audit finding).
        text = _plan("A vague change with no file named.",
                     target="~/workspace/agentware")
        got = CLI.infer_write_set(text, REPO_ROOT)
        self.assertEqual(got, [CLI._WRITE_SET_WHOLE_REPO])
        # The sentinel serializes against ANY concrete sibling write-set.
        self.assertTrue(CLI._write_sets_overlap(got, ["scripts/agentware"]))
        self.assertTrue(CLI._write_sets_overlap(got, ["docs/loop.md"]))
        self.assertTrue(CLI._graph_write_set_touches_kernel(got))

    # --- plan_checkpoint -----------------------------------------------------

    def test_checkpoint_default_none(self):
        self.assertEqual(CLI.plan_checkpoint(_plan("")), "none")

    def test_checkpoint_declared(self):
        text = _plan("", self_ext=True).replace(
            "> Status: draft\n", "> Status: draft\n> Checkpoint: manual\n")
        self.assertEqual(CLI.plan_checkpoint(text), "manual")

    # --- lint R15/R16 (via lint_plan directly, strict at the error ramp) ------

    def _lint(self, text, rel_path="work/260722-x/plan.md"):
        # Force the error ramp so R15/R16 surface as ERRORS (they are WARN under
        # the current schema, hard-error once schema advances).
        return CLI.lint_plan(text, strict=True, rel_path=rel_path,
                             schema_version=CLI._PLAN_GRAPH_ERROR_SCHEMA)

    def test_lint_r15_self_edge(self):
        text = _plan("## Depends-on:\n\n- 260722-x\n")
        errors, _ = self._lint(text, rel_path="work/260722-x/plan.md")
        self.assertTrue(any(e["rule"] == "R15" and "self-edge" in e["message"]
                            for e in errors))

    def test_lint_r15_duplicate_edge(self):
        text = _plan("## Depends-on:\n\n- 260710-a\n- 260710-a\n")
        errors, _ = self._lint(text)
        self.assertTrue(any(e["rule"] == "R15" and "duplicate" in e["message"]
                            for e in errors))

    def test_lint_r15_malformed_row(self):
        text = _plan("## Depends-on:\n\n- 260710-a bogus-kind\n")
        errors, _ = self._lint(text)
        self.assertTrue(any(e["rule"] == "R15" and "malformed" in e["message"]
                            for e in errors))

    def test_lint_r16_selfext_requires_non_none(self):
        # Self-extension plan with no `> Checkpoint:` (defaults none) -> R16.
        text = _plan("", self_ext=True)
        errors, _ = self._lint(text)
        self.assertTrue(any(e["rule"] == "R16" for e in errors))

    def test_lint_r16_selfext_manual_ok(self):
        text = _plan("", self_ext=True).replace(
            "> Status: draft\n", "> Status: draft\n> Checkpoint: manual\n")
        errors, _ = self._lint(text)
        self.assertFalse(any(e["rule"] == "R16" for e in errors))

    def test_lint_r16_unknown_policy(self):
        text = _plan("").replace(
            "> Status: draft\n", "> Status: draft\n> Checkpoint: sometimes\n")
        errors, _ = self._lint(text)
        self.assertTrue(any(e["rule"] == "R16" and "not a recognized" in e["message"]
                            for e in errors))

    def test_lint_r15_r16_warn_under_current_schema(self):
        # Under the current schema (< ramp) the same defects are WARN, not ERROR,
        # so existing self-extension plans are never retroactively hard-failed.
        text = _plan("", self_ext=True)  # missing checkpoint
        errors, warnings = CLI.lint_plan(text, strict=True,
                                         rel_path="work/260722-x/plan.md",
                                         schema_version=0)
        self.assertFalse(any(e["rule"] == "R16" for e in errors))
        self.assertTrue(any(w["rule"] == "R16" for w in warnings))


# --- Task 2: graph MANIFEST schema + single-writer writers + validator -------

def _manifest(nodes=None, edges=None, *, name="260722-g", schema=None,
              concurrency=3):
    """Build a graph manifest dict. `nodes` is a list of (feature, checkpoint,
    self_ext) tuples or full dicts; `edges` a list of (from, to[, kind]) tuples."""
    man = {
        "schema": CLI.GRAPH_MANIFEST_SCHEMA if schema is None else schema,
        "name": name,
        "concurrency": concurrency,
        "nodes": [],
        "edges": [],
    }
    for n in (nodes or []):
        if isinstance(n, dict):
            man["nodes"].append(n)
        else:
            feat, cp, se, ws = (list(n) + ["none", False, []])[:4]
            man["nodes"].append({"feature": feat, "checkpoint": cp,
                                 "self_extension": se, "write_set": list(ws)})
    for e in (edges or []):
        frm, to = e[0], e[1]
        kind = e[2] if len(e) > 2 else "logical"
        man["edges"].append({"from": frm, "to": to, "kind": kind})
    return man


class TestManifestValidate(unittest.TestCase):
    def _errs(self, manifest, kdir=None):
        return CLI.graph_manifest_errors(manifest, kdir=kdir)

    # --- happy path ----------------------------------------------------------

    def test_valid_manifest_no_errors(self):
        man = _manifest(
            nodes=[("260722-a",), ("260722-b",), ("260722-c",)],
            edges=[("260722-a", "260722-b"), ("260722-b", "260722-c")])
        self.assertEqual(self._errs(man), [])

    def test_empty_manifest_is_valid(self):
        self.assertEqual(self._errs(_manifest()), [])

    # --- schema / name / concurrency / bounds --------------------------------

    def test_bad_schema(self):
        man = _manifest(schema=99)
        self.assertTrue(any("schema" in e for e in self._errs(man)))

    def test_bad_graph_name(self):
        man = _manifest(name="bad name!")
        self.assertTrue(any("graph name" in e for e in self._errs(man)))

    def test_concurrency_out_of_range_low(self):
        self.assertTrue(any("concurrency" in e
                            for e in self._errs(_manifest(concurrency=0))))

    def test_concurrency_out_of_range_high(self):
        self.assertTrue(any("concurrency" in e
                            for e in self._errs(_manifest(concurrency=17))))

    def test_concurrency_bool_rejected(self):
        # True is an int in Python — must NOT be accepted as concurrency 1.
        self.assertTrue(any("concurrency" in e
                            for e in self._errs(_manifest(concurrency=True))))

    def test_node_count_over_max_refused(self):
        nodes = [("260722-n%d" % i,) for i in range(CLI.GRAPH_MAX_NODES + 1)]
        self.assertTrue(any("exceeds" in e for e in self._errs(_manifest(nodes))))

    def test_node_count_at_max_ok(self):
        nodes = [("260722-n%d" % i,) for i in range(CLI.GRAPH_MAX_NODES)]
        self.assertEqual(self._errs(_manifest(nodes)), [])

    # --- per-node ------------------------------------------------------------

    def test_bad_node_feature_name(self):
        man = _manifest(nodes=[("bad name",)])
        self.assertTrue(any("node feature name" in e for e in self._errs(man)))

    def test_duplicate_node(self):
        man = _manifest(nodes=[("260722-a",), ("260722-a",)])
        self.assertTrue(any("duplicate node" in e for e in self._errs(man)))

    def test_unknown_checkpoint_policy(self):
        man = _manifest(nodes=[("260722-a", "sometimes", False)])
        self.assertTrue(any("checkpoint policy" in e for e in self._errs(man)))

    def test_self_ext_without_manual_checkpoint_refused(self):
        # The audit-critical rule: a self-extension node MUST be a manual
        # checkpoint. A weaker checkpoint is a validate-time refusal.
        man = _manifest(nodes=[("260722-a", "review", True)])
        errs = self._errs(man)
        self.assertTrue(any("must carry a 'manual' checkpoint" in e
                            for e in errs))

    def test_self_ext_with_manual_checkpoint_ok(self):
        man = _manifest(nodes=[("260722-a", "manual", True)])
        self.assertEqual(self._errs(man), [])

    def test_self_ext_default_none_checkpoint_refused(self):
        man = _manifest(nodes=[("260722-a", "none", True)])
        self.assertTrue(any("must carry a 'manual' checkpoint" in e
                            for e in self._errs(man)))

    # --- INLINE edge integrity (NOT graph_relation_errors) -------------------

    def test_dangling_edge_target_caught(self):
        # THE audit case: a manifest {nodes,edges} dangling edge MUST be caught by
        # the inline check (graph_relation_errors/build_adjacency would silently
        # pass it — they consume the KB-index entries/relates shape).
        man = _manifest(nodes=[("260722-a",)],
                        edges=[("260722-a", "260722-missing")])
        errs = self._errs(man)
        self.assertTrue(any("dangling edge" in e and "260722-missing" in e
                            for e in errs))

    def test_dangling_edge_source_caught(self):
        man = _manifest(nodes=[("260722-a",)],
                        edges=[("260722-ghost", "260722-a")])
        self.assertTrue(any("dangling edge" in e and "260722-ghost" in e
                            for e in self._errs(man)))

    def test_dangling_not_via_graph_relation_errors(self):
        # Prove the manifest validator does NOT rely on the KB-index scanner:
        # graph_relation_errors over a manifest-shaped dict finds nothing (no
        # `entries`), yet graph_manifest_errors catches the dangling edge.
        man = _manifest(nodes=[("260722-a",)],
                        edges=[("260722-a", "260722-missing")])
        dangling, _unknown = CLI.graph_relation_errors(man)
        self.assertEqual(dangling, [])  # the KB scanner is blind to a manifest
        self.assertTrue(self._errs(man))  # the inline check is not

    def test_self_edge_refused(self):
        man = _manifest(nodes=[("260722-a",)], edges=[("260722-a", "260722-a")])
        self.assertTrue(any("self-edge" in e for e in self._errs(man)))

    def test_duplicate_edge_refused(self):
        man = _manifest(nodes=[("260722-a",), ("260722-b",)],
                        edges=[("260722-a", "260722-b"),
                               ("260722-a", "260722-b")])
        self.assertTrue(any("duplicate edge" in e for e in self._errs(man)))

    def test_unknown_edge_kind_refused(self):
        man = _manifest(nodes=[("260722-a",), ("260722-b",)],
                        edges=[("260722-a", "260722-b", "spooky")])
        self.assertTrue(any("unknown kind" in e for e in self._errs(man)))

    # --- cycles (multi-hop + resource) ---------------------------------------

    def test_multi_hop_cycle_refused(self):
        man = _manifest(
            nodes=[("260722-a",), ("260722-b",), ("260722-c",)],
            edges=[("260722-a", "260722-b"), ("260722-b", "260722-c"),
                   ("260722-c", "260722-a")])
        self.assertTrue(any("cycle detected" in e for e in self._errs(man)))

    def test_resource_cycle_refused(self):
        man = _manifest(
            nodes=[("260722-a",), ("260722-b",)],
            edges=[("260722-a", "260722-b", "resource"),
                   ("260722-b", "260722-a", "resource")])
        self.assertTrue(any("cycle detected" in e for e in self._errs(man)))

    # --- _graph_build determinism --------------------------------------------

    def test_graph_build_sorted_deterministic(self):
        man = _manifest(
            nodes=[("260722-c",), ("260722-a",), ("260722-b",)],
            edges=[("260722-a", "260722-c", "resource"),
                   ("260722-a", "260722-b")])
        nodes, forward, reverse = CLI._graph_build(man)
        self.assertEqual(nodes, ["260722-a", "260722-b", "260722-c"])
        # edges sorted (kind, target)
        self.assertEqual(forward["260722-a"],
                         [("logical", "260722-b"), ("resource", "260722-c")])
        self.assertEqual(reverse["260722-c"], [("resource", "260722-a")])
        # byte-identical across repeated builds
        self.assertEqual(CLI._graph_build(man), (nodes, forward, reverse))


class TestManifestWriters(SyntheticKBTestCase):
    """The single-writer CLI writers, exercised by calling the cmd_* functions
    directly with the KB env pinned (the `graph` argparse group is wired in
    Task 8; here we test the writer semantics, not the CLI dispatch)."""

    def _args(self, **kw):
        import types
        kw.setdefault("format", "text")
        return types.SimpleNamespace(**kw)

    def _seed_plan(self, feature, *, self_ext=False, checkpoint=None):
        text = _plan("", self_ext=self_ext)
        if checkpoint:
            text = text.replace(
                "> Status: draft\n",
                "> Status: draft\n> Checkpoint: %s\n" % checkpoint)
        path = os.path.join(self.kdir, "work", feature, "plan.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def _with_env(self, fn):
        import contextlib
        import io
        prev = os.environ.get("AGENTWARE_KNOWLEDGE_DIR")
        os.environ["AGENTWARE_KNOWLEDGE_DIR"] = self.kdir
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                return fn()
        finally:
            if prev is None:
                os.environ.pop("AGENTWARE_KNOWLEDGE_DIR", None)
            else:
                os.environ["AGENTWARE_KNOWLEDGE_DIR"] = prev

    def test_new_add_node_edge_roundtrip_and_graphmd(self):
        import json as _json
        self._seed_plan("260722-a")
        self._seed_plan("260722-b")

        def run():
            self.assertEqual(CLI.cmd_graph_new(self._args(name="g1")), 0)
            self.assertEqual(
                CLI.cmd_graph_add_node(self._args(name="g1", feature="260722-a")),
                0)
            self.assertEqual(
                CLI.cmd_graph_add_node(self._args(name="g1", feature="260722-b")),
                0)
            self.assertEqual(
                CLI.cmd_graph_add_edge(self._args(name="g1", src="260722-a",
                                                  dst="260722-b")), 0)
        self._with_env(run)

        mpath = CLI._graph_manifest_path(self.kdir, "g1")
        self.assertTrue(os.path.exists(mpath))
        with open(mpath, encoding="utf-8") as f:
            man = _json.load(f)
        self.assertEqual([n["feature"] for n in man["nodes"]],
                         ["260722-a", "260722-b"])
        self.assertEqual(man["edges"],
                         [{"from": "260722-a", "to": "260722-b",
                           "kind": "logical"}])
        # derived graph.md regenerated
        self.assertTrue(os.path.exists(CLI._graph_md_path(self.kdir, "g1")))

    def test_add_node_computes_self_extension_from_plan(self):
        self._seed_plan("260722-se", self_ext=True, checkpoint="manual")

        def run():
            self.assertEqual(CLI.cmd_graph_new(self._args(name="g2")), 0)
            self.assertEqual(
                CLI.cmd_graph_add_node(self._args(name="g2",
                                                  feature="260722-se")), 0)
        self._with_env(run)
        import json as _json
        with open(CLI._graph_manifest_path(self.kdir, "g2"), encoding="utf-8") as f:
            man = _json.load(f)
        self.assertTrue(man["nodes"][0]["self_extension"])
        self.assertEqual(man["nodes"][0]["checkpoint"], "manual")

    def test_add_node_infers_write_set_when_undeclared(self):
        # Finding 2/4/6/7/8: a plan with NO `## Write set` section must get the
        # CONSERVATIVE inferred write-set (never an empty set that overlaps
        # nothing). Two such same-target self-ext plans must therefore both stamp
        # a non-empty set that OVERLAPS, so the resource layer serializes them.
        self._seed_plan("260722-u1", self_ext=True, checkpoint="manual")
        self._seed_plan("260722-u2", self_ext=True, checkpoint="manual")

        def run():
            self.assertEqual(CLI.cmd_graph_new(self._args(name="gws")), 0)
            for f in ("260722-u1", "260722-u2"):
                self.assertEqual(
                    CLI.cmd_graph_add_node(self._args(name="gws", feature=f)), 0)
        self._with_env(run)
        import json as _json
        with open(CLI._graph_manifest_path(self.kdir, "gws"),
                  encoding="utf-8") as f:
            man = _json.load(f)
        ws = {n["feature"]: n["write_set"] for n in man["nodes"]}
        self.assertTrue(ws["260722-u1"], "undeclared write-set must be inferred")
        self.assertTrue(ws["260722-u2"])
        # Inferred sets of two same-target self-ext plans OVERLAP -> serialized.
        self.assertTrue(
            CLI._write_sets_overlap(ws["260722-u1"], ws["260722-u2"]))

    def test_add_node_uses_declared_write_set_verbatim(self):
        # The declared `## Write set` still WINS over inference when present.
        text = _plan("## Write set\n\n- docs/loop.md\n")
        path = os.path.join(self.kdir, "work", "260722-decl", "plan.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

        def run():
            self.assertEqual(CLI.cmd_graph_new(self._args(name="gdc")), 0)
            self.assertEqual(CLI.cmd_graph_add_node(
                self._args(name="gdc", feature="260722-decl")), 0)
        self._with_env(run)
        import json as _json
        with open(CLI._graph_manifest_path(self.kdir, "gdc"),
                  encoding="utf-8") as f:
            man = _json.load(f)
        self.assertEqual(man["nodes"][0]["write_set"], ["docs/loop.md"])

    def test_add_edge_that_closes_cycle_refused_no_write(self):
        import json as _json
        self._seed_plan("260722-a")
        self._seed_plan("260722-b")

        def run():
            CLI.cmd_graph_new(self._args(name="g3"))
            CLI.cmd_graph_add_node(self._args(name="g3", feature="260722-a"))
            CLI.cmd_graph_add_node(self._args(name="g3", feature="260722-b"))
            CLI.cmd_graph_add_edge(self._args(name="g3", src="260722-a",
                                              dst="260722-b"))
            # This edge would close a 2-cycle -> refused, NOT written.
            rc = CLI.cmd_graph_add_edge(self._args(name="g3", src="260722-b",
                                                   dst="260722-a"))
            return rc
        rc = self._with_env(run)
        self.assertEqual(rc, 1)
        with open(CLI._graph_manifest_path(self.kdir, "g3"), encoding="utf-8") as f:
            man = _json.load(f)
        self.assertEqual(len(man["edges"]), 1)  # the cycle edge was NOT persisted

    def test_add_edge_dangling_refused_no_write(self):
        self._seed_plan("260722-a")

        def run():
            CLI.cmd_graph_new(self._args(name="g4"))
            CLI.cmd_graph_add_node(self._args(name="g4", feature="260722-a"))
            return CLI.cmd_graph_add_edge(self._args(name="g4", src="260722-a",
                                                     dst="260722-ghost"))
        self.assertEqual(self._with_env(run), 1)


# --- Task 3: DETERMINISTIC scheduler CORE ------------------------------------

def _combined(manifest, derived):
    """A manifest with derived resource edges unioned in, then built into the
    (nodes, forward, reverse) triple the scheduler consumes."""
    m = dict(manifest)
    m["edges"] = list(manifest.get("edges") or []) + list(derived)
    return CLI._graph_build(m)


class TestSchedulerPure(unittest.TestCase):
    # --- topo_order ----------------------------------------------------------

    def test_topo_order_stable_lexical_tiebreak(self):
        # Diamond: a -> b, a -> c, b -> d, c -> d. Lexical tie-break puts b before
        # c (both ready after a). Byte-identical across repeated runs.
        man = _manifest(
            nodes=[("260722-d",), ("260722-c",), ("260722-b",), ("260722-a",)],
            edges=[("260722-a", "260722-b"), ("260722-a", "260722-c"),
                   ("260722-b", "260722-d"), ("260722-c", "260722-d")])
        nodes, forward, _rev = CLI._graph_build(man)
        order = CLI.topo_order(forward, nodes)
        self.assertEqual(order,
                         ["260722-a", "260722-b", "260722-c", "260722-d"])
        for _ in range(5):
            self.assertEqual(CLI.topo_order(forward, nodes), order)

    def test_topo_order_raises_on_cycle(self):
        man = _manifest(
            nodes=[("260722-a",), ("260722-b",)],
            edges=[("260722-a", "260722-b"), ("260722-b", "260722-a")])
        _nodes, forward, _rev = CLI._graph_build(man)
        with self.assertRaises(ValueError):
            CLI.topo_order(forward, ["260722-a", "260722-b"])

    def test_topo_order_orphans_are_lexical(self):
        man = _manifest(nodes=[("260722-z",), ("260722-a",), ("260722-m",)])
        nodes, forward, _rev = CLI._graph_build(man)
        self.assertEqual(CLI.topo_order(forward, nodes),
                         ["260722-a", "260722-m", "260722-z"])

    # --- write-set overlap ---------------------------------------------------

    def test_write_sets_overlap_exact_file(self):
        self.assertTrue(CLI._write_sets_overlap(
            ["scripts/agentware"], ["scripts/agentware", "docs/x.md"]))

    def test_write_sets_disjoint(self):
        self.assertFalse(CLI._write_sets_overlap(["docs/a.md"], ["docs/b.md"]))

    def test_write_sets_glob_cover(self):
        # A `pkg/**` un-narrowable sentinel covers any path under pkg/ and the
        # bare dir itself.
        self.assertTrue(CLI._write_sets_overlap(["pkg/**"], ["pkg/sub/x.py"]))
        self.assertTrue(CLI._write_sets_overlap(["pkg/**"], ["pkg"]))
        self.assertFalse(CLI._write_sets_overlap(["pkg/**"], ["other/x.py"]))

    # --- _graph_derive_resource_edges: the B1/A2/A1 clique -------------------

    def _assert_single_line(self, nodes, forward):
        """Assert the combined graph linearizes to ONE chain: a unique topo order
        in which every consecutive pair is directly connected."""
        order = CLI.topo_order(forward, nodes)
        succ = {n: {d for (_k, d) in forward.get(n, [])} for n in nodes}
        for earlier, later in zip(order, order[1:]):
            self.assertIn(later, succ[earlier],
                          "%s and %s are not directly ordered -> not a single "
                          "line (order=%s)" % (earlier, later, order))

    def test_same_file_clique_serializes_to_single_line(self):
        # Three plans ALL writing scripts/agentware, no logical edges: the derived
        # resource edges must serialize them into one line (deterministic).
        ws = ["scripts/agentware"]
        man = _manifest(nodes=[("260722-a1", "none", False, ws),
                               ("260722-a2", "none", False, ws),
                               ("260722-b1", "none", False, ws)])
        derived = CLI._graph_derive_resource_edges(man)
        self.assertTrue(all(e["kind"] == "resource" and e["derived"]
                            for e in derived))
        nodes, forward, _rev = _combined(man, derived)
        self.assertEqual(CLI.find_cycles(forward), [])
        self._assert_single_line(nodes, forward)

    def test_clique_with_adversarial_lexically_descending_logical_edge(self):
        # THE audit case: a same-file clique with a logical edge that runs
        # lexically DESCENDING (b1 -> a1, i.e. later-name -> earlier-name). The
        # pairwise 'logical-first-else-lexical' rule would synthesize a cycle;
        # orienting resource edges along ONE topo linearization must still yield a
        # single acyclic line.
        ws = ["scripts/agentware"]
        man = _manifest(
            nodes=[("260722-a1", "none", False, ws),
                   ("260722-a2", "none", False, ws),
                   ("260722-b1", "none", False, ws)],
            edges=[("260722-b1", "260722-a1")])  # lexically descending logical
        derived = CLI._graph_derive_resource_edges(man)
        nodes, forward, _rev = _combined(man, derived)
        self.assertEqual(CLI.find_cycles(forward), [])
        self._assert_single_line(nodes, forward)
        # The declared logical edge must be RESPECTED (b1 before a1).
        order = CLI.topo_order(forward, nodes)
        self.assertLess(order.index("260722-b1"), order.index("260722-a1"))

    def test_derive_is_deterministic_and_byte_identical(self):
        ws = ["scripts/agentware"]
        man = _manifest(nodes=[("260722-b1", "none", False, ws),
                               ("260722-a2", "none", False, ws),
                               ("260722-a1", "none", False, ws)])
        first = CLI._graph_derive_resource_edges(man)
        for _ in range(5):
            self.assertEqual(CLI._graph_derive_resource_edges(man), first)

    def test_disjoint_write_sets_derive_no_edges(self):
        man = _manifest(nodes=[("260722-a", "none", False, ["docs/a.md"]),
                               ("260722-b", "none", False, ["docs/b.md"])])
        self.assertEqual(CLI._graph_derive_resource_edges(man), [])

    def test_derive_skips_already_declared_pairs(self):
        # A logical edge already orders the overlapping pair -> no redundant
        # resource edge added.
        ws = ["scripts/agentware"]
        man = _manifest(nodes=[("260722-a", "none", False, ws),
                               ("260722-b", "none", False, ws)],
                        edges=[("260722-a", "260722-b")])
        self.assertEqual(CLI._graph_derive_resource_edges(man), [])

    # --- _graph_ready_set ----------------------------------------------------

    def test_ready_set_respects_deps_and_done_merged(self):
        man = _manifest(
            nodes=[("260722-a",), ("260722-b",), ("260722-c",)],
            edges=[("260722-a", "260722-b"), ("260722-b", "260722-c")])
        graph = CLI._graph_build(man)
        # nothing run yet -> only the root a is ready
        self.assertEqual(CLI._graph_ready_set(graph, {}), ["260722-a"])
        # a done+merged -> b unlocks; c still blocked (b not done)
        self.assertEqual(
            CLI._graph_ready_set(graph, {"260722-a": CLI.GRAPH_NODE_DONE}),
            ["260722-b"])

    def test_ready_set_failed_pred_leaves_dependent_pending(self):
        man = _manifest(nodes=[("260722-a",), ("260722-b",)],
                        edges=[("260722-a", "260722-b")])
        graph = CLI._graph_build(man)
        # a FAILED (not done) -> b is NOT ready (stays pending, never cancelled)
        self.assertEqual(
            CLI._graph_ready_set(graph, {"260722-a": CLI.GRAPH_NODE_FAILED}),
            [])

    def test_ready_set_running_node_not_redispatched(self):
        man = _manifest(nodes=[("260722-a",), ("260722-b",)])
        graph = CLI._graph_build(man)
        self.assertEqual(
            CLI._graph_ready_set(graph, {"260722-a": CLI.GRAPH_NODE_RUNNING}),
            ["260722-b"])

    def test_ready_set_diamond_waits_for_both(self):
        # d needs both b and c; both need a. d unlocks only when BOTH done.
        man = _manifest(
            nodes=[("260722-a",), ("260722-b",), ("260722-c",), ("260722-d",)],
            edges=[("260722-a", "260722-b"), ("260722-a", "260722-c"),
                   ("260722-b", "260722-d"), ("260722-c", "260722-d")])
        graph = CLI._graph_build(man)
        st = {"260722-a": CLI.GRAPH_NODE_DONE, "260722-b": CLI.GRAPH_NODE_DONE}
        self.assertEqual(CLI._graph_ready_set(graph, st), ["260722-c"])
        st["260722-c"] = CLI.GRAPH_NODE_DONE
        self.assertEqual(CLI._graph_ready_set(graph, st), ["260722-d"])

    def test_ready_set_byte_identical(self):
        man = _manifest(
            nodes=[("260722-a",), ("260722-b",), ("260722-c",)],
            edges=[("260722-a", "260722-c"), ("260722-b", "260722-c")])
        graph = CLI._graph_build(man)
        first = CLI._graph_ready_set(graph, {})
        for _ in range(5):
            self.assertEqual(CLI._graph_ready_set(graph, {}), first)

    # --- _graph_disposition (the full table + fail-closed) -------------------

    def test_disposition_table(self):
        table = {
            "completed": CLI.GRAPH_CONTINUE,
            "hit_max_iterations": CLI.GRAPH_BLOCK_SUBTREE,
            "post_hook_failure": CLI.GRAPH_BLOCK_SUBTREE,
            "pre_hook_abort": CLI.GRAPH_BLOCK_SUBTREE,
            "adversarial_review_blocked": CLI.GRAPH_HALT,
            "blocked_on_ambiguity": CLI.GRAPH_HALT,
            "quota_throttled": CLI.GRAPH_PAUSE,
            "merged": CLI.GRAPH_CONTINUE,
            "skipped": CLI.GRAPH_CONTINUE,
            "halt-package": CLI.GRAPH_HALT,
            "halt-kb": CLI.GRAPH_HALT,
            "halt-gate": CLI.GRAPH_HALT,
            "self_extension_reached": CLI.GRAPH_HALT,
            "manifest_mutated": CLI.GRAPH_HALT,
        }
        for outcome, expect in table.items():
            self.assertEqual(CLI._graph_disposition(outcome), expect,
                             "disposition(%r) should be %s" % (outcome, expect))

    def test_disposition_adversarial_review_blocked_not_unknown(self):
        # The audit HIGH: adversarial_review_blocked must HALT (R-EXEC-06), never
        # collapse to an unknown/continue.
        self.assertEqual(
            CLI._graph_disposition("adversarial_review_blocked"), CLI.GRAPH_HALT)

    def test_disposition_unknown_fails_closed_to_halt(self):
        self.assertEqual(CLI._graph_disposition("who_knows"), CLI.GRAPH_HALT)
        self.assertEqual(CLI._graph_disposition(""), CLI.GRAPH_HALT)
        self.assertEqual(CLI._graph_disposition(None), CLI.GRAPH_HALT)

    def test_disposition_premise_falsified_load_bearing_split(self):
        # node-local premise falsification -> block only the subtree
        self.assertEqual(
            CLI._graph_disposition("premise_falsified",
                                   node={"self_extension": False}),
            CLI.GRAPH_BLOCK_SUBTREE)
        # load-bearing (self-extension) shared-premise falsification -> HALT
        self.assertEqual(
            CLI._graph_disposition("premise_falsified",
                                   node={"self_extension": True}),
            CLI.GRAPH_HALT)
        self.assertEqual(
            CLI._graph_disposition("premise_falsified",
                                   node={"load_bearing": True}),
            CLI.GRAPH_HALT)


# --- Task 4: derived read-only NODE-STATUS resolver --------------------------

def _status_plan(status="running", *, open_markers=0, done_markers=1,
                 self_ext=False, claim=None, dup=False):
    """Build a plan.md whose header carries `status`, optional `Claimed-by`, and a
    controllable mix of open/done task markers. `claim` is a `who@host <ISO>` raw
    string; `dup=True` injects a SECOND `> Status:` line (degenerate header)."""
    typ = "> Type: **self-extension (package change)**\n" if self_ext else ""
    claim_line = "> Claimed-by: %s\n" % claim if claim else ""
    dup_line = "> Status: draft\n" if dup else ""
    tasks = []
    for i in range(1, done_markers + 1):
        tasks.append("- ✅ **%d** done task\n  *Verify:* true\n" % i)
    for i in range(done_markers + 1, done_markers + open_markers + 1):
        tasks.append("- ⬜ **%d** open task\n  *Verify:* true\n" % i)
    if not tasks:
        tasks.append("- ✅ **1** done task\n  *Verify:* true\n")
    return (
        "# Plan — node fixture\n\n"
        "> Status: %s\n%s%s%s"
        "\n## Target packages\n\n- ~/workspace/agentware\n\n"
        "## Tasks\n\n%s\n"
        "## Acceptance criteria\n\n- [ ] it works\n"
        % (status, claim_line, dup_line, typ, "".join(tasks))
    )


class TestNodeStatus(SyntheticKBTestCase):
    """Task 4: `_graph_node_status` combines the plan header, the RAW terminal
    outcome (never derive_outcome), the lane outcome, and a per-node watchdog into
    the record the scheduler consumes. All fixtures pin `--now` for determinism."""

    NOW = "2026-07-24T12:00:00Z"

    def _seed(self, feature, **kw):
        path = os.path.join(self.kdir, "work", feature, "plan.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_status_plan(**kw))

    def _terminals(self, feature, outcomes):
        """Write metrics.jsonl `terminal` rows (in order) for `feature`."""
        import json as _json
        logs = os.path.join(self.kdir, "logs")
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, "metrics.jsonl"), "w", encoding="utf-8") as f:
            for oc in outcomes:
                f.write(_json.dumps({"event": "terminal", "feature": feature,
                                     "outcome": oc, "ts": self.NOW}) + "\n")

    def _lanes(self, feature, *, merged=False):
        return {"lanes": {feature: {"feature": feature, "merged": merged,
                                    "kb_path": "/nope", "worktree": "/nope"}}}

    def _status(self, feature, **kw):
        node = kw.pop("node", None) or {"feature": feature}
        node.setdefault("feature", feature)
        kw.setdefault("now", self.NOW)
        return CLI._graph_node_status(self.kdir, node, **kw)

    # --- non-running plan states ---------------------------------------------
    def test_draft_is_pending(self):
        self._seed("260722-a", status="draft")
        r = self._status("260722-a")
        self.assertEqual(r["status"], CLI.GRAPH_NODE_PENDING)

    def test_ready_is_pending(self):
        self._seed("260722-a", status="ready")
        self.assertEqual(self._status("260722-a")["status"],
                         CLI.GRAPH_NODE_PENDING)

    def test_done_merged_is_done(self):
        self._seed("260722-a", status="done")
        r = self._status("260722-a", lanes=self._lanes("260722-a", merged=True))
        self.assertEqual(r["status"], CLI.GRAPH_NODE_DONE)
        self.assertEqual(r["disposition"], CLI.GRAPH_CONTINUE)
        self.assertTrue(r["merged"])

    def test_done_unmerged_awaits_merge(self):
        self._seed("260722-a", status="done")
        r = self._status("260722-a", lanes=self._lanes("260722-a", merged=False))
        # done but NOT merged is NOT dep-satisfying — it awaits the merge-queue.
        self.assertEqual(r["status"], CLI.GRAPH_NODE_RUNNING)
        self.assertEqual(r["detail"], CLI.GRAPH_DETAIL_AWAITING_MERGE)

    def test_selfext_done_no_lane_is_dep_satisfying(self):
        # A self-extension node is run by the OPERATOR directly in the main
        # checkout (R-PKG-03) — never in a fanout lane — so on resume its plan is
        # `done` with NO lane record. It must resolve to DONE (dep-satisfying), NOT
        # AWAITING_MERGE (which would route a lane-less node into the merge-queue,
        # return 'halt-package', and HALT the graph forever). Audit finding.
        self._seed("260722-se", status="done", self_ext=True,
                   open_markers=0, done_markers=2)
        node = {"feature": "260722-se", "self_extension": True}
        r = CLI._graph_node_status(self.kdir, node, self.NOW,
                                   lanes={"lanes": {}})
        self.assertEqual(r["status"], CLI.GRAPH_NODE_DONE)
        self.assertEqual(r["disposition"], CLI.GRAPH_CONTINUE)
        self.assertFalse(r["merged"])

    def test_nonselfext_done_no_lane_still_awaits_merge(self):
        # The relaxation is SCOPED to self-extension nodes only: a normal node is
        # always lane-run, so done + no-lane still awaits the merge-queue (unchanged).
        self._seed("260722-a", status="done", open_markers=0, done_markers=2)
        node = {"feature": "260722-a", "self_extension": False}
        r = CLI._graph_node_status(self.kdir, node, self.NOW, lanes={"lanes": {}})
        self.assertEqual(r["status"], CLI.GRAPH_NODE_RUNNING)
        self.assertEqual(r["detail"], CLI.GRAPH_DETAIL_AWAITING_MERGE)

    def test_superseded_blocks(self):
        self._seed("260722-a", status="superseded")
        r = self._status("260722-a")
        self.assertEqual(r["status"], CLI.GRAPH_NODE_BLOCKED)
        self.assertEqual(r["disposition"], CLI.GRAPH_BLOCK_SUBTREE)

    def test_cancelled_blocks(self):
        self._seed("260722-a", status="cancelled")
        self.assertEqual(self._status("260722-a")["status"],
                         CLI.GRAPH_NODE_BLOCKED)

    def test_duplicate_header_degenerate(self):
        self._seed("260722-a", status="running", dup=True)
        r = self._status("260722-a")
        self.assertEqual(r["status"], CLI.GRAPH_NODE_BLOCKED)
        self.assertEqual(r["detail"], CLI.GRAPH_DETAIL_DEGENERATE)
        self.assertEqual(r["disposition"], CLI.GRAPH_BLOCK_SUBTREE)

    def test_missing_plan_halts(self):
        r = self._status("260722-ghost")
        self.assertEqual(r["status"], CLI.GRAPH_NODE_HALT)
        self.assertEqual(r["detail"], CLI.GRAPH_DETAIL_PLAN_MISSING)
        self.assertEqual(r["disposition"], CLI.GRAPH_HALT)

    # --- running + terminal outcome ------------------------------------------
    def test_running_no_terminal_is_running(self):
        self._seed("260722-a", status="running",
                   claim="alice@host %s" % self.NOW)
        self.assertEqual(self._status("260722-a")["status"],
                         CLI.GRAPH_NODE_RUNNING)

    def test_done_unfinalized_straggler(self):
        # Loop emitted `completed` but the plan header is still running.
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["completed"])
        r = self._status("260722-a")
        self.assertEqual(r["status"], CLI.GRAPH_NODE_RUNNING)
        self.assertEqual(r["detail"], CLI.GRAPH_DETAIL_DONE_UNFINALIZED)
        self.assertEqual(r["raw_outcome"], "completed")

    def test_adversarial_review_blocked_halts(self):
        # THE key Task-4 assertion: the raw outcome is preserved by name and
        # resolves to HALT (R-EXEC-06) — never `unknown`/running.
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["adversarial_review_blocked"])
        r = self._status("260722-a")
        self.assertEqual(r["status"], CLI.GRAPH_NODE_HALT)
        self.assertEqual(r["disposition"], CLI.GRAPH_HALT)
        self.assertEqual(r["raw_outcome"], "adversarial_review_blocked")
        self.assertEqual(r["detail"], "adversarial_review_blocked")

    def test_adversarial_blocked_survives_where_derive_outcome_drops_it(self):
        # Contrast: derive_outcome collapses this to `unknown`; the raw read keeps
        # it, which is the whole reason for Task 4's separate resolver (audit HIGH).
        # An open marker keeps derive_outcome's artifact fallback at `unknown`
        # (it asserts `completed` only when every marker is ✅), isolating the
        # metrics-read behavior under test.
        self._seed("260722-a", status="running", open_markers=1, done_markers=1)
        self._terminals("260722-a", ["adversarial_review_blocked"])
        self.assertEqual(CLI.derive_outcome(self.kdir, "260722-a")["outcome"],
                         "unknown")
        self.assertEqual(
            CLI._graph_raw_terminal_outcome(self.kdir, "260722-a"),
            "adversarial_review_blocked")

    def test_blocked_on_ambiguity_halts(self):
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["blocked_on_ambiguity"])
        self.assertEqual(self._status("260722-a")["status"], CLI.GRAPH_NODE_HALT)

    def test_contained_failure_hit_max_iterations(self):
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["hit_max_iterations"])
        r = self._status("260722-a")
        self.assertEqual(r["status"], CLI.GRAPH_NODE_FAILED)
        self.assertEqual(r["disposition"], CLI.GRAPH_BLOCK_SUBTREE)
        # failure is NEVER written to the plan state — it honestly stays running.
        self.assertEqual(r["plan_state"], "running")

    def test_premise_falsified_load_bearing_halts(self):
        self._seed("260722-a", status="running", self_ext=True)
        self._terminals("260722-a", ["premise_falsified"])
        r = self._status("260722-a", node={"feature": "260722-a",
                                            "self_extension": True})
        self.assertEqual(r["status"], CLI.GRAPH_NODE_HALT)
        self.assertEqual(r["disposition"], CLI.GRAPH_HALT)

    def test_premise_falsified_node_local_blocks(self):
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["premise_falsified"])
        r = self._status("260722-a", node={"feature": "260722-a",
                                            "self_extension": False})
        self.assertEqual(r["status"], CLI.GRAPH_NODE_BLOCKED)
        self.assertEqual(r["disposition"], CLI.GRAPH_BLOCK_SUBTREE)

    def test_quota_throttled_pauses(self):
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["quota_throttled"])
        r = self._status("260722-a")
        self.assertEqual(r["disposition"], CLI.GRAPH_PAUSE)
        # pause is resumable, NOT a failure — the node stays running.
        self.assertEqual(r["status"], CLI.GRAPH_NODE_RUNNING)

    def test_unknown_outcome_fails_closed_halt(self):
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["gremlin"])
        self.assertEqual(self._status("260722-a")["status"], CLI.GRAPH_NODE_HALT)

    def test_last_definite_terminal_wins_unknown_skipped(self):
        # A later `unknown` never overrides an earlier definite outcome.
        self._seed("260722-a", status="running")
        self._terminals("260722-a", ["adversarial_review_blocked", "unknown"])
        self.assertEqual(
            CLI._graph_raw_terminal_outcome(self.kdir, "260722-a"),
            "adversarial_review_blocked")

    # --- per-node wall-clock watchdog ----------------------------------------
    def test_watchdog_fires_on_stale_claim(self):
        # Claim 30h old, no terminal, node-timeout 24h -> FAILED_WATCHDOG.
        self._seed("260722-a", status="running",
                   claim="alice@host 2026-07-23T06:00:00Z")
        r = self._status("260722-a", node_timeout_hours=24)
        self.assertEqual(r["status"], CLI.GRAPH_NODE_FAILED)
        self.assertEqual(r["detail"], CLI.GRAPH_DETAIL_FAILED_WATCHDOG)
        self.assertEqual(r["disposition"], CLI.GRAPH_BLOCK_SUBTREE)

    def test_watchdog_not_fired_on_recent_claim(self):
        self._seed("260722-a", status="running",
                   claim="alice@host 2026-07-24T11:00:00Z")  # 1h old
        r = self._status("260722-a", node_timeout_hours=24)
        self.assertEqual(r["status"], CLI.GRAPH_NODE_RUNNING)

    def test_watchdog_distinct_from_stalled_claim_hours(self):
        # The node-timeout default is DISTINCT from the 72h stale-claim window.
        self.assertNotEqual(CLI.GRAPH_NODE_TIMEOUT_HOURS, CLI.STALLED_CLAIM_HOURS)
        self.assertLess(CLI.GRAPH_NODE_TIMEOUT_HOURS, CLI.STALLED_CLAIM_HOURS)

    # --- determinism ----------------------------------------------------------
    def test_now_injectable_byte_identical(self):
        self._seed("260722-a", status="running",
                   claim="alice@host 2026-07-20T00:00:00Z")
        self._terminals("260722-a", ["hit_max_iterations"])
        node = {"feature": "260722-a"}
        r1 = CLI._graph_node_status(self.kdir, node, self.NOW, node_timeout_hours=24)
        r2 = CLI._graph_node_status(self.kdir, node, self.NOW, node_timeout_hours=24)
        self.assertEqual(r1, r2)

    # --- DONE_UNFINALIZED idempotent finalizer -------------------------------
    def _with_env(self, fn):
        import contextlib
        import io
        prev = os.environ.get("AGENTWARE_KNOWLEDGE_DIR")
        os.environ["AGENTWARE_KNOWLEDGE_DIR"] = self.kdir
        try:
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                return fn()
        finally:
            if prev is None:
                os.environ.pop("AGENTWARE_KNOWLEDGE_DIR", None)
            else:
                os.environ["AGENTWARE_KNOWLEDGE_DIR"] = prev

    def test_finalize_unfinalized_idempotent(self):
        # A straggler (running header, all markers done, completed terminal) is
        # finalized to done via the single-writer set-state, then becomes DONE once
        # merged; a second finalize is a no-op.
        self._seed("260722-a", status="running", open_markers=0, done_markers=2)
        self._terminals("260722-a", ["completed"])
        pre = self._status("260722-a")
        self.assertEqual(pre["detail"], CLI.GRAPH_DETAIL_DONE_UNFINALIZED)

        ok, _msg = self._with_env(
            lambda: CLI._graph_finalize_unfinalized(self.kdir, "260722-a",
                                                    now=self.NOW))
        self.assertTrue(ok)
        # Header now flipped to done (single-writer); done+merged -> DONE.
        post = self._status("260722-a",
                            lanes=self._lanes("260722-a", merged=True))
        self.assertEqual(post["plan_state"], "done")
        self.assertEqual(post["status"], CLI.GRAPH_NODE_DONE)
        # Idempotent: finalizing again is a no-op success.
        ok2, msg2 = self._with_env(
            lambda: CLI._graph_finalize_unfinalized(self.kdir, "260722-a",
                                                    now=self.NOW))
        self.assertTrue(ok2)
        self.assertIn("no-op", msg2)

    def test_finalize_refuses_open_markers(self):
        # An open marker means the loop did NOT truly finish — the open-markers==0
        # gate refuses the running->done finalize.
        self._seed("260722-a", status="running", open_markers=1, done_markers=1)
        self._terminals("260722-a", ["completed"])
        ok, _msg = self._with_env(
            lambda: CLI._graph_finalize_unfinalized(self.kdir, "260722-a",
                                                    now=self.NOW))
        self.assertFalse(ok)
        # Still running — never silently forced done.
        self.assertEqual(self._status("260722-a")["plan_state"], "running")


# --- Task 5: LANE DRIVER + wave EXECUTOR + caps ------------------------------

def _have_git_t5():
    import subprocess
    try:
        subprocess.run(["git", "--version"], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, check=True)
        return True
    except Exception:  # pragma: no cover
        return False


HAVE_GIT_T5 = _have_git_t5()


class _FakeHandle:
    """A stand-in for a subprocess.Popen handle: `.wait()` returns instantly so the
    wave barrier never touches a real process."""

    def __init__(self):
        self.waited = False

    def wait(self):
        self.waited = True
        return 0


class TestDriverAndCaps(unittest.TestCase):
    """Task 5: the lane DRIVER, the deterministic wave EXECUTOR, and the cap +
    kernel-serialization + self-ext-halt decisions — all drivable with NO real
    subprocess via the injectable spawn/prepare/merge seams."""

    NOW = "2026-07-24T12:00:00Z"

    def setUp(self):
        import shutil
        self.kdir = tempfile.mkdtemp(prefix="agentware-t5-kb-")
        self.addCleanup(shutil.rmtree, self.kdir, True)
        self.fanout_dir = tempfile.mkdtemp(prefix="agentware-t5-fanout-")
        self.addCleanup(shutil.rmtree, self.fanout_dir, True)
        self._prev_fanout = os.environ.get("AGENTWARE_FANOUT_DIR")
        os.environ["AGENTWARE_FANOUT_DIR"] = self.fanout_dir
        self.addCleanup(self._restore_fanout)
        self._orig_spawn = CLI._graph_spawn
        self.addCleanup(lambda: setattr(CLI, "_graph_spawn", self._orig_spawn))

    def _restore_fanout(self):
        if self._prev_fanout is None:
            os.environ.pop("AGENTWARE_FANOUT_DIR", None)
        else:
            os.environ["AGENTWARE_FANOUT_DIR"] = self._prev_fanout

    # -- seeding helpers ------------------------------------------------------
    def _seed(self, feature, **kw):
        path = os.path.join(self.kdir, "work", feature, "plan.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_status_plan(**kw))

    def _terminal(self, feature, outcome):
        import json as _json
        logs = os.path.join(self.kdir, "logs")
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, "metrics.jsonl"), "a", encoding="utf-8") as f:
            f.write(_json.dumps({"event": "terminal", "feature": feature,
                                 "outcome": outcome, "ts": self.NOW}) + "\n")

    def _stubs(self, outcomes=None):
        """Injectable seams that model a lane loop WITHOUT a subprocess. `outcomes`
        maps feature -> terminal outcome ('completed' flips the plan to done; any
        other stays running-with-a-terminal, i.e. an honest contained failure)."""
        outcomes = outcomes or {}
        dispatched = []

        def prepare(reg, feature, repo, kb_repo, integration):
            reg["lanes"][feature] = {
                "feature": feature, "merged": False,
                "kb_path": "/nope", "worktree": "/nope",
                "feat_branch": "feat/%s" % feature, "kb_branch": "kb/%s" % feature}
            return "created", None

        def spawn(rec, feature):
            dispatched.append(feature)
            oc = outcomes.get(feature, "completed")
            if oc == "completed":
                self._seed(feature, status="done", open_markers=0, done_markers=2)
                self._terminal(feature, "completed")
            else:
                # A dispatched lane claims first (running); failure is NEVER
                # written to a plan state — the loop honestly STAYS running and its
                # verdict lives only in the metrics terminal event.
                self._seed(feature, status="running",
                           claim="lane@host %s" % self.NOW,
                           open_markers=1, done_markers=1)
                self._terminal(feature, oc)
            return _FakeHandle()

        def merge(reg, feature, repo, kb_repo, integration, gates=None):
            reg["lanes"][feature]["merged"] = True
            return "merged"

        return dispatched, prepare, spawn, merge

    def _run(self, manifest, outcomes=None, **kw):
        dispatched, prepare, spawn, merge = self._stubs(outcomes)
        res = CLI._graph_execute(
            self.kdir, manifest, self.NOW, repo=self.kdir, kb_repo=self.kdir,
            spawn=spawn, merge=merge, prepare=prepare,
            bootstrap=False, check_env=False, **kw)
        return res, dispatched

    # -- orchestrator-env fail-closed -----------------------------------------
    def test_orchestrator_env_blockers_detects_each_switch(self):
        for v in CLI.GRAPH_KILL_SWITCH_VARS:
            self.assertEqual(CLI._graph_orchestrator_env_blockers({v: "1"}), [v])
        self.assertEqual(CLI._graph_orchestrator_env_blockers({}), [])
        # an empty / whitespace value is NOT "present".
        self.assertEqual(
            CLI._graph_orchestrator_env_blockers(
                {"AGENTWARE_DISABLE_RELEASE_GATE": "  "}), [])

    def test_execute_refuses_on_orchestrator_kill_switch(self):
        man = _manifest(nodes=[("260722-a",)])
        env = {"AGENTWARE_SKIP_ADVERSARIAL_REVIEW": "1"}
        res = CLI._graph_execute(self.kdir, man, self.NOW, repo=self.kdir,
                                 kb_repo=self.kdir, bootstrap=False,
                                 check_env=True, environ=env)
        self.assertEqual(res["status"], "halt")
        self.assertEqual(res["reason"], "orchestrator-kill-switch")
        self.assertIn("AGENTWARE_SKIP_ADVERSARIAL_REVIEW", res["offenders"])

    # -- lane env allow-list --------------------------------------------------
    def test_lane_env_built_by_allowlist_excludes_kill_switches(self):
        env_in = {"PATH": "/usr/bin", "HOME": "/home/x"}
        for v in CLI.GRAPH_KILL_SWITCH_VARS:
            env_in[v] = "1"
        env_in["AGENTWARE_DREAM"] = "1"  # any other stray AGENTWARE_* toggle
        out = CLI._graph_lane_env("/lane/kb", environ=env_in)
        for v in CLI.GRAPH_KILL_SWITCH_VARS:
            self.assertNotIn(v, out)
        self.assertNotIn("AGENTWARE_DREAM", out)
        self.assertEqual(out["AGENTWARE_KNOWLEDGE_DIR"], "/lane/kb")
        self.assertEqual(out["PATH"], "/usr/bin")
        self.assertEqual(out["GIT_TERMINAL_PROMPT"], "0")

    def test_drive_lane_uses_spawn_seam_with_allowlist_env(self):
        captured = {}

        def fake_spawn(argv, cwd, env):
            captured.update(argv=argv, cwd=cwd, env=env)
            return _FakeHandle()

        CLI._graph_spawn = fake_spawn
        rec = {"feature": "260722-a", "kb_path": "/lane/kb",
               "worktree": "/lane/pkg"}
        CLI._graph_drive_lane(rec)
        self.assertEqual(captured["argv"], ["./agentware.sh", "260722-a"])
        self.assertEqual(captured["cwd"], "/lane/pkg")
        self.assertEqual(captured["env"]["AGENTWARE_KNOWLEDGE_DIR"], "/lane/kb")
        for v in CLI.GRAPH_KILL_SWITCH_VARS:
            self.assertNotIn(v, captured["env"])

    # -- pure cap + serialization decisions -----------------------------------
    def test_select_dispatch_respects_cap(self):
        ready = ["a", "b", "c", "d"]
        ws = {f: [] for f in ready}
        se = {f: False for f in ready}
        sel = CLI._graph_select_dispatch(ready, ws, se, 2)
        self.assertEqual(sel["dispatch"], ["a", "b"])
        self.assertEqual(sel["deferred"], ["c", "d"])

    def test_select_dispatch_serializes_kernel_lanes(self):
        ready = ["a", "b", "c"]
        ws = {"a": ["scripts/agentware"], "b": ["scripts/agentware"],
              "c": ["docs"]}
        se = {f: False for f in ready}
        sel = CLI._graph_select_dispatch(ready, ws, se, 3)
        # exactly ONE kernel lane dispatched; the disjoint docs lane rides along.
        self.assertIn("a", sel["dispatch"])
        self.assertIn("c", sel["dispatch"])
        self.assertNotIn("b", sel["dispatch"])
        self.assertIn("b", sel["deferred"])

    def test_select_dispatch_surfaces_self_ext_as_halt(self):
        ready = ["a", "s"]
        sel = CLI._graph_select_dispatch(
            ready, {"a": [], "s": []}, {"a": False, "s": True}, 3)
        self.assertEqual(sel["halt_selfext"], ["s"])
        self.assertEqual(sel["dispatch"], ["a"])

    def test_should_reprovision_pure(self):
        self.assertTrue(CLI._graph_should_reprovision(True, False))
        self.assertFalse(CLI._graph_should_reprovision(True, True))
        self.assertFalse(CLI._graph_should_reprovision(False, False))

    def test_write_set_touches_kernel(self):
        self.assertTrue(CLI._graph_write_set_touches_kernel(["scripts/agentware"]))
        self.assertTrue(
            CLI._graph_write_set_touches_kernel(["scripts/agentware/**"]))
        self.assertFalse(CLI._graph_write_set_touches_kernel(["docs", "README.md"]))

    # -- executor end-to-end (stub seams, deterministic) ----------------------
    def test_executor_serializes_kernel_clique_disjoint_concurrent(self):
        man = _manifest(nodes=[
            ("260722-a1", "none", False, ["scripts/agentware"]),
            ("260722-a2", "none", False, ["scripts/agentware"]),
            ("260722-a3", "none", False, ["scripts/agentware"]),
            ("260722-d", "none", False, ["docs"]),
        ], concurrency=3)
        for f in ("260722-a1", "260722-a2", "260722-a3", "260722-d"):
            self._seed(f, status="ready")
        res, dispatched = self._run(man)
        self.assertEqual(res["status"], "complete", res)
        # The same-file clique serialized in the single topo linearization.
        self.assertLess(dispatched.index("260722-a1"),
                        dispatched.index("260722-a2"))
        self.assertLess(dispatched.index("260722-a2"),
                        dispatched.index("260722-a3"))
        # The disjoint docs node fanned out in wave 1 (concurrent with a1).
        self.assertIn("260722-d", dispatched[:2])

    def test_executor_halts_at_self_ext_node(self):
        man = _manifest(nodes=[("260722-a",), ("260722-s", "manual", True)],
                        edges=[("260722-a", "260722-s")])
        self._seed("260722-a", status="ready")
        self._seed("260722-s", status="ready", self_ext=True)
        res, dispatched = self._run(man)
        self.assertEqual(res["status"], "halt")
        self.assertEqual(res["reason"], "awaiting-manual-selfext:260722-s")
        # It runs UP TO the self-ext node, then never autonomously spawns it.
        self.assertEqual(dispatched, ["260722-a"])

    def test_executor_recomputes_self_ext_from_plan_not_manifest(self):
        # Audit HIGH regression: `self_extension` is COMPUTED from the LIVE plan.md
        # at execution time, NEVER trusted from the operator-editable manifest
        # field. A node stored `self_extension:false` (a hand-edited graph.json, or
        # a plan.md that drifted to self-extension AFTER `add-node`) whose plan.md
        # IS self-extension must STILL halt for the mandatory manual checkpoint —
        # never be autonomously spawned with weakened gates.
        man = _manifest(nodes=[("260722-a",),
                               ("260722-s", "manual", False)],  # stored: NOT self-ext
                        edges=[("260722-a", "260722-s")])
        self._seed("260722-a", status="ready")
        self._seed("260722-s", status="ready", self_ext=True)  # plan.md: self-ext
        res, dispatched = self._run(man)
        self.assertEqual(res["status"], "halt")
        self.assertEqual(res["reason"], "awaiting-manual-selfext:260722-s")
        self.assertEqual(dispatched, ["260722-a"])

    def test_executor_halts_on_adversarial_review_blocked(self):
        man = _manifest(nodes=[("260722-a",)])
        self._seed("260722-a", status="running", claim="a@h %s" % self.NOW)
        self._terminal("260722-a", "adversarial_review_blocked")
        res, dispatched = self._run(man)
        self.assertEqual(res["status"], "halt")
        self.assertEqual(res["reason"], "adversarial_review_blocked")
        self.assertEqual(dispatched, [])

    def test_executor_contained_failure_blocks_subtree(self):
        man = _manifest(nodes=[("260722-x",), ("260722-y",), ("260722-z",)],
                        edges=[("260722-x", "260722-y")])
        for f in ("260722-x", "260722-y", "260722-z"):
            self._seed(f, status="ready")
        res, dispatched = self._run(
            man, outcomes={"260722-x": "hit_max_iterations"})
        self.assertEqual(res["status"], "blocked")
        # x failed (contained), y (its dependent) blocked, z (independent) ran.
        self.assertIn("260722-x", dispatched)
        self.assertIn("260722-z", dispatched)
        self.assertNotIn("260722-y", dispatched)
        self.assertIn("260722-x", res["pending"])
        self.assertIn("260722-y", res["pending"])

    def test_executor_empty_manifest_is_noop_complete(self):
        res, dispatched = self._run(_manifest())
        self.assertEqual(res["status"], "complete")
        self.assertEqual(dispatched, [])

    def test_selfext_done_no_lane_resume_continues_not_halt_package(self):
        # Finding 0/5: after the operator runs a self-ext node in the MAIN checkout
        # (no lane), resume must CONTINUE to its dependents — never route the
        # lane-less node into the merge-queue (which returns 'halt-package' and
        # HALTs the whole graph forever). The `merge` seam here faithfully models
        # the REAL integrator's contract: a lane-less node -> 'halt-package'.
        man = _manifest(
            nodes=[("260722-se", "manual", True, []),
                   ("260722-d", "none", False, [])],
            edges=[("260722-se", "260722-d")])
        self._seed("260722-se", status="done", self_ext=True,
                   open_markers=0, done_markers=2)          # operator-run, NO lane
        self._seed("260722-d", status="ready", open_markers=1, done_markers=0)
        dispatched, prepare, spawn, _stub = self._stubs()

        def merge(reg, feature, repo, kb_repo, integration, gates=None):
            # Mirror `_graph_merge_lane`: `if not rec: return "halt-package"`.
            if (reg.get("lanes") or {}).get(feature) is None:
                return "halt-package"
            reg["lanes"][feature]["merged"] = True
            return "merged"

        res = CLI._graph_execute(
            self.kdir, man, self.NOW, repo=self.kdir, kb_repo=self.kdir,
            spawn=spawn, merge=merge, prepare=prepare,
            bootstrap=False, check_env=False)
        self.assertEqual(res["status"], "complete", res)
        self.assertIn("260722-d", dispatched)          # dependent ran
        # SE was NEVER handed to the merge-queue.
        self.assertNotIn("260722-se",
                         [m["feature"] for m in res["merged"]])

    def test_unfinalizable_straggler_halts_instead_of_hanging(self):
        # Finding 1/9: a DONE_UNFINALIZED straggler whose finalize is REFUSED (the
        # loop emitted `completed` but left an open marker, so running->done is
        # gated out) must fail closed, NOT spin the tick loop forever without
        # advancing `waves` (which would defeat the GRAPH_TICK_CEILING backstop).
        man = _manifest(nodes=[("260722-s",)])
        self._seed("260722-s", status="running",
                   claim="lane@host %s" % self.NOW,
                   open_markers=1, done_markers=1)          # open marker blocks finalize
        self._terminal("260722-s", "completed")             # but loop said completed
        dispatched, prepare, spawn, merge = self._stubs()
        res = CLI._graph_execute(
            self.kdir, man, self.NOW, repo=self.kdir, kb_repo=self.kdir,
            spawn=spawn, merge=merge, prepare=prepare,
            bootstrap=False, check_env=False, tick_ceiling=25)
        self.assertEqual(res["status"], "halt")
        self.assertTrue(res["reason"].startswith("unfinalizable-straggler:"),
                        res["reason"])
        self.assertEqual(res["node"], "260722-s")

    def test_executor_deterministic_dispatch_order(self):
        man = _manifest(nodes=[("260722-a",), ("260722-b",), ("260722-c",)],
                        concurrency=1)
        for f in ("260722-a", "260722-b", "260722-c"):
            self._seed(f, status="ready")
        _r1, d1 = self._run(man)
        # Fresh state, same inputs -> byte-identical dispatch order.
        for f in ("260722-a", "260722-b", "260722-c"):
            self._seed(f, status="ready")
        os.remove(os.path.join(self.kdir, "logs", "metrics.jsonl"))
        CLI.save_lanes({"lanes": {}})
        _r2, d2 = self._run(man)
        self.assertEqual(d1, d2)
        self.assertEqual(d1, ["260722-a", "260722-b", "260722-c"])

    # -- base-off-integration + stale-base reprovision (real git) -------------
    @unittest.skipUnless(HAVE_GIT_T5, "git not available")
    def test_prepare_lane_reprovisions_off_stale_integration(self):
        import subprocess
        import shutil

        def git(cwd, *a):
            subprocess.run(["git", "-C", cwd] + list(a), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=True, text=True)

        repo = tempfile.mkdtemp(prefix="t5-pkg-")
        self.addCleanup(shutil.rmtree, repo, True)
        kb = tempfile.mkdtemp(prefix="t5-kb-")
        self.addCleanup(shutil.rmtree, kb, True)
        for r, rel, content in ((repo, "scripts/agentware", "x = 1\n"),
                                (kb, "MAIN.md", "# kb\n")):
            p = os.path.join(r, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            git(r, "init", "-q", "-b", "main")
            git(r, "config", "user.email", "t@t")
            git(r, "config", "user.name", "t")
            git(r, "add", "-A")
            git(r, "commit", "-q", "-m", "seed")

        integration = "integration"
        ok, err = CLI._graph_bootstrap_integration(repo, kb, integration)
        self.assertTrue(ok, err)

        reg, _e = CLI.load_lanes()
        feat = "260722-r"
        lane = CLI._fanout_lane_paths(feat, repo, kb)

        action, perr = CLI._graph_prepare_lane(reg, feat, repo, kb, integration)
        self.assertIsNone(perr, perr)
        self.assertEqual(action, "created")
        # The lane was cut off integration -> integration IS an ancestor of feat.
        self.assertTrue(CLI._is_ancestor(repo, integration, lane["feat_branch"]))

        # Advance integration beyond the lane (simulating a sibling's merge).
        with open(os.path.join(repo, "NEW.txt"), "w", encoding="utf-8") as f:
            f.write("new\n")
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", "advance")
        git(repo, "branch", "-f", integration, "main")
        self.assertFalse(CLI._is_ancestor(repo, integration, lane["feat_branch"]))

        # Prepare again -> detects the stale base -> tears down + re-cuts off HEAD.
        action2, perr2 = CLI._graph_prepare_lane(reg, feat, repo, kb, integration)
        self.assertIsNone(perr2, perr2)
        self.assertEqual(action2, "reprovisioned")
        self.assertTrue(CLI._is_ancestor(repo, integration, lane["feat_branch"]))


class TestMergeGates(unittest.TestCase):
    """Task 6: the EXTENDED (never weakened) per-node merge-gate chain
    (`_graph_merge_gates`) and the explicit operator-only manual checkpoint
    barrier (`graph checkpoint`)."""

    NOW = "2026-07-24T12:00:00Z"

    def setUp(self):
        import shutil
        self.kdir = tempfile.mkdtemp(prefix="agentware-t6-kb-")
        self.addCleanup(shutil.rmtree, self.kdir, True)
        self.fanout_dir = tempfile.mkdtemp(prefix="agentware-t6-fanout-")
        self.addCleanup(shutil.rmtree, self.fanout_dir, True)
        self._env_prev = {}
        for k, v in (("AGENTWARE_FANOUT_DIR", self.fanout_dir),
                     ("AGENTWARE_KNOWLEDGE_DIR", self.kdir)):
            self._env_prev[k] = os.environ.get(k)
            os.environ[k] = v
        self.addCleanup(self._restore_env)

    def _restore_env(self):
        for k, v in self._env_prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # -- seeding helpers ------------------------------------------------------
    def _seed(self, feature, **kw):
        path = os.path.join(self.kdir, "work", feature, "plan.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_status_plan(**kw))

    def _terminal(self, feature, outcome):
        import json as _json
        logs = os.path.join(self.kdir, "logs")
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, "metrics.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(_json.dumps({"event": "terminal", "feature": feature,
                                 "outcome": outcome, "ts": self.NOW}) + "\n")

    def _write_manifest(self, name, nodes, edges=None, concurrency=3):
        import json as _json
        man = _manifest(nodes=nodes, edges=edges, name=name,
                        concurrency=concurrency)
        mpath = CLI._graph_manifest_path(self.kdir, name)
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        with open(mpath, "w", encoding="utf-8") as f:
            _json.dump(man, f)
        return man

    def _args(self, **kw):
        import types
        kw.setdefault("format", "text")
        kw.setdefault("now", self.NOW)
        return types.SimpleNamespace(**kw)

    def _ckpt_log(self, feature):
        with open(os.path.join(self.kdir, "work", feature, "plan.md"),
                  encoding="utf-8") as f:
            return f.read()

    def _plan_state(self, feature):
        text = self._ckpt_log(feature)
        state, _cnt = CLI._read_plan_status(text)
        return state

    # -- _graph_merge_gates: always a SUPERSET of the defaults -----------------
    def test_plain_node_gates_are_exactly_the_defaults_verbatim(self):
        gates = CLI._graph_merge_gates({"feature": "260722-a"})
        # Re-listed VERBATIM (the merge-queue --gate REPLACES the defaults, so a
        # per-node list MUST re-list them or it would silently drop them).
        self.assertEqual(gates, list(CLI.FANOUT_DEFAULT_GATES))

    def test_gates_always_a_superset_and_prefix_of_defaults(self):
        for node in ({"feature": "260722-a"},
                     {"feature": "260722-a", "self_extension": True},
                     {"feature": "260722-a", "load_bearing": True}):
            gates = CLI._graph_merge_gates(node)
            # superset — no default is ever removed
            self.assertTrue(set(CLI.FANOUT_DEFAULT_GATES) <= set(gates))
            # verbatim order-preserving prefix — the replace-not-append guard
            self.assertEqual(gates[:len(CLI.FANOUT_DEFAULT_GATES)],
                             list(CLI.FANOUT_DEFAULT_GATES))

    def test_selfext_node_appends_gate_release_full_and_verify(self):
        gates = CLI._graph_merge_gates(
            {"feature": "260722-se", "self_extension": True})
        self.assertIn("scripts/agentware gate release --full", gates)
        self.assertTrue(any("verify-premises" in g for g in gates))
        self.assertTrue(set(CLI.FANOUT_DEFAULT_GATES) <= set(gates))

    def test_load_bearing_nonselfext_appends_verify_not_release(self):
        gates = CLI._graph_merge_gates(
            {"feature": "260722-lb", "load_bearing": True})
        self.assertTrue(any("verify-premises" in g for g in gates))
        self.assertNotIn("scripts/agentware gate release --full", gates)

    def test_verify_premises_gate_is_kb_rooted_and_strict(self):
        gates = CLI._graph_merge_gates(
            {"feature": "260722-se", "self_extension": True})
        vp = [g for g in gates if "verify-premises" in g][0]
        self.assertIn('$AGENTWARE_KNOWLEDGE_DIR/work/260722-se/plan.md', vp)
        self.assertIn("--strict", vp)

    # -- the executor hands the merge seam a SUPERSET, on the integrated tree --
    def test_execute_hands_superset_gates_to_merge_on_integration(self):
        man = self._write_manifest(
            "260722-mg", nodes=[("260722-lb", "none", False, [])])
        man["nodes"][0]["load_bearing"] = True   # force a strict superset
        self._seed("260722-lb", status="done", open_markers=0, done_markers=2)
        self._terminal("260722-lb", "completed")
        captured = {}

        def prepare(reg, feature, repo, kb_repo, integration):
            reg["lanes"][feature] = {
                "feature": feature, "merged": False,
                "feat_branch": "feat/%s" % feature,
                "kb_branch": "kb/%s" % feature}
            return "created", None

        def spawn(rec, feature):
            return _FakeHandle()

        def merge(reg, feature, repo, kb_repo, integration, gates=None):
            captured["gates"] = gates
            captured["repo"] = repo
            reg.setdefault("lanes", {}).setdefault(
                feature, {"feature": feature})["merged"] = True
            return "merged"

        res = CLI._graph_execute(
            self.kdir, man, self.NOW, repo=self.kdir, kb_repo=self.kdir,
            spawn=spawn, merge=merge, prepare=prepare,
            bootstrap=False, check_env=False)
        self.assertEqual(res["status"], "complete", res)
        self.assertIn("gates", captured)
        self.assertTrue(set(CLI.FANOUT_DEFAULT_GATES) <= set(captured["gates"]))
        self.assertTrue(any("verify-premises" in g for g in captured["gates"]))
        # The gates run on the INTEGRATION tree (the repo handed to the merge).
        self.assertEqual(captured["repo"], self.kdir)

    def test_merge_lane_runs_nonempty_gate_list_and_halts_on_failure(self):
        # Finding 12: exercise the REAL `_graph_merge_lane` gate-execution loop
        # with a NON-EMPTY gate list. The only prior caller passed gates=[] (gates
        # off), so a regression that skipped the `for g in glist: _run_gate(...)`
        # loop or stopped returning 'halt-gate' on a failing gate would ship
        # undetected — this locks the integrated-tree review-then-proceed floor.
        def _reg():
            return {"lanes": {"260722-g": {
                "feature": "260722-g", "merged": False,
                "feat_branch": "feat/260722-g", "kb_branch": "kb/260722-g"}}}

        orig = {n: getattr(CLI, n) for n in (
            "_mq_integrate_package", "_mq_integrate_kb", "_run_gate",
            "git_is_work_tree", "git_is_clean", "save_lanes")}
        self.addCleanup(lambda: [setattr(CLI, n, f) for n, f in orig.items()])
        CLI._mq_integrate_package = lambda *a, **k: ("merged", "")
        CLI._mq_integrate_kb = lambda *a, **k: ("merged", "")
        CLI.git_is_work_tree = lambda *a, **k: False   # no KB repo -> skip KB path
        CLI.git_is_clean = lambda *a, **k: True
        CLI.save_lanes = lambda *a, **k: None

        # (a) all gates PASS -> merged, and EACH gate in the list actually ran.
        ran = []
        CLI._run_gate = lambda g, *a, **k: (ran.append(g) or (0, ""))
        st = CLI._graph_merge_lane(_reg(), "260722-g", "/repo", None,
                                   "integration", gates=["gate-a", "gate-b"])
        self.assertEqual(st, "merged")
        self.assertEqual(ran, ["gate-a", "gate-b"])

        # (b) a FAILING gate -> halt-gate (the gate loop is load-bearing).
        ran2 = []

        def failing(g, *a, **k):
            ran2.append(g)
            return (0, "") if g == "gate-a" else (1, "boom")

        CLI._run_gate = failing
        st2 = CLI._graph_merge_lane(_reg(), "260722-g", "/repo", None,
                                    "integration", gates=["gate-a", "gate-b"])
        self.assertEqual(st2, "halt-gate")
        self.assertIn("gate-b", ran2)

    # -- graph checkpoint: the operator-only manual barrier --------------------
    def test_manual_checkpoint_refuses_loop_actor_no_write(self):
        self._write_manifest("260722-cp",
                             nodes=[("260722-se", "manual", True, [])])
        self._seed("260722-se", status="running", self_ext=True)
        rc = CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-se", verdict="proceed", actor="loop"))
        self.assertEqual(rc, 1)
        # NOTHING written — no checkpoint log on a refusal.
        self.assertNotIn(CLI._PLAN_CHECKPOINT_LOG_HEADING,
                         self._ckpt_log("260722-se"))

    def test_manual_checkpoint_accepts_operator_and_appends_log(self):
        self._write_manifest("260722-cp",
                             nodes=[("260722-se", "manual", True, [])])
        self._seed("260722-se", status="running", self_ext=True)
        rc = CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-se", verdict="proceed",
            actor="operator"))
        self.assertEqual(rc, 0)
        body = self._ckpt_log("260722-se")
        self.assertIn(CLI._PLAN_CHECKPOINT_LOG_HEADING, body)
        self.assertIn("proceed by operator", body)

    def test_review_checkpoint_accepts_loop_verdict(self):
        self._write_manifest("260722-cp",
                             nodes=[("260722-r", "review", False, [])])
        self._seed("260722-r", status="running")
        rc = CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-r", verdict="hold", actor="loop"))
        self.assertEqual(rc, 0)
        self.assertIn("hold by loop", self._ckpt_log("260722-r"))

    def test_checkpoint_log_is_append_only(self):
        self._write_manifest("260722-cp",
                             nodes=[("260722-r", "review", False, [])])
        self._seed("260722-r", status="running")
        CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-r", verdict="hold", actor="operator"))
        CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-r", verdict="proceed",
            actor="operator"))
        body = self._ckpt_log("260722-r")
        self.assertEqual(body.count(CLI._PLAN_CHECKPOINT_LOG_HEADING), 1)
        self.assertIn("hold by operator", body)
        self.assertIn("proceed by operator", body)

    def test_abort_subtree_cancels_transitive_successors(self):
        self._write_manifest(
            "260722-cp",
            nodes=[("260722-a", "manual", True, []),
                   ("260722-b", "none", False, []),
                   ("260722-c", "none", False, [])],
            edges=[("260722-a", "260722-b"), ("260722-b", "260722-c")])
        self._seed("260722-a", status="running", self_ext=True)
        self._seed("260722-b", status="ready")
        self._seed("260722-c", status="ready")
        rc = CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-a", verdict="abort-subtree",
            actor="operator"))
        self.assertEqual(rc, 0)
        # The whole transitive successor set is cancelled via the set-state matrix.
        self.assertEqual(self._plan_state("260722-b"), "cancelled")
        self.assertEqual(self._plan_state("260722-c"), "cancelled")
        # The node itself is NOT cancelled (only its successors).
        self.assertEqual(self._plan_state("260722-a"), "running")

    def test_abort_subtree_leaves_terminal_successor_untouched(self):
        self._write_manifest(
            "260722-cp",
            nodes=[("260722-a", "manual", True, []),
                   ("260722-b", "none", False, [])],
            edges=[("260722-a", "260722-b")])
        self._seed("260722-a", status="running", self_ext=True)
        # A `done` successor has no legal edge to cancelled — left as-is.
        self._seed("260722-b", status="done", open_markers=0, done_markers=2)
        rc = CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-a", verdict="abort-subtree",
            actor="operator"))
        self.assertEqual(rc, 0)
        self.assertEqual(self._plan_state("260722-b"), "done")

    def test_checkpoint_unknown_node_refused(self):
        self._write_manifest("260722-cp",
                             nodes=[("260722-a", "none", False, [])])
        self._seed("260722-a", status="running")
        rc = CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-ghost", verdict="proceed",
            actor="operator"))
        self.assertEqual(rc, 1)

    def test_checkpoint_unknown_verdict_refused(self):
        self._write_manifest("260722-cp",
                             nodes=[("260722-a", "review", False, [])])
        self._seed("260722-a", status="running")
        rc = CLI.cmd_graph_checkpoint(self._args(
            name="260722-cp", node="260722-a", verdict="bogus",
            actor="operator"))
        self.assertEqual(rc, 1)


class TestApprovalResumeObservability(unittest.TestCase):
    """Task 7: the `--i-approve-manifest` gate + sha256 content-pin, resume as pure
    recomputation (DONE-skip / hash-mutation refusal / FAILED-park-unless-retry),
    the self-extension HALT-for-manual-checkpoint, deterministic `graph status`,
    and the additive `graph`/`graph_node` observability events."""

    NOW = "2026-07-24T12:00:00Z"

    def setUp(self):
        import shutil
        self.kdir = tempfile.mkdtemp(prefix="agentware-t7-kb-")
        self.addCleanup(shutil.rmtree, self.kdir, True)
        self.fanout_dir = tempfile.mkdtemp(prefix="agentware-t7-fanout-")
        self.addCleanup(shutil.rmtree, self.fanout_dir, True)
        self._env_prev = {}
        for k, v in (("AGENTWARE_FANOUT_DIR", self.fanout_dir),
                     ("AGENTWARE_KNOWLEDGE_DIR", self.kdir)):
            self._env_prev[k] = os.environ.get(k)
            os.environ[k] = v
        self.addCleanup(self._restore_env)
        # Neutralize the real-git bootstrap + lane seams so the command wrappers
        # never touch a real repo/subprocess. A drive/prepare/merge call in these
        # fixtures is a BUG (the graph should halt/complete without dispatch); the
        # stubs record so a test can assert they were never called.
        self.calls = {"drive": [], "prepare": [], "merge": []}
        self._orig = {
            "boot": CLI._graph_bootstrap_integration,
            "drive": CLI._graph_drive_lane,
            "prepare": CLI._graph_prepare_lane,
            "merge": CLI._graph_merge_lane,
        }
        CLI._graph_bootstrap_integration = lambda *a, **k: (True, None)
        CLI._graph_drive_lane = lambda rec, *a, **k: (
            self.calls["drive"].append(rec) or _FakeHandle())
        CLI._graph_prepare_lane = lambda *a, **k: (
            self.calls["prepare"].append(a) or ("created", None))
        CLI._graph_merge_lane = lambda *a, **k: (
            self.calls["merge"].append(a) or "merged")
        self.addCleanup(self._restore_seams)

    def _restore_env(self):
        for k, v in self._env_prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _restore_seams(self):
        for name, fn in self._orig.items():
            setattr(CLI, {"boot": "_graph_bootstrap_integration",
                          "drive": "_graph_drive_lane",
                          "prepare": "_graph_prepare_lane",
                          "merge": "_graph_merge_lane"}[name], fn)

    # -- seeding helpers ------------------------------------------------------
    def _seed(self, feature, **kw):
        path = os.path.join(self.kdir, "work", feature, "plan.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_status_plan(**kw))

    def _terminal(self, feature, outcome):
        import json as _json
        logs = os.path.join(self.kdir, "logs")
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, "metrics.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(_json.dumps({"event": "terminal", "feature": feature,
                                 "outcome": outcome, "ts": self.NOW}) + "\n")

    def _mark_merged(self, *features):
        data, _ = CLI.load_lanes()
        for feature in features:
            data["lanes"][feature] = {"feature": feature, "merged": True}
        CLI.save_lanes(data)

    def _write_manifest(self, name, nodes, edges=None, concurrency=3):
        import json as _json
        man = _manifest(nodes=nodes, edges=edges, name=name,
                        concurrency=concurrency)
        mpath = CLI._graph_manifest_path(self.kdir, name)
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        with open(mpath, "w", encoding="utf-8") as f:
            _json.dump(man, f)
        return man

    def _args(self, **kw):
        import types
        kw.setdefault("format", "text")
        kw.setdefault("now", self.NOW)
        kw.setdefault("concurrency", None)
        kw.setdefault("node_timeout", None)
        return types.SimpleNamespace(**kw)

    def _ledger(self, name):
        return CLI._graph_ledger_read(self.kdir, name)

    # -- approval gate --------------------------------------------------------
    def test_run_refuses_without_approval(self):
        self._write_manifest("260722-ap",
                             nodes=[("260722-a", "none", False, [])])
        self._seed("260722-a", status="ready", open_markers=1, done_markers=0)
        rc = CLI.cmd_graph_run(self._args(name="260722-ap",
                                          i_approve_manifest=False))
        self.assertEqual(rc, 1)
        # nothing executed, no approval recorded
        self.assertEqual(self.calls["drive"], [])
        self.assertEqual(self._ledger("260722-ap"), [])

    def test_run_records_approval_with_manifest_sha256(self):
        man = self._write_manifest(
            "260722-ap", nodes=[("260722-a", "none", False, [])])
        # a done+merged single node -> completes with no dispatch.
        self._seed("260722-a", status="done", open_markers=0, done_markers=2)
        self._terminal("260722-a", "completed")
        self._mark_merged("260722-a")
        rc = CLI.cmd_graph_run(self._args(name="260722-ap",
                                          i_approve_manifest=True))
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls["drive"], [])   # nothing to dispatch
        appr = [e for e in self._ledger("260722-ap")
                if e.get("event") == "approval"]
        self.assertEqual(len(appr), 1)
        self.assertEqual(appr[0]["manifest_sha256"],
                         CLI._graph_manifest_sha256(man))
        # additive observability events landed and are ignored by iteration costs.
        events = CLI._read_metrics_jsonl(self.kdir)
        self.assertTrue(any(e.get("event") == "graph" for e in events))
        self.assertEqual(CLI.derive_iteration_costs(events), [])

    # -- self-extension node HALTs for a manual checkpoint --------------------
    def test_selfext_node_halts_for_manual_checkpoint(self):
        self._write_manifest(
            "260722-se", nodes=[("260722-x", "manual", True, [])])
        self._seed("260722-x", status="ready", self_ext=True,
                   open_markers=1, done_markers=0)
        rc = CLI.cmd_graph_run(self._args(name="260722-se",
                                          i_approve_manifest=True))
        self.assertEqual(rc, 1)                      # not complete
        # the self-ext node was NEVER autonomously spawned.
        self.assertEqual(self.calls["drive"], [])
        res = [e for e in self._ledger("260722-se")
               if e.get("event") == "result"][-1]
        self.assertEqual(res["status"], "halt")
        self.assertEqual(res["reason"], "awaiting-manual-selfext:260722-x")

    # -- resume: prior approval required + hash-mutation refusal ---------------
    def test_resume_refuses_without_prior_approval(self):
        self._write_manifest("260722-rs",
                             nodes=[("260722-a", "none", False, [])])
        self._seed("260722-a", status="done", open_markers=0, done_markers=2)
        rc = CLI.cmd_graph_resume(self._args(name="260722-rs", retry=False))
        self.assertEqual(rc, 1)
        self.assertEqual(self.calls["drive"], [])

    def test_resume_refuses_on_manifest_mutation(self):
        import json as _json
        self._write_manifest(
            "260722-rs", nodes=[("260722-a", "none", False, [])])
        self._seed("260722-a", status="done", open_markers=0, done_markers=2)
        self._terminal("260722-a", "completed")
        self._mark_merged("260722-a")
        self.assertEqual(
            CLI.cmd_graph_run(self._args(name="260722-rs",
                                         i_approve_manifest=True)), 0)
        # MUTATE the approved manifest on disk (add a node), then resume.
        man2 = _manifest(name="260722-rs",
                         nodes=[("260722-a", "none", False, []),
                                ("260722-b", "none", False, [])])
        with open(CLI._graph_manifest_path(self.kdir, "260722-rs"),
                  "w", encoding="utf-8") as f:
            _json.dump(man2, f)
        self._seed("260722-b", status="ready", open_markers=1, done_markers=0)
        rc = CLI.cmd_graph_resume(self._args(name="260722-rs", retry=False))
        self.assertEqual(rc, 1)
        res = [e for e in self._ledger("260722-rs")
               if e.get("event") == "result"][-1]
        self.assertEqual(res["reason"], "manifest-mutated")
        self.assertEqual(self.calls["drive"], [])   # refused before dispatch

    def test_resume_idempotent_over_partially_done_graph(self):
        self._write_manifest(
            "260722-rs",
            nodes=[("260722-a", "none", False, []),
                   ("260722-b", "none", False, [])],
            edges=[("260722-a", "260722-b")])
        for f in ("260722-a", "260722-b"):
            self._seed(f, status="done", open_markers=0, done_markers=2)
            self._terminal(f, "completed")
        self._mark_merged("260722-a", "260722-b")
        self.assertEqual(
            CLI.cmd_graph_run(self._args(name="260722-rs",
                                         i_approve_manifest=True)), 0)
        # A fresh resume recomputes purely from live state: every node is
        # done+merged -> complete, and NO done node is re-dispatched.
        rc = CLI.cmd_graph_resume(self._args(name="260722-rs", retry=False))
        self.assertEqual(rc, 0)
        self.assertEqual(self.calls["drive"], [])

    # -- FAILED parked unless --retry -----------------------------------------
    def test_retry_reoffers_a_failed_node_exactly_once(self):
        # Driven directly through _graph_execute with stub seams (no git): a
        # contained failure parks the node without --retry; a --retry resume
        # re-offers it once and it then completes.
        import shutil
        kdir = tempfile.mkdtemp(prefix="agentware-t7r-kb-")
        self.addCleanup(shutil.rmtree, kdir, True)
        fdir = tempfile.mkdtemp(prefix="agentware-t7r-fanout-")
        self.addCleanup(shutil.rmtree, fdir, True)
        prev = os.environ.get("AGENTWARE_FANOUT_DIR")
        os.environ["AGENTWARE_FANOUT_DIR"] = fdir
        self.addCleanup(lambda: os.environ.__setitem__(
            "AGENTWARE_FANOUT_DIR", prev) if prev is not None
            else os.environ.pop("AGENTWARE_FANOUT_DIR", None))

        def seed(feature, **kw):
            p = os.path.join(kdir, "work", feature, "plan.md")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(_status_plan(**kw))

        def terminal(feature, outcome):
            import json as _json
            logs = os.path.join(kdir, "logs")
            os.makedirs(logs, exist_ok=True)
            with open(os.path.join(logs, "metrics.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(_json.dumps({"event": "terminal", "feature": feature,
                                     "outcome": outcome, "ts": self.NOW}) + "\n")

        outcomes = {"260722-a": "hit_max_iterations"}
        spawned = []

        def prepare(reg, feature, repo, kb_repo, integration):
            reg["lanes"][feature] = {"feature": feature, "merged": False}
            return "created", None

        def spawn(rec, feature):
            spawned.append(feature)
            oc = outcomes.get(feature, "completed")
            if oc == "completed":
                seed(feature, status="done", open_markers=0, done_markers=2)
                terminal(feature, "completed")
            else:
                seed(feature, status="running",
                     claim="lane@host %s" % self.NOW,
                     open_markers=1, done_markers=1)
                terminal(feature, oc)
            return _FakeHandle()

        def merge(reg, feature, repo, kb_repo, integration, gates=None):
            reg["lanes"][feature]["merged"] = True
            return "merged"

        man = _manifest(name="260722-rt",
                        nodes=[("260722-a", "none", False, [])])
        seed("260722-a", status="ready", open_markers=1, done_markers=0)
        # First run: the node fails (contained) -> parked, blocked frontier.
        r1 = CLI._graph_execute(kdir, man, self.NOW, repo=kdir, kb_repo=kdir,
                                spawn=spawn, merge=merge, prepare=prepare,
                                bootstrap=False, check_env=False, retry=False)
        self.assertEqual(r1["status"], "blocked")
        self.assertEqual(spawned, ["260722-a"])      # dispatched once, then parked

        # Now make the retry succeed, and resume WITH --retry.
        outcomes["260722-a"] = "completed"
        r2 = CLI._graph_execute(kdir, man, self.NOW, repo=kdir, kb_repo=kdir,
                                spawn=spawn, merge=merge, prepare=prepare,
                                bootstrap=False, check_env=False, retry=True)
        self.assertEqual(r2["status"], "complete")
        self.assertEqual(spawned, ["260722-a", "260722-a"])  # re-offered once

    def test_failed_node_parked_without_retry(self):
        import shutil
        kdir = tempfile.mkdtemp(prefix="agentware-t7p-kb-")
        self.addCleanup(shutil.rmtree, kdir, True)
        fdir = tempfile.mkdtemp(prefix="agentware-t7p-fanout-")
        self.addCleanup(shutil.rmtree, fdir, True)
        prev = os.environ.get("AGENTWARE_FANOUT_DIR")
        os.environ["AGENTWARE_FANOUT_DIR"] = fdir
        self.addCleanup(lambda: os.environ.__setitem__(
            "AGENTWARE_FANOUT_DIR", prev) if prev is not None
            else os.environ.pop("AGENTWARE_FANOUT_DIR", None))
        spawned = []

        def prepare(reg, feature, repo, kb_repo, integration):
            reg["lanes"][feature] = {"feature": feature, "merged": False}
            return "created", None

        def spawn(rec, feature):
            spawned.append(feature)
            p = os.path.join(kdir, "work", feature, "plan.md")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(_status_plan(status="running",
                                     claim="lane@host %s" % self.NOW,
                                     open_markers=1, done_markers=1))
            import json as _json
            logs = os.path.join(kdir, "logs")
            os.makedirs(logs, exist_ok=True)
            with open(os.path.join(logs, "metrics.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(_json.dumps({"event": "terminal", "feature": feature,
                                     "outcome": "post_hook_failure",
                                     "ts": self.NOW}) + "\n")
            return _FakeHandle()

        def merge(reg, feature, repo, kb_repo, integration, gates=None):
            reg["lanes"][feature]["merged"] = True
            return "merged"

        man = _manifest(name="260722-pk",
                        nodes=[("260722-a", "none", False, [])])
        p0 = os.path.join(kdir, "work", "260722-a", "plan.md")
        os.makedirs(os.path.dirname(p0), exist_ok=True)
        with open(p0, "w", encoding="utf-8") as f:
            f.write(_status_plan(status="ready", open_markers=1, done_markers=0))
        r = CLI._graph_execute(kdir, man, self.NOW, repo=kdir, kb_repo=kdir,
                               spawn=spawn, merge=merge, prepare=prepare,
                               bootstrap=False, check_env=False, retry=False)
        self.assertEqual(r["status"], "blocked")
        # dispatched exactly once and then LEFT parked (no auto-retry, R-FAIL-04).
        self.assertEqual(spawned, ["260722-a"])

    # -- quota throttle pauses (never fails) ----------------------------------
    def test_quota_throttle_pauses_run(self):
        import shutil
        kdir = tempfile.mkdtemp(prefix="agentware-t7q-kb-")
        self.addCleanup(shutil.rmtree, kdir, True)
        fdir = tempfile.mkdtemp(prefix="agentware-t7q-fanout-")
        self.addCleanup(shutil.rmtree, fdir, True)
        prev = os.environ.get("AGENTWARE_FANOUT_DIR")
        os.environ["AGENTWARE_FANOUT_DIR"] = fdir
        self.addCleanup(lambda: os.environ.__setitem__(
            "AGENTWARE_FANOUT_DIR", prev) if prev is not None
            else os.environ.pop("AGENTWARE_FANOUT_DIR", None))

        def prepare(reg, feature, repo, kb_repo, integration):
            reg["lanes"][feature] = {"feature": feature, "merged": False}
            return "created", None

        def spawn(rec, feature):
            p = os.path.join(kdir, "work", feature, "plan.md")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(_status_plan(status="running",
                                     claim="lane@host %s" % self.NOW,
                                     open_markers=1, done_markers=1))
            import json as _json
            logs = os.path.join(kdir, "logs")
            os.makedirs(logs, exist_ok=True)
            with open(os.path.join(logs, "metrics.jsonl"), "a",
                      encoding="utf-8") as f:
                f.write(_json.dumps({"event": "terminal", "feature": feature,
                                     "outcome": "quota_throttled",
                                     "ts": self.NOW}) + "\n")
            return _FakeHandle()

        def merge(reg, feature, repo, kb_repo, integration, gates=None):
            return "merged"

        man = _manifest(name="260722-q",
                        nodes=[("260722-a", "none", False, [])])
        p0 = os.path.join(kdir, "work", "260722-a", "plan.md")
        os.makedirs(os.path.dirname(p0), exist_ok=True)
        with open(p0, "w", encoding="utf-8") as f:
            f.write(_status_plan(status="ready", open_markers=1, done_markers=0))
        r = CLI._graph_execute(kdir, man, self.NOW, repo=kdir, kb_repo=kdir,
                               spawn=spawn, merge=merge, prepare=prepare,
                               bootstrap=False, check_env=False)
        self.assertEqual(r["status"], "paused")
        self.assertTrue(r["reason"].startswith("quota-throttle:"))

    # -- deterministic status render ------------------------------------------
    def test_status_json_is_deterministic_and_pins_sha256(self):
        import io
        import contextlib
        man = self._write_manifest(
            "260722-st",
            nodes=[("260722-a", "none", False, []),
                   ("260722-b", "review", False, [])],
            edges=[("260722-a", "260722-b")])
        self._seed("260722-a", status="done", open_markers=0, done_markers=2)
        self._terminal("260722-a", "completed")
        self._mark_merged("260722-a")
        self._seed("260722-b", status="ready", open_markers=1, done_markers=0)
        CLI._graph_ledger_append(self.kdir, "260722-st", {
            "event": "approval", "graph": "260722-st",
            "manifest_sha256": CLI._graph_manifest_sha256(man), "ts": self.NOW})

        def render():
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = CLI.cmd_graph_status(self._args(name="260722-st",
                                                     format="json"))
            return rc, buf.getvalue()

        rc1, out1 = render()
        rc2, out2 = render()
        self.assertEqual(rc1, 0)
        self.assertEqual(out1, out2)                 # byte-identical
        import json as _json
        payload = _json.loads(out1)
        self.assertEqual(payload["node_count"], 2)
        self.assertEqual(payload["approved_sha256"],
                         CLI._graph_manifest_sha256(man))
        self.assertEqual(payload["manifest_sha256"], payload["approved_sha256"])
        self.assertFalse(payload["manifest_mutated"])
        by_feat = {n["feature"]: n for n in payload["nodes"]}
        self.assertEqual(by_feat["260722-a"]["status"], "done")
        self.assertTrue(by_feat["260722-a"]["merged"])
        self.assertEqual(by_feat["260722-b"]["status"], "pending")

    def test_status_reports_manifest_mutation(self):
        import io
        import contextlib
        import json as _json
        man = self._write_manifest(
            "260722-st", nodes=[("260722-a", "none", False, [])])
        self._seed("260722-a", status="ready", open_markers=1, done_markers=0)
        CLI._graph_ledger_append(self.kdir, "260722-st", {
            "event": "approval", "graph": "260722-st",
            "manifest_sha256": CLI._graph_manifest_sha256(man), "ts": self.NOW})
        # mutate the manifest concurrency (a content change).
        man["concurrency"] = 5
        with open(CLI._graph_manifest_path(self.kdir, "260722-st"),
                  "w", encoding="utf-8") as f:
            _json.dump(man, f)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            CLI.cmd_graph_status(self._args(name="260722-st", format="json"))
        payload = _json.loads(buf.getvalue())
        self.assertTrue(payload["manifest_mutated"])
        self.assertNotEqual(payload["approved_sha256"],
                            payload["manifest_sha256"])


# --- Task 9: [e2e] hermetic multi-node graph end-to-end ----------------------

class TestEndToEnd(unittest.TestCase):
    """Task 9: a full end-to-end drive of a hermetic multi-node graph through the
    REAL CLI command path (`cmd_graph_validate` / `cmd_graph_run
    --i-approve-manifest --now` / `cmd_graph_resume`), with the four side-effecting
    seams (`bootstrap`/`prepare`/`drive`/`merge`) monkeypatched to a fake
    `agentware.sh` that flips plan markers + emits a terminal metric — NO real loop,
    NO subprocess. `--now` is pinned so every run is byte-identical.

    The one graph exercises: a non-self-extension same-file write-set CLIQUE (with an
    adversarial lexically-descending logical edge) that must serialize to one line, a
    disjoint docs node that fans out concurrently, a logical chain whose dependent
    waits for done+merged, and a self-extension node that HALTS for a manual
    checkpoint instead of being autonomously spawned. Separate methods cover the
    `adversarial_review_blocked`→HALT and contained-failure-blocks-only-subtree
    dispositions and idempotent resume. A final REAL-GIT method proves the clique
    merges one-at-a-time off `integration` with no textual conflict."""

    NOW = "2026-07-24T12:00:00Z"

    # feature slugs -----------------------------------------------------------
    A1, A2, A3 = "260722-a1", "260722-a2", "260722-a3"       # same-file clique
    DOC = "260722-doc"                                        # disjoint docs node
    CH1, CH2 = "260722-ch1", "260722-ch2"                    # logical chain
    SE = "260722-se"                                          # self-extension node
    SHARED_WS = ["app/shared.py"]

    def setUp(self):
        import shutil
        self.kdir = tempfile.mkdtemp(prefix="agentware-t9-kb-")
        self.addCleanup(shutil.rmtree, self.kdir, True)
        self.fanout_dir = tempfile.mkdtemp(prefix="agentware-t9-fanout-")
        self.addCleanup(shutil.rmtree, self.fanout_dir, True)
        self._env_prev = {}
        for k, v in (("AGENTWARE_FANOUT_DIR", self.fanout_dir),
                     ("AGENTWARE_KNOWLEDGE_DIR", self.kdir)):
            self._env_prev[k] = os.environ.get(k)
            os.environ[k] = v
        # A kill-switch left in the operator shell would fail `graph run` closed —
        # scrub the four so the hermetic run is not derailed by the CI environment.
        for v in CLI.GRAPH_KILL_SWITCH_VARS:
            self._env_prev[v] = os.environ.get(v)
            os.environ.pop(v, None)
        self.addCleanup(self._restore_env)
        # Save/restore the four module-level side-effecting seams.
        self._orig = {n: getattr(CLI, n) for n in (
            "_graph_bootstrap_integration", "_graph_prepare_lane",
            "_graph_drive_lane", "_graph_merge_lane")}
        self.addCleanup(self._restore_seams)
        self.dispatched = []

    def _restore_env(self):
        for k, v in self._env_prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _restore_seams(self):
        for n, fn in self._orig.items():
            setattr(CLI, n, fn)

    # -- the fake-agentware.sh seams ------------------------------------------
    def _install_seams(self, outcomes=None):
        """Wire the four seams to a fake loop. `outcomes` maps feature -> terminal
        outcome; 'completed' (default) flips the plan to done, any other outcome
        leaves the plan honestly running with only the metrics terminal event (a
        contained failure). `bootstrap` is a no-op success; `prepare`/`merge` mutate
        + persist the lane registry so a later resume recomputes from live state."""
        outcomes = outcomes or {}

        def bootstrap(*a, **k):
            return (True, None)

        def prepare(reg, feature, repo, kb_repo, integration):
            reg.setdefault("lanes", {})[feature] = {
                "feature": feature, "merged": False,
                "kb_path": "/nope", "worktree": "/nope",
                "feat_branch": "feat/%s" % feature,
                "kb_branch": "kb/%s" % feature}
            CLI.save_lanes(reg)
            return "created", None

        def drive(rec, feature=None):
            feature = feature or rec.get("feature")
            self.dispatched.append(feature)
            oc = outcomes.get(feature, "completed")
            if oc == "completed":
                self._seed(feature, status="done", open_markers=0, done_markers=2)
                self._terminal(feature, "completed")
            else:
                self._seed(feature, status="running",
                           claim="lane@host %s" % self.NOW,
                           open_markers=1, done_markers=1)
                self._terminal(feature, oc)
            return _FakeHandle()

        def merge(reg, feature, repo, kb_repo, integration, gates=None):
            reg.setdefault("lanes", {}).setdefault(
                feature, {"feature": feature})["merged"] = True
            CLI.save_lanes(reg)
            return "merged"

        CLI._graph_bootstrap_integration = bootstrap
        CLI._graph_prepare_lane = prepare
        CLI._graph_drive_lane = drive
        CLI._graph_merge_lane = merge

    # -- seeding helpers ------------------------------------------------------
    def _seed(self, feature, *, self_ext=False, **kw):
        path = os.path.join(self.kdir, "work", feature, "plan.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_status_plan(self_ext=self_ext, **kw))

    def _terminal(self, feature, outcome):
        import json as _json
        logs = os.path.join(self.kdir, "logs")
        os.makedirs(logs, exist_ok=True)
        with open(os.path.join(logs, "metrics.jsonl"), "a",
                  encoding="utf-8") as f:
            f.write(_json.dumps({"event": "terminal", "feature": feature,
                                 "outcome": outcome, "ts": self.NOW}) + "\n")

    def _write_manifest(self, name, nodes, edges=None, concurrency=3):
        import json as _json
        man = _manifest(nodes=nodes, edges=edges, name=name,
                        concurrency=concurrency)
        mpath = CLI._graph_manifest_path(self.kdir, name)
        os.makedirs(os.path.dirname(mpath), exist_ok=True)
        with open(mpath, "w", encoding="utf-8") as f:
            _json.dump(man, f)
        return man

    def _args(self, **kw):
        import types
        kw.setdefault("format", "text")
        kw.setdefault("now", self.NOW)
        kw.setdefault("concurrency", None)
        kw.setdefault("node_timeout", None)
        kw.setdefault("deadline", CLI.GRAPH_DEADLINE_HOURS)
        return types.SimpleNamespace(**kw)

    def _quiet(self, fn):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            return fn()

    def _ledger(self, name):
        return CLI._graph_ledger_read(self.kdir, name)

    def _last_result(self, name):
        return [e for e in self._ledger(name)
                if e.get("event") == "result"][-1]

    def _node_status(self, feature, name):
        reg, _ = CLI.load_lanes()
        return CLI._graph_node_status(
            self.kdir, {"feature": feature}, self.NOW, lanes=reg)["status"]

    # -- the headline multi-node graph ----------------------------------------
    def _headline_manifest(self, name="260722-e2e"):
        return self._write_manifest(
            name,
            nodes=[
                (self.A1, "none", False, self.SHARED_WS),
                (self.A2, "none", False, self.SHARED_WS),
                (self.A3, "none", False, self.SHARED_WS),
                (self.DOC, "none", False, ["docs/notes.md"]),
                (self.CH1, "none", False, []),
                (self.CH2, "none", False, []),
                (self.SE, "manual", True, []),
            ],
            edges=[
                (self.A3, self.A1),         # adversarial lexically-descending logical
                (self.CH1, self.CH2),       # logical chain
                # the self-ext node sinks the whole autonomous portion so the graph
                # runs everything, THEN halts at the manual checkpoint.
                (self.A1, self.SE), (self.A2, self.SE), (self.A3, self.SE),
                (self.CH2, self.SE), (self.DOC, self.SE),
            ],
            concurrency=3)

    def _seed_headline(self):
        for f in (self.A1, self.A2, self.A3, self.DOC, self.CH1, self.CH2):
            self._seed(f, status="ready", open_markers=1, done_markers=0)
        self._seed(self.SE, status="ready", self_ext=True,
                   open_markers=1, done_markers=0)

    def test_headline_graph_runs_autonomous_then_halts_at_selfext(self):
        name = "260722-e2e"
        man = self._headline_manifest(name)
        self._seed_headline()
        self._install_seams()

        # 1) validate: acyclic, NON-empty, resource edges derivable along the topo
        #    linearization (the union stays a DAG).
        rc = self._quiet(lambda: CLI.cmd_graph_validate(
            self._args(name=name, format="text")))
        self.assertEqual(rc, 0)
        derived = CLI._graph_derive_resource_edges(man)
        self.assertTrue(derived)
        self.assertTrue(all(e["kind"] == "resource" and e["derived"]
                            for e in derived))
        full = {"nodes": man["nodes"],
                "edges": list(man["edges"]) + derived}
        _n, forward, _r = CLI._graph_build(full)
        self.assertEqual(CLI.find_cycles(forward), [])

        # 2) run: approved, pinned --now.
        rc = self._quiet(lambda: CLI.cmd_graph_run(
            self._args(name=name, i_approve_manifest=True)))
        # The graph HALTED at the self-extension node -> non-complete exit.
        self.assertEqual(rc, 1)
        res = self._last_result(name)
        self.assertEqual(res["status"], "halt")
        self.assertEqual(res["reason"], "awaiting-manual-selfext:%s" % self.SE)
        self.assertEqual(res["node"], self.SE)

        # 3) the self-extension node was NEVER autonomously spawned.
        self.assertNotIn(self.SE, self.dispatched)

        # 4) the same-file clique serialized to ONE line, in the derived topo order
        #    (respecting the adversarial descending logical edge a3 before a1).
        clique = {self.A1, self.A2, self.A3}
        order = CLI.topo_order(forward, _n)
        clique_order = [f for f in order if f in clique]
        disp_clique = [f for f in self.dispatched if f in clique]
        self.assertEqual(disp_clique, clique_order,
                         "clique must dispatch in the single-line topo order")
        self.assertLess(self.dispatched.index(self.A3),
                        self.dispatched.index(self.A1))

        # 5) the disjoint docs node + the chain root fanned out CONCURRENTLY with
        #    the clique's first node (wave 1 = the first `concurrency` dispatches).
        wave1 = set(self.dispatched[:3])
        self.assertIn(self.DOC, wave1)
        self.assertIn(self.CH1, wave1)
        self.assertEqual(len(wave1 & clique), 1)   # exactly one clique node in wave 1

        # 6) the chain dependent waited for its predecessor to be done+merged.
        self.assertLess(self.dispatched.index(self.CH1),
                        self.dispatched.index(self.CH2))

        # 7) every autonomous node reached done+merged before the halt.
        for f in (self.A1, self.A2, self.A3, self.DOC, self.CH1, self.CH2):
            self.assertEqual(self._node_status(f, name), CLI.GRAPH_NODE_DONE,
                             "%s should be done+merged" % f)

    def test_headline_run_is_byte_identical_across_replays(self):
        # Determinism floor: the same pinned --now over reset live state yields a
        # byte-identical dispatch sequence and the same halt.
        name = "260722-det"
        self._headline_manifest(name)

        def one_run():
            # fresh live state for each replay.
            self.dispatched = []
            for f in (self.A1, self.A2, self.A3, self.DOC, self.CH1, self.CH2):
                self._seed(f, status="ready", open_markers=1, done_markers=0)
            self._seed(self.SE, status="ready", self_ext=True,
                       open_markers=1, done_markers=0)
            logs = os.path.join(self.kdir, "logs", "metrics.jsonl")
            if os.path.exists(logs):
                os.remove(logs)
            CLI.save_lanes({"lanes": {}})
            self._install_seams()
            self._quiet(lambda: CLI.cmd_graph_run(
                self._args(name=name, i_approve_manifest=True)))
            return list(self.dispatched)

        d1 = one_run()
        d2 = one_run()
        self.assertEqual(d1, d2)

    def test_run_refuses_without_approval(self):
        name = "260722-noappr"
        self._headline_manifest(name)
        self._seed_headline()
        self._install_seams()
        rc = self._quiet(lambda: CLI.cmd_graph_run(
            self._args(name=name, i_approve_manifest=False)))
        self.assertEqual(rc, 1)
        self.assertEqual(self.dispatched, [])            # nothing dispatched
        self.assertEqual(self._ledger(name), [])         # no approval recorded

    # -- adversarial_review_blocked -> HALT (through the CLI path) -------------
    def test_adversarial_review_blocked_halts_graph(self):
        name = "260722-adv"
        self._write_manifest(name, nodes=[
            ("260722-adv-a", "none", False, []),
            ("260722-adv-b", "none", False, []),
        ])
        self._seed("260722-adv-a", status="ready", open_markers=1, done_markers=0)
        self._seed("260722-adv-b", status="ready", open_markers=1, done_markers=0)
        self._install_seams(outcomes={"260722-adv-a": "adversarial_review_blocked"})
        rc = self._quiet(lambda: CLI.cmd_graph_run(
            self._args(name=name, i_approve_manifest=True)))
        self.assertEqual(rc, 1)
        res = self._last_result(name)
        self.assertEqual(res["status"], "halt")
        self.assertEqual(res["reason"], "adversarial_review_blocked")

    # -- contained failure blocks ONLY its subtree, independent branch continues -
    def test_contained_failure_blocks_only_subtree(self):
        name = "260722-cf"
        # x -> y (dependent); z independent.
        self._write_manifest(name, nodes=[
            ("260722-cf-x", "none", False, []),
            ("260722-cf-y", "none", False, []),
            ("260722-cf-z", "none", False, []),
        ], edges=[("260722-cf-x", "260722-cf-y")])
        for f in ("260722-cf-x", "260722-cf-y", "260722-cf-z"):
            self._seed(f, status="ready", open_markers=1, done_markers=0)
        self._install_seams(outcomes={"260722-cf-x": "hit_max_iterations"})
        rc = self._quiet(lambda: CLI.cmd_graph_run(
            self._args(name=name, i_approve_manifest=True)))
        self.assertEqual(rc, 1)
        res = self._last_result(name)
        self.assertEqual(res["status"], "blocked")
        # x failed (contained); z (independent) ran; y (dependent) never dispatched.
        self.assertIn("260722-cf-x", self.dispatched)
        self.assertIn("260722-cf-z", self.dispatched)
        self.assertNotIn("260722-cf-y", self.dispatched)
        self.assertEqual(self._node_status("260722-cf-z", name),
                         CLI.GRAPH_NODE_DONE)   # independent branch completed

    # -- resume is idempotent over the already-done autonomous portion --------
    def test_resume_after_selfext_completes_idempotently(self):
        name = "260722-rse"
        self._headline_manifest(name)
        self._seed_headline()
        self._install_seams()
        # First run HALTs at the self-ext node (autonomous portion done+merged).
        rc = self._quiet(lambda: CLI.cmd_graph_run(
            self._args(name=name, i_approve_manifest=True)))
        self.assertEqual(rc, 1)
        self.assertEqual(self._last_result(name)["reason"],
                         "awaiting-manual-selfext:%s" % self.SE)

        # The operator records a proceed verdict at the mandatory manual checkpoint
        # (operator-only barrier), then runs the self-ext node themselves — modelled
        # here by flipping it done+merged.
        rc = self._quiet(lambda: CLI.cmd_graph_checkpoint(self._args(
            name=name, node=self.SE, verdict="proceed", actor="operator")))
        self.assertEqual(rc, 0)
        self._seed(self.SE, status="done", self_ext=True,
                   open_markers=0, done_markers=2)
        self._terminal(self.SE, "completed")
        reg, _ = CLI.load_lanes()
        reg.setdefault("lanes", {}).setdefault(
            self.SE, {"feature": self.SE})["merged"] = True
        CLI.save_lanes(reg)

        # Resume recomputes purely from live state: every node is now done+merged ->
        # complete, and NO already-done node is re-dispatched.
        self.dispatched = []
        rc = self._quiet(lambda: CLI.cmd_graph_resume(
            self._args(name=name, retry=False)))
        self.assertEqual(rc, 0)
        self.assertEqual(self.dispatched, [])
        self.assertEqual(self._last_result(name)["status"], "complete")

    # -- REAL GIT: the same-file clique merges off integration with no conflict --
    @unittest.skipUnless(HAVE_GIT_T5, "git not available")
    def test_same_file_clique_serializes_off_integration_no_conflict(self):
        """The B1/A2/A1 correctness floor with REAL git: bootstrap `integration`
        before wave 1, then provision + merge each same-file clique lane serially,
        each cut off the CURRENT integration HEAD. Because a successor branches off a
        base that already contains every merged predecessor, the serial
        `git merge --no-ff` applies each append cleanly — NO textual conflict."""
        import subprocess
        import shutil

        def git(cwd, *a):
            subprocess.run(["git", "-C", cwd] + list(a), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, check=True, text=True)

        repo = tempfile.mkdtemp(prefix="t9-pkg-")
        self.addCleanup(shutil.rmtree, repo, True)
        kb = tempfile.mkdtemp(prefix="t9-kb-")
        self.addCleanup(shutil.rmtree, kb, True)
        shared_rel = "app/shared.py"
        for r, rel, content in ((repo, shared_rel, "# shared\nbase = 0\n"),
                                (kb, "MAIN.md", "# kb\n")):
            p = os.path.join(r, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            git(r, "init", "-q", "-b", "main")
            git(r, "config", "user.email", "t@t")
            git(r, "config", "user.name", "t")
            git(r, "add", "-A")
            git(r, "commit", "-q", "-m", "seed")

        integration = "integration"
        ok, err = CLI._graph_bootstrap_integration(repo, kb, integration)
        self.assertTrue(ok, err)

        reg, _e = CLI.load_lanes()
        clique = [self.A1, self.A2, self.A3]
        for feat in clique:
            action, perr = CLI._graph_prepare_lane(reg, feat, repo, kb, integration)
            self.assertIsNone(perr, perr)
            lane = CLI._fanout_lane_paths(feat, repo, kb)
            # the lane was cut off the CURRENT integration HEAD (the invariant).
            self.assertTrue(CLI._is_ancestor(repo, integration,
                                             lane["feat_branch"]),
                            "%s must branch off integration" % feat)
            # fake loop: append this lane's distinct line to the SHARED file, commit
            # on the feat branch in the lane worktree.
            wt = reg["lanes"][feat]["worktree"]
            sp = os.path.join(wt, shared_rel)
            with open(sp, "a", encoding="utf-8") as f:
                f.write("touched_by = %r\n" % feat)
            git(wt, "add", "-A")
            git(wt, "commit", "-q", "-m", "lane %s" % feat)
            # merge onto integration through the real serial integrator (gates off:
            # a hermetic repo has no unittest/eval suite — the merge floor is what we
            # are proving here, not the gate chain, which Task 6 covers).
            mstatus = CLI._graph_merge_lane(reg, feat, repo, kb, integration,
                                            gates=[])
            self.assertEqual(mstatus, "merged", "%s must merge clean" % feat)

        # integration now carries ALL three distinct same-file edits — proof the
        # serialized clique merged with no textual conflict.
        out = subprocess.run(
            ["git", "-C", repo, "show", "%s:%s" % (integration, shared_rel)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        body = out.stdout
        for feat in clique:
            self.assertIn("touched_by = %r" % feat, body)
        self.assertNotIn("<<<<<<<", body)   # no conflict markers ever landed


if __name__ == "__main__":
    unittest.main()
