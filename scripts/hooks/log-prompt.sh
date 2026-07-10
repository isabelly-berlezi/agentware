#!/bin/bash
# UserPromptSubmit hook — durably record EVERY user prompt so nothing is ever
# lost. Appends to the EXTERNAL knowledge base (logs/prompts.log) when one is
# configured, else falls back to a repo-local log (pre-onboarding only). Writes
# nothing to stdout, so no extra context is injected.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

KDIR="$("$REPO_ROOT/scripts/aw-knowledge-dir" 2>/dev/null || true)"
if [[ -n "$KDIR" ]]; then LOG_DIR="$KDIR/logs"; else LOG_DIR="$REPO_ROOT/.agentware-logs"; fi
mkdir -p "$LOG_DIR"

input="$(cat)"
sid=""; prompt=""; cwd=""
if command -v jq >/dev/null 2>&1; then
  sid="$(printf '%s' "$input" | jq -r '.session_id // "unknown"' 2>/dev/null)"
  prompt="$(printf '%s' "$input" | jq -r '.prompt // empty' 2>/dev/null)"
  cwd="$(printf '%s' "$input" | jq -r '.cwd // empty' 2>/dev/null)"
fi

ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Provenance origin (Task 7a, 260710-learning-repair-p1): loop (agentware.sh
# exported AGENTWARE_ORIGIN) / system (machine-injection heuristic) / human
# (default operator prompt). Additive: appended to the header, existing parsers
# that read the first three bracket groups are unaffected.
origin="human"
if [[ -n "${AGENTWARE_ORIGIN:-}" ]]; then
  origin="loop"
elif printf '%s' "$prompt" | grep -qiE '<task-notification>|<system-reminder>|<persona-preamble>|^You are (a subagent|Claude Code|the agentware)'; then
  origin="system"
fi

# Turn id (Task 7b): content hash joining this record to its rendered turn in
# sessions/<sid>/main.md (render-transcript.py computes the SAME
# sha1(text.strip())[:12]). Best-effort (needs python3).
turn=""
if command -v python3 >/dev/null 2>&1; then
  turn="$(printf '%s' "$prompt" | python3 -c 'import hashlib,sys; print(hashlib.sha1(sys.stdin.read().strip().encode("utf-8")).hexdigest()[:12])' 2>/dev/null)"
fi

{
  printf '[%s] [session %s] [cwd %s] [origin=%s] [turn=%s]\n' "$ts" "$sid" "$cwd" "$origin" "$turn"
  printf '%s\n\n' "$prompt"
} >> "$LOG_DIR/prompts.log"
exit 0
