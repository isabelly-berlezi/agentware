# Release signing — go-live runbook (agentware P2b dedicated updater)

> The dedicated updater applies **only a signed SemVer tag `vX.Y.Z`** verified
> against a HOME-pinned publisher key — never `@{upstream}`/tip-of-main. This
> runbook turns the previously out-of-band signing bootstrap into a concrete,
> checkable procedure.
>
> **Autonomous vs operator-gated (R-GIT-01).** The agentware loop BUILDS and VERIFIES
> the machinery (verify path, trust-anchor pinning, `release preflight`), but it
> **MUST NOT** generate the production signing key and **MUST NOT** create or push a
> real tag on the real package remote or the operator KB remote. Those are
> **operator actions**, performed via this runbook post-merge. Until go-live, the
> updater safely **no-ops (fail-closed)** — it verifies-and-declines, never "applies
> anyway".

---

## Trust model (who signs, who verifies)

| Side | Holds | Action |
|------|-------|--------|
| **Publisher** (release owner) | the **private** SSH signing key (offline custody) | signs annotated tags `vX.Y.Z` |
| **Client** (every machine) | the publisher's **public** key, pinned in `~/.agentware/allowed_signers` (OUTSIDE the tree the updater rewrites) | verifies the signed tag before applying |

No keyring, no network, no `gh` — verification uses plain git/SSH
(`gpg.ssh.allowedSignersFile`). The trust anchor lives in HOME so a malicious push
can never rotate the key it is checked against.

---

## Publisher: one-time signing bootstrap (OPERATOR-GATED)

1. **Generate an SSH signing key** (Ed25519) and **custody the private key OFFLINE**
   (a hardware token or an offline machine — it is the fleet's root of trust):
   ```bash
   ssh-keygen -t ed25519 -C "agentware-release" -f ~/.ssh/agentware-release
   # keep ~/.ssh/agentware-release (PRIVATE) offline; distribute only the .pub
   ```
2. **Configure git to SSH-sign tags** (in the package repo):
   ```bash
   git config gpg.format ssh
   git config user.signingkey ~/.ssh/agentware-release.pub
   git config tag.gpgsign true
   ```
3. **Publish the PUBLIC key for client discovery** — commit it in-tree at
   `catalog/agentware-release.pub` (format: `ssh-ed25519 <key-data> agentware-release`).
   Clients pin it via `scripts/agentware trust pin --from-repo` (see below). The
   in-tree copy is only for discovery; the HOME-pinned copy is the trust anchor,
   and `trust verify` warns loudly if they ever diverge.

## Publisher: cut the first (and every) signed release

1. **Gate FIRST** — never sign a release that has not passed the full gate chain:
   ```bash
   scripts/agentware gate release --full     # content-preservation + gold retrieval + reliability + SWE
   ```
   A SWE pass-rate regression STOPS and signals a pivot — do NOT tag.
2. **Cut + push the signed tag** (SemVer stable only — `vMAJOR.MINOR.PATCH`, no
   pre-release/build):
   ```bash
   git tag -s -m "agentware v2.0.0" v2.0.0
   git push origin v2.0.0
   ```
   The tagger identity (`user.email`) is the **principal** clients match against —
   keep it stable and pin the key under that principal.

## Client: pin the trust anchor (onboarding / first clone)

```bash
scripts/agentware trust pin --from-repo      # pins catalog/agentware-release.pub
scripts/agentware trust list                 # confirm the fingerprint OUT OF BAND
```
Verify the printed `SHA256:` fingerprint against the publisher's out-of-band value
before trusting it. Then opt in to automatic updates:
```bash
scripts/agentware config --set-updater on
scripts/agentware tick --install-schedule
```

---

## Key rotation (overlap window — never strands the fleet)

`allowed_signers` is a **multi-key** file honoring SSH `valid-after`/`valid-before`.
To rotate without stranding machines mid-flight:

1. Publisher generates a NEW key and publishes its public half.
2. During the overlap window, clients trust BOTH (old + new):
   ```bash
   # trust the NEW key with a valid-after; keep the OLD one until it retires
   scripts/agentware trust rotate --key-file new-release.pub --valid-after 20260101
   scripts/agentware trust list        # both keys present -> fleet never stranded
   ```
   Confirm the new fingerprint out of band (`trust rotate` prints the reminder).
3. Publisher signs new releases with the NEW key; old releases stay verifiable.
4. After the window, retire the OLD key (add a `valid-before`, or remove its line by
   editing `~/.agentware/allowed_signers` — the only supported manual edit).

## Key revocation (compromise)

If a signing key is compromised: stop signing with it immediately, cut an
out-of-band advisory, and have every client **remove the compromised key's line**
from `~/.agentware/allowed_signers` (or set a `valid-before` in the past). Because
no-downgrade refuses any version ≤ `last_applied_version`, an attacker cannot
re-serve an old vulnerable release; but a compromised *current* key can sign a
malicious higher tag, so revocation must be prompt and out-of-band.

**Residual (accepted, v1):** signed tags give integrity + no-downgrade but NOT
content-freshness — a mirror/MITM that succeeds at `git fetch --tags` while
*withholding* a newer signed security tag is undetected here. Cross-machine
peer-freshness detection is deferred to team-mode.

---

## Verify readiness before go-live

```bash
scripts/agentware release preflight
```
Exits 0 when the **hermetic verify machinery** is sound (it signs, pins, verifies,
and resolves a throwaway-key tag end-to-end — touching no real key/tag/remote) and
reports `publisher_ready` / `client_ready` / `go_live_ready` so you can complete the
operator-gated steps. `go_live_ready=true` means: machinery OK **and** publisher
signing config present **and** a client key pinned.
