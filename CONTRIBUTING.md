# Contributing to Omniva's SGLang fork

This is Omniva's production fork of [sgl-project/sglang](https://github.com/sgl-project/sglang), maintained by the AI & Cloud Security team. It carries production-validated modifications that are optimized for our hardware (currently 8×H100 on the SecurityLLMs cluster; potentially multi-node in the future) and our model deployments (currently Kimi K2.6 with TurboQuant-MLA KV compression; more to follow).

The long-term goal: become a drop-in optimized SGLang that any Omniva inference deployment can use, with the research-grade improvements that matter for our workloads folded in. This file documents how to contribute to that cleanly.

## Branch layout

| Branch | Purpose | Push discipline |
|---|---|---|
| `main` | **Read-only mirror of upstream `sgl-project/sglang:main`.** Never commit here directly. Fast-forward-only updates from upstream. | Periodic `git fetch upstream && git push origin main` to refresh. |
| `omniva-main` | **Our production trunk.** Only clean, validated, shippable commits. This is what deployments ship from. | Protected. Merge from feature branches only after bench + coherence pass. No force-push. |
| `feat/<name>` | One per experiment or feature. Branched off `omniva-main`. | Force-push allowed while WIP. Delete the branch after merge. |
| `exp/<name>` | Pure experimentation not expected to merge (spike, what-if, perf probe). | Delete after the experiment concludes — success or failure. |
| `archive/<name>` | Tags, not branches. Preserve reachability of commits that got rebased out or experiments we want to keep accessible without polluting the branch list. | Never deleted. |

### Rule: failed experiments never land on `omniva-main`

Stage B-1 (scatter-dequant, 2026-04-26) was a legitimate experiment that didn't pay off. It lived on `kimi-k26-turboquant-mla` as ~200 lines of gated-off dead code for a day before getting rebased out when we restructured to this convention. The lesson: **gated-off code in the trunk is dead code.** If an experiment fails, delete its branch and the corresponding overlay; if you want the learning preserved, write a log entry in the research notes (vault) — not a code commit.

## Starting a feature branch

```sh
git fetch origin
git checkout -b feat/my-feature origin/omniva-main
# ... work, commit, push ...
git push -u origin feat/my-feature
```

Keep the branch focused: one experiment, one concern. Don't mix "refactor tests" with "add new kernel" — separate branches.

## Merging back to `omniva-main`

Before merging a feature branch into `omniva-main`:

1. **Bench pass**: the measured metric the branch was supposed to improve (TPOT, admission cap, cap-at-quality-floor) has to have moved in the predicted direction by at least the predicted margin. If not, the branch gets deleted, not merged.
2. **Coherence check**: smoke test with a known-good prompt (e.g. "write a factorial function in Python") producing sensible output. No gibberish. Storage round-trip passing is *not* sufficient — see Research Rules in the vault (`OmniSec/Inference/Research/Rules and principles/Research Rules.md`).
3. **Commit hygiene**: commits on the branch describe *why*, not just *what*. Example good subject: `fix(mla-tq): skip o_proj rotation fusion on MLA path` — followed by a body explaining the root cause and the mechanism of the fix. Bad subject: `updated model_runner.py`.
4. **Rebase onto latest `omniva-main`** before merging to keep linear history.

```sh
git checkout feat/my-feature
git fetch origin
git rebase origin/omniva-main
# resolve conflicts if any, verify the bench still passes
git push --force-with-lease origin feat/my-feature
# then merge (fast-forward) into omniva-main
git checkout omniva-main
git merge --ff-only feat/my-feature
git push origin omniva-main
git branch -d feat/my-feature
git push origin --delete feat/my-feature
```

## Picking up upstream changes

Upstream SGLang moves fast. To pick up new upstream features onto our trunk:

```sh
# refresh the upstream mirror on our `main`
git fetch upstream
git checkout main
git merge --ff-only upstream/main
git push origin main

# merge the freshly-synced main INTO omniva-main (not rebase — see below)
git checkout omniva-main
git merge main
# resolve any conflicts manually (never use -X ours / -X theirs — auto-side-
# picking silently drops the other half of a real conflict). Re-run the
# regression gate (tests + a short live boot smoke-test) before pushing.
git push origin omniva-main
```

**Merge, not rebase, for `omniva-main`.** We deliberately chose merge over rebase here because `omniva-main` is a long-lived published integrator branch with downstream consumers (the `ai-node-setup` overlay's ConfigMap paths, other developers' clones). Rebasing would rewrite its commit SHAs, force a `--force-with-lease` push, and turn every downstream clone into a required force-pull. A single merge commit per upstream sync preserves everyone's SHAs and makes any individual sync atomically revertible. The cost is a non-linear graph, which is a small price for safety. For short-lived private feature branches, rebase is still fine (see "Merging back to `omniva-main`" above).

If an upstream PR is still open and we've vendored it as a seed (the PR #23135 / TurboQuant story is the canonical example), carry a merge commit at the top of the chain that identifies the PR number — when the PR eventually merges upstream, that marker makes the "drop our vendor copy" step much easier.

## Upstream tracking policy: nightly, not stable

Our `main` mirrors upstream `sgl-project/sglang:main` (the nightly branch), **not** the latest stable tag. This is a deliberate strategic choice, reviewed and re-affirmed on 2026-04-30. The rationale below exists so the next time someone asks "shouldn't we be on stable?" they have the answer.

**Why not stable:**
- SGLang's release cadence is roughly one stable tag every 3–6 weeks, and as of 2026-04-30 the most recent stable (`v0.5.6.post2`, 2025-12-11) was **5 months behind `upstream/main` by ~4500 commits**. "Latest stable" in this project often means "very stale."
- Our proprietary work depends on upstream features that do not land in stable until months later (TurboQuant KV compression PR #23135 is the canonical example — unmerged as of 2026-04-30, but the foundation of both our Kimi MLA-TQ work and our gpt-oss MXFP4 work). A stable-tracking fork would mean vendoring every such feature ourselves, which is strictly more fragile than tracking the nightly that already contains them.
- We run regression gates before every production deploy (kernel unit tests, live correctness gates, spot-check benchmarks). Nightly instability that affects our models surfaces in the gate, not in production.

**Why nightly works for us:**
- Small, frequent syncs are cheaper than rare, large ones. Our empirical data point: on 2026-04-30 we synced 230 upstream commits spanning ~5 weeks of nightly, and got a **zero-conflict git-level auto-merge** on the 7 files our proprietary code touches. Auto-merge stayed semantically safe because we review the merged files and re-run tests before pushing.
- Our image tag pins to a specific nightly digest (`lmsysorg/sglang:nightly-dev-<YYYYMMDD>-<sha>`), not `:latest`. That means even when we sync, production only moves when we deliberately re-roll the overlay manifest onto a new image tag.
- Upstream bug fixes reach us faster.

**Sync cadence (the commitment we're making to ourselves):**
- **Target: weekly.** Measured: ~40–50 upstream commits land per week at the current upstream pace. That's a reviewable batch.
- Trigger: pick a fixed day (default Monday) or opportunistically at the start of any new milestone. Either way, don't let `main` go more than 2 weeks behind without explicit sync.
- Each sync: fast-forward `main` to a specific pinned upstream SHA (not a branch tip — the tip moves between fetch and merge). Verify `main` is a clean ancestor (fast-forward possible) before touching `omniva-main`. Then merge `main` into `omniva-main`, run the regression gate, push.

**Triggers that would prompt a re-evaluation and possible switch to stable:**
- Upstream merges TurboQuant (#23135) and any other unmerged features we vendor, AND a stable release includes them.
- We enter maintenance mode (no new proprietary features, just bug-fix and dependency updates).
- Nightly stability degrades to the point where regression gates routinely fail on syncs unrelated to our work.

None of these are true as of 2026-04-30.

## Commits

- **Commit messages describe why.** One-line subject in the imperative (`feat(area): do X`, `fix(area): correct Y`). Body explains the reason, the evidence, and any gotchas a future reader would need.
- **No AI attribution.** Per Omniva's working agreement (see vault `CLAUDE.md` files), nothing in commit messages, MR descriptions, or PR bodies references Claude, AI assistants, or co-authored-by.
- **No backports of corporate context** unless the commit absolutely needs it. Commits should read as stand-alone engineering work — a future contributor from elsewhere in Omniva (or the open-source community, if the commit gets upstreamed) shouldn't need internal-doc pointers to understand the change.

## What's currently on `omniva-main`

For the exact list, run `git log upstream/main..omniva-main --oneline`. As of 2026-04-30 (M1+M2 merged), omniva-main has roughly 48 proprietary commits grouped into three arcs:

1. **PR #23135 vendor (TurboQuant MHA KV compression)** — one merge commit at `925138ada` pulling in the unmerged upstream PR. Foundation for everything below.
2. **Kimi K2.6 MLA-TurboQuant work** — ~18 commits under `feat(mla-tq):`, `fix(mla-tq):`, `test(mla-tq):`, `bench(mla-tq):`. Adds `MLATokenToKVPoolTurboQuant`, MLA dispatch wiring, fused Triton MLA decode kernel (Stage C), numerical-parity tests.
3. **gpt-oss-120b single-GPU TP=1 arc (M1+M2)** — ~27 commits from `feat/gpt-oss-tq-1gpu`, merged as one `--no-ff` unit. Adds the Omniva MXFP4 MoE runner kernel + tile configs + concurrent-dispatch tests.

Arcs are **listed in merge order, which is also the dependency order** — the gpt-oss work sits on top of a TurboQuant-integrated base. When upstream eventually merges PR #23135, the merge commit at `925138ada` is the marker we'd use to identify "this chunk can be dropped; upstream now has it."

Active deployments in `ai-node-setup` pin to specific omniva-main SHAs (not tip) via ConfigMap overlay paths. Whenever omniva-main moves, those pins stay on the old SHA until the overlay manifest is deliberately updated. This means `omniva-main` advancing does not affect any running pod until a deploy is triggered.

## Related documentation

- **Deployment**: [ai-node-setup repo](https://gitlab.aws.omniva.com/acls/security-analytics/ai-node-setup), particularly `kubernetes/ai/apps/sglang/` for the production Kimi K2.6 manifest and overlay ConfigMap patches.
- **Research notes**: Omniva vault `OmniSec/Inference/` — integration logs, performance analysis, research rules.
- **Ops runbook**: vault `Omniva/Ops/K8s.md` for cluster + deployment operations.
