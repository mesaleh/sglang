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
2. **Coherence check**: smoke test with a known-good prompt (e.g. "write a factorial function in Python") producing sensible output. No gibberish. Storage round-trip passing is *not* sufficient — see [Research Rules §6](../Omniva/OmniSec/Inference/Research%20Rules.md).
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

# rebase omniva-main onto the new upstream
git checkout omniva-main
git rebase upstream/main    # or git rebase main, equivalent at this point
# resolve conflicts, re-bench end-to-end
git push --force-with-lease origin omniva-main
```

Rebasing `omniva-main` is **destructive to downstream clones** — anyone who pulled the old version has to force-pull. Communicate before doing it.

If an upstream PR is still open and we've taken it as a seed (see [Kimi K2.6 TurboQuant-MLA log](../Omniva/OmniSec/Inference/Performance%20Optimization/Kimi%20K2.6%20TurboQuant-MLA%20Integration%20Log%202026-04-26.md) for the PR #23135 story), keep a merge commit at the top of the chain that identifies the PR — makes rebases after the upstream PR merges significantly easier.

## Commits

- **Commit messages describe why.** One-line subject in the imperative (`feat(area): do X`, `fix(area): correct Y`). Body explains the reason, the evidence, and any gotchas a future reader would need.
- **No AI attribution.** Per Omniva's working agreement (see vault `CLAUDE.md` files), nothing in commit messages, MR descriptions, or PR bodies references Claude, AI assistants, or co-authored-by.
- **No backports of corporate context** unless the commit absolutely needs it. Commits should read as stand-alone engineering work — a future contributor from elsewhere in Omniva (or the open-source community, if the commit gets upstreamed) shouldn't need internal-doc pointers to understand the change.

## What's currently on `omniva-main`

As of 2026-04-26, `omniva-main` has 9 commits on top of upstream `sgl-project/sglang:main`:

1. Merge of upstream PR #23135 (TurboQuant KV cache compression, MHA-only base)
2. `bab1d4488` safety patch: widen `_maybe_fuse_tq_output_rotation` skip clause to cover non-real-float quantization artifact dtypes
3. `c63114715` `feat(mla): add MLATokenToKVPoolTurboQuant pool class`
4. `dd8d59c6d` `fix(mla-tq): unconditional bf16 cast for nope input in set_mla_kv_buffer`
5. `d8c659d28` `feat(mla-tq): wire MLATokenToKVPoolTurboQuant into dispatch & sizing`
6. `34e26e4c8` `fix(mla-tq): dequant_scale zero-guard must use where, not maximum`
7. `7dcda7dc1` `fix(mla-tq): do not force triton decode backend on MLA path`
8. `21ad033e1` `fix(mla-tq): skip o_proj rotation fusion on MLA path`

Commits 7 and 8 were proposed upstream as review comments on PR #23135 on 2026-04-26. If the author folds them in, we drop them from `omniva-main` on the next upstream rebase.

## Related documentation

- **Deployment**: [ai-node-setup repo](https://gitlab.aws.omniva.com/acls/security-analytics/ai-node-setup), particularly `kubernetes/ai/apps/sglang/` for the production Kimi K2.6 manifest and overlay ConfigMap patches.
- **Research notes**: Omniva vault `OmniSec/Inference/` — integration logs, performance analysis, research rules.
- **Ops runbook**: vault `Omniva/Ops/K8s.md` for cluster + deployment operations.
