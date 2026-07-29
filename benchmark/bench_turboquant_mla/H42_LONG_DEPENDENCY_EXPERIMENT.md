# H42 long-context dependency localization

Status: frozen source-only diagnostic plan; no live run yet
SGLang base: `8abe8aa2f`
TokenSpeed base: `a9af4c57`
Machine scope: local source work plus CT13 GPU0 only for isolated profiling. Both accepted
endpoint ranks on CT13 and CT14 must be stopped and restored together. CT14 runs no experimental
process. CT15, CT16, and SecurityLLMs are excluded.

## Evidence that motivates H42

H41 I1 passed all material, safety, exact-context timing, and 0.847816467-GiB-per-rank no-shadow
memory checks. Its historical-long integrated delta was `33.638324-33.820196 us` per selected
layer, above the frozen `27.6973 us` allowance lower bound. The upper bound must improve by at
least `6.122896 us/layer` before an integrated repeat can pass.

The independent H41 measurements predict approximately `14.1948 us/layer` for the long material
reader and `4.1478 us/layer` for the complete frontend, leaving roughly `15.4 us/layer` of
context-growing interaction in the ordered graph. Source inspection identifies a concrete launch
asymmetry:

- the dense FlashInfer RoPE/FP8 frontend, SGLang scatter, TokenSpeed split kernel, and TokenSpeed
  reduction all use programmatic-dependent-launch (PDL) waits/triggers on SM100;
- the selected BF16 RoPE helper and H41 combined query/material writer use ordinary launches;
- the accepted W1 reader ring and W2 frontend ring measure like families back-to-back, whereas I1
  alternates frontend and attention and feeds every attention output to the next layer's query.

The CUDA Programming Guide says PDL can early-launch a downstream kernel in the same stream, but
the downstream kernel must synchronize before consuming primary results; overlap is opportunistic.
CUDA graph stream capture preserves this as programmatic graph-edge data. Therefore H42 measures
the physical interaction instead of assuming that isolated kernel deltas add.

## Frozen diagnostic hypothesis

At historical-long, most of the unexplained `~15.4 us/layer` is attributable to one of three
separable mechanisms:

1. the true attention-output-to-next-query dependency;
2. loss of back-to-back PDL opportunity when frontend and attention families are interleaved; or
3. the two dense/selected transitions and the surrounding 47 dense layers.

Reader split selection is not retuned in the first diagnostic. H40 already screened long split
`32/40/64/96` and selected 40, so another unrestricted sweep would duplicate evidence and create a
multiple-testing path.

## Frozen diagnostic arms

All arms use context `37,932`, q_len 5, H8, batch 1, max context 256,000, selected layers 24-37,
row size 322 bytes, split 40, eight writer warps, FP8 RoPE material, no codebook, no dense shadow,
the H41 content-matched shuffled page table, and preallocated outputs/workspaces. Control and
candidate caches remain simultaneously resident and use control-first allocation with seed
`20260729`. Each arm runs in a fresh process as a CUDA graph with at least 100 warmups and 2,000
uninstrumented replays, bracketed C/T/C. Run the frozen arm order shown below; it is operational
ordering, not a selection rule.

The diagnostic variants are:

| ID | Layers timed | Query topology | Scheduling | Attention PDL | Purpose |
|---|---:|---|---|---|---|
| `F-chain` | 61 | output chain | interleaved | on | fresh H41 I1 replication |
| `S-chain` | 14 | output chain | interleaved | on | remove surrounding dense layers/transitions |
| `S-static` | 14 | fixed independent input | interleaved | on | remove only the inter-layer query dependency |
| `S-phased` | 14 | fixed independent input | all frontends, then all readers | on | expose the isolated-ring PDL opportunity |
| `S-chain-noapdl` | 14 | output chain | interleaved | off for both readers | identify reader PDL contribution without changing frontend PDL |

`S-phased` owns distinct query and attention outputs per layer so phase separation cannot race on
shared material. It is a diagnostic lower bound, not a legal transformer schedule or deployment
candidate. `S-static` is likewise attribution-only. The control receives the same topology and
scheduling transformation as its paired candidate; no arm manufactures a win by slowing only the
dense control.

For every selected-only arm, normalize `(candidate_graph_us - control_graph_us) / 14`. Compute:

- surroundings/transition contribution: `F-chain - S-chain`;
- true query-chain contribution: `S-chain - S-static`;
- interleaving contribution: `S-static - S-phased`;
- reader-PDL sensitivity: `S-chain-noapdl - S-chain`.

The signs and magnitudes are diagnostic; they are not endpoint or acceptance results. An
optimization mechanism is eligible for implementation only if its favorable measured contribution
is at least the missing `6.122896 us/layer`, its two control flanks differ by at most 2%, and the
graph output/status and GPU covariates are valid. A true-chain contribution is evidence of a legal
scheduling constraint, not permission to remove that dependency. If contributions interact and do
not sum, the physical trace—not an additive reconstruction—determines the next hypothesis.

## Physical trace and falsifiers

One separate trace-only fresh process per variant replays briefly under Nsight Systems
`--cuda-graph-trace=node`; these runs are excluded from timing. Archive the reports plus exported
SQLite/kernel summaries. The traces must show the declared family ordering and allow the selected
14-layer block to be isolated. Duration under instrumentation is never a verdict.

Reject or repair the harness before interpretation if any of the following occurs:

- `F-chain` does not reproduce the H41 I1 long delta within the earlier observed envelope widened
  by 10% (`30.274-37.202 us/layer`);
- an attribution arm changes cache representation, selected rows, split, warps, content, current
  physical locations, or persistent byte accounting;
- control and candidate topology transformations are not paired;
- any output is non-finite, sticky status is nonzero, replay allocation changes, or the one-layer
  selected reader differs from its reconstructed dense oracle beyond `atol=0.002`;
- either control flank drifts by more than 2%, the GPU is not P0 at max SM clock, uncorrected ECC is
  nonzero, recovery action is present, or an Xid occurs in the maintenance window.

## Decision after localization

H42 D1 changes no serving or kernel source. After its evidence is archived off-node and the exact
N2 DFlash service is restored and smoke-tested, select at most one implementation mechanism:

- if lost PDL/interleaving is at least `6.122896 us/layer`, design a default-off native SM100
  frontend that preserves dependency safety while reducing ordinary launch boundaries (candidate
  options are PDL-safe launch plus fused BF16 RoPE, not a reordered transformer);
- if split scheduling is implicated after PDL attribution, freeze a small integrated split set
  before testing; H40's split-40 result remains the control;
- if the true query dependency consumes the excess and no legal overlap is available, do not claim
  the isolated W1/W2 sum as deployable. Move to a more substantial fused attention/frontend SM100
  design or reject N14 for the 5% gate.

Any selected source hypothesis gets its own committed plan, byte-exact/eager/graph/sanitizer gate,
and three fresh C/T/C decision processes at both 10,219 and 37,932. It must recover at least
`6.122896 us/layer` at the long upper bound and still satisfy the original H41 I1 allowance in both
windows. Only then may the unchanged complete 61-layer I1 gate be repeated. No image, endpoint,
quality, production, or upstream claim is authorized by D1.

## Evidence, rollback, and service discipline

Commit the frozen plan and accepted harness before stopping service. Archive commands, source
manifest, JSON results, graph metadata, Nsight report/SQLite export, GPU/fabric/Xid checks, service
fingerprints, and a sorted SHA-256 manifest outside CT13. Rejected code is quarantined on its named
pushed branch and restored deterministically with Git. Never reconstruct an accepted file from
memory.

Before experimentation, record both endpoint ranks and health, stop both together, and prove GPU0
idle before CUDA initialization. After the diagnostic, relaunch the exact accepted N2 DFlash
configuration on both nodes, require both health/model-info endpoints plus an independent 64-token
completion, and verify P0/max-clock, ECC zero, no recovery action, and no new Xid.

## Plan self-review

- Round 1 found and fixed two sources of experimental freedom: allocation order/seed are now fixed,
  and every timed arm is a fresh process rather than a warmed multi-arm process.
- Round 2 found an aliasing ambiguity in the phased lower bound and a trace/protocol mismatch. The
  phased arm now owns query and attention outputs per layer, and each trace is explicitly a separate
  excluded fresh process matching one variant.
- Round 3 checked paired control transformations, numerical gates, the no-retuning rule, machine
  restrictions, deterministic rollback, off-node evidence, and service restoration. LGTM for
  harness implementation; it does not approve a kernel change or live result.
