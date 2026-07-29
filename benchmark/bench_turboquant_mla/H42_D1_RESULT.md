# H42 D1 long-dependency localization result

Status: `VALID`; all predeclared mechanisms rejected as insufficient
Date: 2026-07-29
Timing source: SGLang `c1d1f9122`, TokenSpeed `a9af4c57`
Machine: CT13 GPU0 only; CT14 ran no experimental process

## Result

The five fresh-process historical-long arms passed the frozen validity gates.
`F-chain` reproduced the earlier H41 I1 failure at `33.099257 us` per selected
layer, inside the required `30.274-37.202 us` envelope.  The measured component
differences were:

| Diagnostic contribution | Delta per selected layer |
|---|---:|
| Surrounding 47 dense layers and transitions | `-0.091441 us` |
| Query pointer/value/buffer negative control | `-0.049546 us` |
| Interleaved versus illegal phased order | `+0.571127 us` |
| Reader PDL on versus off | `-0.649243 us` |

None reaches the frozen `6.122896 us/layer` minimum recovery.  Therefore D1
rejects selected-block query topology, operation interleaving, reader PDL, and
the surrounding dense layers/transitions as standalone explanations or source
hypotheses.  This is a diagnostic rejection, not an endpoint or quality result.

All arms retained exactly `910,336,000` bytes (`0.847816467 GiB`) gross
persistent saving per rank with no dense shadow or codebook.  Replay allocation
was unchanged at `17,282,886,656` bytes, maximum control-flank drift was
`0.072643%`, and maximum writer/reader error was `0.000152588` against the
`0.002` limit.

## Physical trace

The excluded Nsight Systems runs validate the graph construction.  `F-chain`
contains 244 graph nodes per arm; every selected-only graph contains 56.  Every
control node executed 12 times and every candidate node 11 times.  The
interleaved variants repeat frontend/reader four-node groups; `S-phased` places
all 28 frontend nodes before all 28 reader nodes.  Instrumented durations are
not used as timing evidence.  `analyze_h42_nsys.py` reproduces these checks from
the archived SQLite exports.

## Cross-experiment synthesis and corrected attribution

The failed localization triggered a source-contract audit rather than another
dependency experiment.  That audit found that H41 W1 and H41 I1/H42 did not
measure the same TQ reader representation:

- H41 W1 passed a materialized FP8 codebook shaped `[..., 16]` to every selected
  cache row and counted `57,344,000` codebook bytes for N14.  Its kernel uses
  `lookup_tq4_word_from_fp8_codebook_prmt`.
- H41 I1 and H42 deliberately passed `kv_nope_codebook=None`.  Their kernel uses
  `dequantize_tq4_word_to_fp8_shfl`, loading the BF16 scale and reconstructing
  FP8 values from E2M1 centroids in the reader.

The earlier arithmetic that treated W1's approximately `14.19 us/layer`
reader delta as the isolated estimate for the no-codebook I1 reader was thus
invalid.  The apparent approximately `15.4 us/layer` "dependency interaction"
is plausibly dominated by this unpaired reader fast-path difference.  H42 D1
does not itself prove the codebook benefit because all five candidate arms used
the no-codebook path.

This is also a memory tradeoff, not a return to a dense shadow.  For N14 at
256,000 rows, adding the 16-byte codebook costs `57,344,000` bytes per rank and
leaves `852,992,000` bytes (`0.794410706 GiB`) saved per rank: `93.70%` of the
no-codebook gross saving.  Across eight TP ranks that is approximately
`6.355286 GiB` retained saving instead of `6.782532 GiB`.  The codebook values
are a derived FP8 lookup table for the same E2M1 codes/scales; its numerical and
quality equivalence still requires an explicit gate.

## Decision and reflection

Do not implement a PDL-only or reorder-only change from H42.  Freeze a new,
same-source codebook/no-codebook reader A/B before changing the writer.  If its
conservative improvement exceeds `6.122896 us/layer` at historical-long, add an
optional codebook destination to the native SM100 frontend, prove byte-exact
current-row writes plus eager/graph/sanitizer safety, and repeat the complete
61-layer integrated gate at both contexts.

The useful failure here is methodological: component estimates are composable
only when representation and optional kernel inputs match exactly.  Future
analyzers must include codebook presence, byte count, and reader specialization
in their cross-experiment compatibility checks.

## Operational closeout

Both accepted N2 DFlash ranks were stopped together and restored by restarting
their original containers, CT14 rank 1 before CT13 rank 0.  Both health checks
returned HTTP 200; CT13 model information identified Kimi K2.6; an independent
request completed exactly 64 generated tokens.  All eight GPUs were P0 at
`1965/1965 MHz`, volatile and aggregate uncorrectable ECC were zero, recovery
action was `None`, fabric state was `Completed/Success`, and no Xid/fabric error
appeared in the maintenance window.

The timing, smoke, pre-maintenance, Nsight, and source artifacts are archived
off CT13 under the vault H42 evidence directory with a verified sorted SHA-256
manifest.  The first trace smoke with the invalid container-hostname assertion
and the first `nsys stats` invocation using unsupported `--option=value` syntax
remain excluded infrastructure evidence; neither entered timing.

## Result self-review

- Round 1 recomputed every component from the five raw C/T/C JSON files and
  confirmed the frozen F-chain replication, drift, correctness, allocation,
  and memory gates.  No eligible D1 mechanism exists.
- Round 2 queried every Nsight SQLite export directly and found exact node
  counts, replay counts, and declared order.  It also confirmed that profiling
  overhead changes absolute durations, so only structure is claimed.
- Round 3 audited the W1 and I1 function arguments and kernel specialization.
  It found the materialized-codebook mismatch and withdrew the prior additive
  dependency attribution.  The corrected result and next experiment are now
  bounded by that mismatch.  LGTM for closing H42 D1 and planning a controlled
  codebook A/B; no implementation, endpoint, production, or upstream approval.
