# H43 I2 — Native codebook writer and integrated graph gate

Status: three-pass self-review `LGTM`; implementation authorized
Date: 2026-07-30
Machine scope: CT13 GPU0 for isolated CUDA work. CT14 is used only to stop and
restore the accepted rank coherently with CT13. CT15, CT16, and SecurityLLMs
are excluded.

## Pinned evidence and hypothesis

- SGLang starts from H41 I1 closeout head
  `8abe8aa2f1205fb4d18ad09fbdf180c4cdd91723`. That commit records a rejected
  no-codebook performance result; the underlying accepted W2 implementation,
  lifecycle primitive, harness, and evidence tooling remain the source base.
- TokenSpeed starts from accepted H43 reader source `1eff3d1cb9b0612ce12cf5eedf693434cf8c25ae`;
  its evidence closeout is `08b357e85e7e27e7cbe24151088f420a94ba0db3`.
- The accepted codebook lifecycle primitive `2c1c73fd5f887f0002e93c4b330032ed60c83a25`
  is already an ancestor of the SGLang base. It is verified, not cherry-picked
  again.
- The immutable runtime base remains image ID
  `sha256:cc6fa1f338da513f9da20ee9310ba8dafbc964e2d9a6b7d528bdef9500c1920c`.
- Selected layers remain N14 global layers `24-37`; the layer set, q5 shape,
  splits, and memory geometry may not be tuned from I2 timing.

H41 I1 failed historical-long with an observed upper result of 33.820196 us
per selected layer against 27.697300. H43's sealed codebook-reader decision
measured a one-sided 95% recovery lower bound of 7.992357 us/layer at long
context and 2.082574 at actual 10K. Reserving the frozen 0.400000-us writer
placeholder gives conservative projected integrated upper values of
26.227839 and 11.580370 us/layer respectively. The hypothesis is that the
native codebook writer plus accepted post-wait reader therefore passes both
complete dependency-carrying graph gates without a dense/decoded shadow.

This is a prediction only. Reader and writer effects need not be additive;
the integrated gate is the authority.

## Exact source change

1. Extend the default-off H41 SM100 front end with an optional contiguous
   `(pool_rows, 1, 16)` uint8 codebook destination. The no-codebook
   specialization must remain compile-time separate.
2. Derive all 16 raw E4M3 bytes from the just-rounded BF16 scale and the
   storage-order E2M1 decode centroids inside the existing writer launch. Use
   CUDA SATFINITE E4M3 conversion. Do not add a full-row local array, global
   workspace, allocation, dense/FP8 latent shadow, or extra launch.
3. Validate the optional output's device, dtype, shape, contiguity, 16-byte
   alignment, pool-size agreement, and non-aliasing before launch. Invalid
   locations must retain the existing process-lifetime sticky status and must
   not form any destination pointer.
4. Enable the already accepted optional lifecycle only through the I2
   test/harness constructor's explicit `enable_fp8_codebook=True`. Do not
   change `should_allocate_mla_tq_fp8_codebook` or add a serving CLI in I2;
   default E2M1 serving remains codebook-free. Prove allocation/accounting,
   move/reuse, CPU/HiCache copy and restore, clear/free behavior, and
   mixed-layer `None` slots for N14. A separately reviewed endpoint stage owns
   any future default-off serving opt-in after I2 passes.
5. Bind the integrated harness to the H43 post-wait TokenSpeed reader and its
   immutable AOT artifacts. Runtime CuTe compilation or dispatch-key drift is
   forbidden in qualification and decision processes. The timed graph passes
   the non-`None` per-layer codebook buffer to the reader; candidate selected
   layers may not reconstruct or retain a dense/decoded shadow.
6. Bump the Torch-extension module identity and freeze its source, compiler
   command, loaded `.so`, and read-only build-cache digests. Qualification and
   decision processes may load only that prebuilt extension; they may not run
   Ninja, NVCC, or another runtime build.

Accepted source changes are committed with DCO signoff before the next
hypothesis. Rejected source is restored from the last accepted commit or a Git
revert; it is never reconstructed from memory.

## Correctness and ordering gates

The writer must preserve all existing H41 outputs and additionally byte-match
an independent CPU-FP32 reference for the 16-byte codebook after the BF16 scale
rounding point. Cover q1/q5, graph rows 1, 2, 5, 7, 10, 15, 20, 25, 30, 35,
and 40; random, zero, impulse, repeated, and realistic inputs; shuffled,
partial-page, min/max, noncontiguous, and benign identical duplicate
locations; static and explicit rotation; and 1/2/4/8-warp compile variants.
The accepted eight-warp static-rotation path remains the timing variant.

A direct device conversion probe must cover FP8 subnormals, positive and
negative zero, normal boundaries, finite saturation, infinity, and NaN inputs
against `cvt.rn.satfinite.e4m3x2.f32`. Finite end-to-end writer cases must
exercise scale values around the subnormal and saturation boundaries after
BF16 rounding. Non-finite whole-writer inputs are not a meaningful cache
contract and are not used to infer writer correctness. Existing packed,
BF16-scale, FP8-RoPE, FP8-query, canary, sticky-status, alias, stream, and
CUDA-graph gates remain.

For producer/consumer ordering, run a positive sensitivity test without a host
synchronization between writer and reader. A changing native writer updates
physical rows selected through a randomized page table; the immediately
PDL-launched q1/q5 reader output is copied into per-step device history. The
pre-move TokenSpeed reference must exhibit at least one stale-step mismatch in
1,000 alternating steps, otherwise the test is not sensitive and I2 stops.
The H43 post-wait reader must produce zero mismatches across the same steps.
A PDL-disabled same-stream control must also produce zero mismatches. Static
source-order checks remain mandatory; sanitizer is hygiene, not stale-data
proof.

Run candidate-only Compute Sanitizer memcheck and initcheck with nonzero error
exit codes, plus the H43 targeted zero-hazard racecheck. NCU must show zero
local loads/stores and zero stack/local bytes for the timed eight-warp writer,
no shared-memory growth, at most 48 registers/thread, and at least the W2
resident-block class. Record the codebook/no-codebook writer delta, but do not
substitute it for the integrated result.

Before timing, rerun H40's 30 reader tests, all H41 byte-exact cases extended
with codebook bytes, q1/q5 writer-to-reader dense-oracle checks at contexts
10,219 and 37,932, q128 and boundary-straddling chunked-prefill codebook writes,
graph replay allocation stability, lifecycle tests, and the positive PDL
ordering test. The q1/q5 reader outputs must match the same codebook-free dense
oracle used by H41, under its frozen exact/atol rules. Any failure blocks timing.

## Memory contract

At 256,000 rows and N14:

- dense FP8 control cache: 8,994,816,000 B/rank;
- integrated codebook cache: 8,141,824,000 B/rank;
- persistent saving: 852,992,000 B/rank = 0.794410706 GiB/rank;
- accepted 4,096-row writer workspace: 16,793,600 B/rank;
- projected net saving: 836,198,400 B/rank = 0.778770447 GiB/rank;
- dual-node TP8 projected net saving: 6.230163574 GiB.

The codebook path retains 93.7008% of H41 I1's no-codebook gross saving but is
below the older 10% gross target-KV bar. I2 reports this explicitly; it may not
claim 10% saving, tune to N15, or hide host/offload mirrors. Any future layer
count change is a separate quality/memory hypothesis.

## Integrated qualification and decision

Both arms reside simultaneously: 61 dense control layers and a candidate with
24 dense, 14 TQ-codebook, then 23 dense layers. Total persistent co-residency
is 17,136,640,000 B before workspaces. The query, page table, packed values,
RoPE values, and logical dependency chain are content-matched. Every selected
writer slot is a physical row dereferenced by that layer's reader. Graph replay
must allocate zero bytes and sticky status must remain zero.

Qualification requires independently resolved SGLang and TokenSpeed commit
IDs and dirty-state checks; complete source manifests for both trees; image,
Torch-extension source/command/`.so`/read-only-cache, and TokenSpeed AOT
digests; q1/q5 eager and graph correctness; raw codebook bytes; writer NCU;
H43 reader NCU; memcheck,
initcheck, targeted racecheck, lifecycle, PDL sensitivity, cache persistence,
empty-compute-process checks, P0/1965-MHz steady-state samples, clean ECC,
recovery/fabric health, and a whole-window Xid scan. Any non-performance
failure is `NO_DECISION` and authorizes no timing inference.

Decision uses ten fresh processes per context. Each process captures both full
graphs after at least 100 warmups, brackets its paired block with excluded
100-replay dense sentinels, and records 20 balanced AB/BA pairs of 100 replays.
Sequence parity reverses arm order; the designated sequence also reverses full
allocation order. Transient pair telemetry is excluded under the H43 rules;
hard health, identity, correctness, source, cache, sentinel drift, inventory,
or retained-pair failure invalidates the campaign. The process is the
resampling unit. Use 50,000 deterministic process-bootstrap draws and a
one-sided 95% upper bound for:

`(mixed_codebook_graph_us - dense61_graph_us) / 14`.

Both bounds must pass without reinterpretation:

- context 10,219: upper bound <= 27.029200 us/selected layer;
- context 37,932: upper bound <= 27.697300 us/selected layer.

The decision window is single-shot. No extra samples, seed changes, split
changes, layer changes, replay-only retry, or post-result exclusion is allowed.
At most one isolated qualification campaign, one qualification-only
infrastructure retry, and one decision campaign are authorized. Before either
accepted service outage, measured dry-run durations must establish explicit
per-command timeouts and a fail-safe restore deadline; otherwise the outage is
not started. A decision timeout or non-infrastructure qualification failure is
final for I2 and cannot be converted into an infrastructure retry.
Exact accepted CT13+CT14 containers are snapshotted, stopped together,
fail-safe restored, validated with health/model/completion/GPU/Xid checks, and
all evidence—including immutable inputs, decision contract, raw records,
analyzer outputs, health logs, and exact replay instructions—is copied outside
CT13 with a sorted SHA-256 manifest before a result is accepted.

## Advancement and stop rules

An integrated `PASS` authorizes only an immutable, default-off dual-node
endpoint candidate and the already frozen exact-10K/long endpoint, DFlash,
memory, TTFT/TPOT, and deterministic semantic/retrieval gates. Endpoint mean
TPOT may be at most 5% slower and TTFT at most 10% slower than the matched
non-TQ control; intervals crossing a boundary are indeterminate. It does not
authorize production promotion, default-on behavior, H100/SecurityLLMs use,
or an upstream contribution.

An integrated performance miss rejects the direct codebook path. A
correctness, ordering, lifecycle, resource, or memory failure blocks it. H44
ordered prefetch remains a separate future hypothesis only if its own resource
plan is reviewed; it is not substituted into I2.

## Review record

The prior Claude Opus 5 session reached its mandatory five-round cap and is not
reinvoked. Iterative self-review must reach literal `LGTM` before source work
and again before any qualification or decision window.

- Pass 1 found and resolved four gaps: whole-writer non-finite semantics,
  two-repository/compiled-artifact provenance, chunked-prefill coverage, and
  outage/campaign bounds. Verdict: `CHANGES_REQUIRED`; continue review.
- Pass 2 found and resolved an opt-in scope ambiguity and corrected the source
  base wording so a rejected result is not confused with rejected source.
  Verdict: `CHANGES_REQUIRED`; continue review.
- Pass 3 independently rechecked source ancestry, arithmetic, memory and
  performance contracts, correctness/ordering sensitivity, statistical
  decision rules, rollback, evidence custody, and the CT13+CT14-only machine
  boundary. No unresolved issue remains. Verdict: `LGTM`; implementation is
  authorized.

Implementation self-review:

- Pass 1 isolated the PDL sensitivity variable to mutable cache state, made
  sealed H43 AOT loading mandatory for reader-bearing tests, and introduced a
  digest-bound prebuilt native extension path. Verdict: `CHANGES_REQUIRED`.
- Pass 2 removed obsolete H41 timing logic, froze 20 x 100 paired execution and
  allocation reversal in the producer, hardened prebuilt path/digest checks,
  and compiled the extension successfully against the accepted image while
  the endpoint remained healthy. Verdict: `CHANGES_REQUIRED`.
- Pass 3 removed a stale analyzer arm reference, made telemetry validity a
  recomputation from raw samples, froze exact evidence inventory, and passed
  Python compilation, diff hygiene, and Black checks. Verdict: source-level
  `LGTM`; GPU qualification and performance acceptance remain pending.
