# H41 I1 integrated four-family graph experiment

Status: harness implemented and locally self-reviewed; live execution not started
SGLang base: `8d47c429299458564a45d427ef2a6ded29a49bf2`
TokenSpeed base: `a9af4c57f682e800c42cc53063e5f013369210a9`
Machine scope: source work is local. Final isolated correctness/timing uses CT13 GPU0 with both
accepted endpoint ranks stopped together; CT14 runs no experimental process. No other machines.

## Operational scheduling amendment

W2 self-review found that the original post-RoPE bracket omitted the candidate's accepted BF16 RoPE
launch. Correcting and independently replicating the complete path consumed the W2 timing window;
the accepted service was restored rather than left unavailable while I1 was designed. I1 therefore
uses one additional, separately capped stop/restore window of at most two hours. This changes only
outage scheduling, not a numerical boundary, variant, workload, or machine. The original ten-hour
aggregate outage cap remains binding. The immutable endpoint campaign keeps its own later window
only if I1 passes.

## Frozen hypothesis

At batch 1, q_len 5, and eight query heads, the complete real-order N14 graph fits the fresh W1 full
allowance at both contexts. The control graph contains 61 dense layers. The candidate graph contains
24 dense layers, 14 selected TQ layers (`24-37`), then 23 dense layers. Every dense layer executes:

1. FlashInfer fused RoPE plus FP8 query/KV conversion;
2. dense FP8 cache scatter; and
3. TokenSpeed dense attention including split/LSE work.

Every selected layer executes:

1. accepted BF16 query/KV RoPE into preallocated outputs;
2. the accepted eight-warp H41 post-RoPE query plus material writer; and
3. the H40 FP8-RoPE material TQ4 attention reader including split/LSE work.

The selected row remains 322 bytes and has no dense/FP8 shadow or per-row codebook. The selected
reader split is frozen to 64 at context 10,219 and 40 at context 37,932. Current-token physical slots
are derived from the same shuffled page table consumed by attention. All kernels execute in the
declared layer order on one current stream.

## Correctness and graph falsifiers

- H41 writer output cannot be consumed directly by the H40 reader in its packed/scale/FP8-RoPE
  layout for q1/q5, shuffled and partial pages, current-token writes, graph replay, or exact/long
  rolled shapes.
- A selected reader differs from a dense cache reconstructed from the exact E2M1 codes/scales/FP8
  RoPE beyond the accepted `atol=0.002`, or produces non-finite output/status.
- Any of H40's 30 material-reader contract tests, H41's final 352-case gate, candidate-only
  sanitizer, or default-path import/compile checks regress on the merged source.
- The captured graphs' logical operation trace is not exactly 61 dense triplets for control and
  `24 dense + 14 selected + 23 dense` triplets for candidate, or graph debug output contradicts the
  declared order/node-family counts.
- Graph capture/replay allocates, the process-lifetime sticky status changes on valid inputs, or any
  destination leaves its owned material representation.

Correctness uses the selected TQ oracle, not the normal fused FP8 control. Normal/TQ RoPE rounding
differences are the existing Phase-28 quality contract and are recorded rather than reclassified as
an H41 error.

## Timing protocol and decision

Allocate control and candidate full-context cache rings simultaneously from content-matched finite
nonzero values. Reverse full-arm allocation order in decision sequence two. Compile and screen before
the decision set; screening cannot select a new warp variant, layer count, split, or boundary.

For each context, run three independent fresh-process C/T/C sequences with at least 100 CUDA-graph
warmups and 2,000 timed replays per arm. Reject a sequence if control-flank drift exceeds 2%, a GPU
covariate is unhealthy, or metadata/operation traces disagree. The fresh process is the bootstrap
unit; use 50,000 deterministic sequence-level percentile draws.

Normalize exactly:

`integrated_delta_per_selected_layer = (candidate_graph_us - control_graph_us) / 14`.

The 95% upper bound must not exceed the fresh W1 full-allowance 95% lower bound in either window:

- actual-10K: `27.0292 us/selected layer`;
- historical long: `27.6973 us/selected layer`.

A crossing interval is indeterminate and permits no ad-hoc replay or retuning. A pass authorizes an
immutable default-off endpoint candidate build only. It does not authorize an endpoint result,
quality claim, production promotion, upstream contribution, or default-on behavior.

## Evidence and rollback

Archive source manifests, exact commands, operation traces, graph debug dumps, raw timing, analysis,
GPU/fabric/Xid state, restore fingerprints, and a sorted SHA-256 manifest outside CT13 before the
decision. Commit each accepted harness/source correction before the next hypothesis. A rejected
source candidate is quarantined on a named pushed ref and restored only with a Git revert from the
last accepted commit; never reconstruct source from memory.

## Accepted harness implementation

The implementation keeps the persistent ring at exactly 256,000 rows (8,000 pages), matching H40
and the 0.847816467 GiB gross N14 saving; it has no benchmark-only padding page. A focused q1/q5
composition gate reconstructs the exact dense FP8 oracle after eager and graph-replayed H41 writes.
The decision analyzer requires those four composition cases, exact cache byte counts, exact logical
operation traces, healthy P0/max-clock GPU covariates, zero uncorrected ECC/recovery action, zero
replay allocation, and the frozen C/T/C timing protocol. Two self-review rounds caught and removed
the extra-page memory-accounting error before any live run.

The first live H40 contract invocation addressed a nonexistent `/usr/local/bin/pytest` entry point
in the immutable image and exited before any GPU kernel ran. It is rejected infrastructure evidence;
the accepted runner invokes the installed module with `python3 -m pytest` and requires a fresh rerun.

The first W2 regression client detached during the long 352-case process while its `--rm` container
continued and then removed its own exit/log record. The run is excluded because its verdict cannot
be recovered. The accepted runner retains uniquely named stopped containers until their exit code
and logs are archived, after which cleanup is a separate explicit operation.

The first sanitizer invocation used an unsupported `regex:` filter spelling and Compute Sanitizer
exited in argument parsing before Python or CUDA launched. It is excluded CLI-harness evidence; the
next diagnostic used the installed tool's reported key/value form.

The second sanitizer invocation completed all 24 valid launches plus the invalid skip, but reported
the known JIT/tooling `cuKernelGetFunction INVALID_HANDLE` API event and therefore had error summary
one. It is retained as diagnostic evidence, not a pass. The decisive W2-equivalent filter uses the
candidate kernel substring and disables CUDA API-error reporting only; device memory access errors
remain enabled and must end with error summary zero.

The first integrated screen passed numerical, memory, allocation, drift, and timing checks, but its
PyTorch debug dump created only the parent directory when given extensionless targets. It is valid
screen timing but incomplete graph-debug evidence. The accepted harness uses explicit `control.dot`
and `candidate.dot` file targets as documented by PyTorch and requires a repeated pre-decision
screen; the frozen graph, splits, warps, layer interval, and numerical gates are unchanged.
