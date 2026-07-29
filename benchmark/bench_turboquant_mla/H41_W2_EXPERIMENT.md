# H41 W2 native SM100 front-end experiment

Status: predeclared; implementation not started
Base: `1a7f97535` (`research/kimi-tq-h41-w1-20260729`)
Machine scope: CT13 GPU0 for standalone compilation, correctness, graph replay, and timing;
CT13+CT14 only for a later integrated endpoint gate. No other machines.

## Hypothesis

For official Kimi K2.6 target verification (`tokens=5`, `heads=8`, latent 512, RoPE 64), one
SM100 CUDA launch can replace the accepted TQ query-quantization launch and three-launch TQ writer
without scratch tensors. It will write the N14 persistent row (`256 B` packed E2M1 latent + `2 B`
BF16 scale + `64 B` FP8 RoPE = `322 B`) and produce the exact FP8 query consumed by TokenSpeed.
The real combined front-end delta must fit W1's independently bootstrapped remainder at both
actual-10K and historical-long context.

Expected mechanism: one block owns one token. Configurable 1/2/4/8-warp variants stride the eight
query heads; warp zero additionally owns the one MLA writer row. The kernel concatenates query
latent/RoPE directly into FP8 output. When static absorption has not fused the latent rotation, each
query warp applies the accepted `D2 * H512/sqrt(512) * D1` transform before FP8 conversion. The
writer computes its L2 norm, the same normalized signed WHT, software E2M1 binning/remap and
quantized norm, BF16 dequant scale, packed nibbles, and SATFINITE E4M3 RoPE. No full-array,
dynamic, intermediate, or shadow allocation is allowed.

## Exact change

- Add a default-off native SM100 JIT operator and strict Python wrapper.
- Inputs: BF16 query latent `(T,8,512)`, BF16 query RoPE `(T,8,64)`, BF16 writer latent
  `(T,1,512)`, BF16 writer RoPE `(T,1,64)`, int64 locations `(T,)`, the accepted signs,
  boundaries, quant centroids, storage-code LUT, and a process-lifetime int32 fault word.
- Caller-owned outputs: FP8 query `(T,8,576)`, packed uint8 cache `(P,1,256)`, BF16 scale `(P,1)`,
  and FP8 RoPE `(P,1,64)`. The capture-safe core allocates nothing; an explicitly separate eager
  convenience API may allocate the query output before invoking the same core.
- Bounds are checked before any destination pointer is formed. Strict test mode traps on invalid
  locations. Integrated mode skips the invalid writer row and performs `atomicOr(status, 1)`; the
  status is never cleared per replay.
- Launch on PyTorch's current CUDA stream and support capture/replay without allocation.

## Correctness and safety falsifiers

1. Static-rotation query bytes differ from the accepted `_quantize_tq4_query` oracle, or the
   explicit-rotation branch differs after applying the accepted rotation oracle.
2. Packed nibbles, FP8 RoPE bytes, or BF16 scale bytes differ from the accepted writer on
   realistic/random/zero/basis inputs. Artificial boundary probes may report an adjacent-bin-only
   mismatch separately, but cannot qualify captured/realistic mismatches.
3. Zero, near-zero, threshold-boundary, NaN/Inf, minimum/maximum valid-location, invalid-location,
   non-contiguous, wrong-dtype/device/shape, graph-replay, or current-stream tests violate the
   wrapper/fault contract. Destination locations must be unique, matching the existing allocator
   contract; W2 integration must prove that upstream invariant without adding a device sync or
   per-replay bitmap.
4. Compute Sanitizer reports an access/race/init fault, or codegen targets anything other than
   SM100 for the experimental path.

## Performance gate

For `T in {1,2,5,10,15,20,25,30,35,40}` and warp variants `{1,2,4,8}`, use fresh-process C/T/C
timing, 100 graph warmups, 2,000 graph replays per arm, three decision processes, and allocation
order reversal in sequence two. Select a variant only from correctness-clean samples. At the
production `T=5` point, bootstrap the real combined candidate against the same normal query+writer
control used by W1. The candidate's 95% upper front-end delta must be below W1's 95% lower
remainder in both windows (`22.9462 us/layer` exact, `13.5240 us/layer` long). A physical launch
floor is not a substitute.

Passing standalone timing authorizes four-family graph integration only. It does not authorize an
endpoint image, quality claim, production promotion, or restart-recovery claim.

## Audit rule

Accepted source changes are committed on `research/kimi-tq-h41-w2-20260729`. A rejected candidate
is restored only by a Git revert from the last accepted commit and is then backed up on an explicit
rejected/quarantine ref; prior file contents are never reconstructed from memory.
