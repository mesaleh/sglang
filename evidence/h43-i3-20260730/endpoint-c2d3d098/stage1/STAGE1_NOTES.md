# H43 I3 Stage-1 notes

Date: 2026-07-31
Machine scope: CT13+CT14 only
Image: `gitlab.aws.omniva.com:5050/acls/security-analytics/sglang:kimi-k26-tq-h43-i3-20260730-3f7d419a-1eff3d1c-w8fix-v3@sha256:c2d3d098cce506721c9e57dddde7b3f4406b2da8119ff85198cd631498c51154`

## Hypothesis

The N14 H43 path (global layers 24-37, 338-byte selected rows) should serve without a dense/decoded
shadow, retain exact runtime identity and sticky-fault safety on all eight TP ranks, capture actual
10K decode, and switch safely to eager decode above the sealed 32K graph limit.

## Exact change

No inference source changed during Stage-1. The candidate used immutable SGLang `3f7d419a5` and
TokenSpeed `1eff3d1c`. One operational coordinator defect was fixed: successful `docker logs`
stderr was previously discarded, while SGLang writes attestations and sticky-fault lines there.
The fix is vault commits `1b23bf9` and `4d24696`; 21/21 tests, py_compile, checksum sealing,
three self-review passes, and two Claude Opus 5 review rounds ended in LGTM.

## Result

- All 64 model shards loaded. All eight ranks emitted exact initialization and post-CUDA-graph H43
  attestations: H43 and E2M1 enabled, selected row 338 bytes, codebook slots 14, FP8-RoPE slots 14,
  global selected layers 24-37, native extension SHA-256 exact, pool bytes exact, and sticky status
  zero. The coordinator positive gate observed four ranks per node and entered MONITORING.
- Short concurrency c1/c2/c4: 7/7 HTTP-200 responses, 64/64 completion tokens, `length` finishes,
  coherent nonempty output.
- Actual-10K: 10,215 prompt + 128 completion tokens, graph decode, HTTP 200, coherent output.
- Long eager: 37,928 prompt + 128 completion tokens, eager decode, HTTP 200, coherent output.
- Boundary request 1: 32,700 prompt + 512 completion = 33,212 total, HTTP 200 and coherent length-cap
  output. It proved the eager side but the 68-token graph segment crossed before a scheduler log.
- Boundary request 2: 31,002 prompt + 3,200 completion = 34,202 total, HTTP 200 and coherent
  length-cap output. The same request logged `cuda graph: True` through token 32,704 and
  `cuda graph: False` from token 32,800 onward.
- Both nodes remained healthy. All eight GPUs were P0 at 1965/4000 MHz with zero volatile and
  aggregate uncorrected ECC. No Xid/SXid appeared after launch. Containers were not OOM-killed.
- Exact tensor accounting remains 0.778770447 GiB/rank net saved versus FP8, 6.230163574 GiB over
  TP8; incremental persistent saving versus incumbent E2M1 is 0.160217285 GiB/rank, 1.281738281
  GiB over TP8.

Decision: PASS Stage-1 and proceed to the frozen timed order FP8-pre, incumbent E2M1, H43, FP8-post.

## Failed probes and reflection

1. The first harness invocation failed before requests because the new result subdirectory was not
   writable. Ownership was corrected once and the identical command reran. This rules out inference
   involvement and suggests precreating role evidence directories in future campaigns.
2. The first coordinator arm failed closed with zero observed ranks. Manual logs proved 4+4
   attestations; the coordinator returned stdout only on success while Docker supplied SGLang logs
   on stderr. The accepted fix now opts into both streams only for container logs, preserves stdout
   parsing elsewhere, inserts a safe line boundary, and has checksum-locked tests. This was a real
   safety-observability bug, not an H43 model failure.
3. `/v1/tokenize` returned HTTP 500 because Kimi token IDs exceed orjson's signed 64-bit response
   range. Offline `AutoTokenizer` calibration was used instead. The endpoint remained HTTP-healthy
   and normal completions continued. This is an independent SGLang API serialization bug worth an
   upstream fix; it neither changes H43 memory nor invalidates completion correctness.
4. The first boundary request's pre-cap graph interval was too short for a scheduler log. The
   second request deliberately lengthened that interval and directly proved the same-request graph
   to eager transition. This avoids treating two independent requests as stronger evidence than
   they are.
5. The first post-Stage-1 coordinator transition was accidentally invoked through `sudo`.
   Atomic replacement therefore made `state.json` root-owned, and the unprivileged coordinator
   service failed closed on its next read. The already-active FP8-pre and absolute timers preserved
   recovery coverage. Ownership was repaired, the coordinator restarted DISARMED, and FP8-pre
   TRANSITION was recreated as `mesaleh`. The hardening rejects any future CLI whose effective UID
   differs from the pre-provisioned state-directory owner before it touches lock, event, or state
   artifacts. This is an operational safety bug independent of H43 inference.
