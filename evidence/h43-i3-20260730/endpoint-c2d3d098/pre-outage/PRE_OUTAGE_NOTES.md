# H43 I3 endpoint pre-outage evidence

Captured: 2026-07-30/31 UTC. Machine scope: CT13+CT14 only.

## Identity and recovery state

- Candidate image:
  `gitlab.aws.omniva.com:5050/acls/security-analytics/sglang:kimi-k26-tq-h43-i3-20260730-3f7d419a-1eff3d1c-w8fix-v3@sha256:c2d3d098cce506721c9e57dddde7b3f4406b2da8119ff85198cd631498c51154`.
- Both nodes independently ran the embedded verifier inside that exact image with `--network none`.
  Both attempt-2 return codes are zero, their JSON is byte-identical, and both report `status=PASS`,
  H43 default off, SGLang `3354/b40fed390edd465cfc8a1dfc5407127c15a3d215f2ef37e19344b82f4c936ec3`,
  and TokenSpeed `14/e917926a0a136c1d0212b1be299916277cccee871e9e09273be08e07f1caf1c6`.
- The zero-byte `ct13/ct14/candidate-image-verifier.json` files are retained failed capture attempt 1.
  That command omitted the required `BASE_IMAGE` environment and exited before emitting JSON. It
  was a command-construction failure, not a candidate verifier failure. Attempt 2 used the exact
  build-time inputs recovered from the preserved build log and passed independently on both nodes.
- The CT13 coordinator is installed at vault commit `e27cba6`, executable SHA-256
  `273d03f829f7dabf076ba41395202e2fe9b089b2577b89f13dc739ebc14cc77c`, active/running, and
  persistently `DISARMED`. All six real endpoint timers are installed but inactive. A live-safe
  `DISARMED -> TRANSITION -> DISARMED` drill left both accepted containers unchanged.

## Frozen corpus and accepted-incumbent baseline

- `frozen-corpus/corpus.jsonl` contains exactly two excluded warmups followed by request indices
  `0-9`. Corpus SHA-256 is `733da9d57f70a20d5ef6f8b60a9dada84e501bbde67b18b552c9db675654544c`.
  Harness SHA-256 is `f18e1e9fc0ea9cc2101b1d7c0d80c79f5db3a98b54531cb1be4f16f1b218bf6f`.
- CT13 independently regenerated all 12 prompt/payload hashes from the frozen parameters before the
  baseline. CT14 intentionally has no second mutable benchmark driver; an attempted CT14 path check
  found `/opt/kimi-gb200/scripts/bench_canonical.py` absent. CT13 remains the single request driver.
- The first baseline command exited before sending a request because its create-new result parent
  was not writable by `mesaleh`. The result parent was created owner `mesaleh`, and the unchanged
  command then produced the preserved directory
  `ct13/accepted-fp8-actual10k-c1-o512-frozen-20260730`.
- That directory name is a pre-result naming error: full `accepted-server-info-full.json` proves the
  accepted incumbent is **not FP8**. It is `turboquant_4bit_e2m1` on layers `29,30` with DFlash
  `dt5/block5`, TokenSpeed MLA, page size 32, and 256K total tokens. Do not use this row as the
  non-TQ control. The sealed candidate image's timed `fp8_pre` and `fp8_post` roles are the fresh
  true-FP8 controls.
- The accepted-incumbent frozen baseline completed 10/10 measured requests. Every measured request
  used exactly 10,218 prompt tokens and 512 completion tokens with `finish_reason=length`; all 12
  prompt hashes match the frozen manifest in order. Arithmetic mean TPOT is `3.649079243 ms`, p50
  TPOT `3.623563820 ms`, mean per-request decode `274.357737 tok/s`, mean TTFT `294.559755 ms`, and
  aggregate end-to-end throughput `237.074 tok/s`. The historical approximately 297 tok/s winning
  number is the inverse-TPOT decode metric for a true FP8 control, not aggregate throughput and not
  this accepted N2 incumbent.

## Node-specific observations

- CT14 rank 1 returns HTTP 200 from `/health` but 404 from `/v1/models`; only CT13 rank 0 is the API
  request driver. This is expected two-node rank behavior, not a readiness failure.
- CT13 has one historical Xid 43 at 2026-07-30 13:57 local, before the accepted container start at
  `2026-07-30T22:28:36.002729066Z`. The exact since-start evidence is empty on both nodes. Do not
  rewrite the broader historical log as “no Xid”; the decision window is clean since accepted start.
- Accepted CT13/CT14 source-overlay archives are stored off-node under `recovery-overlays/`. Their
  runtime hashes are independently identical across the two nodes and are also pinned in the
  installed restore helper.

Decision: pre-outage evidence is complete. Arm the absolute and Stage-1 timers on both nodes only
immediately before stopping accepted, then use `fp8_pre` as the non-TQ performance control in the
predeclared `fp8_pre -> e2m1 -> h43 -> fp8_post` order after H43 Stage 1.
