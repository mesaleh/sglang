# H43 I3 matched endpoint notes

Date: 2026-07-31
Machine scope: CT13+CT14 only
Image: `gitlab.aws.omniva.com:5050/acls/security-analytics/sglang:kimi-k26-tq-h43-i3-20260730-3f7d419a-1eff3d1c-w8fix-v3@sha256:c2d3d098cce506721c9e57dddde7b3f4406b2da8119ff85198cd631498c51154`
Workload: frozen actual-10K, 10,218 realized prompt tokens, c1/o512, two excluded warmups and ten measured requests per role
Order: `fp8_pre -> e2m1 -> h43 -> fp8_post`

## Hypothesis

The no-shadow N14 H43 path should preserve the qualified 338-byte selected rows and
`0.778770447 GiB/rank` net FP8 memory saving while keeping mean endpoint TPOT within 5% of a
same-image, time-interpolated FP8 control. The predeclared automatic decision bands were TPOT
`PASS <= +4.5%`, `FAIL_PERF > +5.5%`; TTFT `PASS <= +9%`, `FAIL_PERF > +11%`; and FP8-flank
validity required absolute mean drift no greater than 1% TPOT, 2% TTFT, 5% response acceptance,
and 2% target-verification time. Values in the gaps were `NO_DECISION`.

## Exact change

No source or runtime setting changed during the frozen sequence. Every role used a fresh container
and empty role-specific compile cache from the same immutable image. FP8 controls disabled all TQ
and H43 gates. E2M1 and H43 both selected global layers 24-37; E2M1 used the incumbent 386-byte
row, while H43 used the 338-byte packed-latent/BF16-scale/FP8-RoPE/codebook row and exact prebuilt
native extension SHA-256 `2dde881c30c44d2fdbbdda050bf641cdd31ffdf2e7256de1b729699728f65dd2`.

The standard-library analyzer validates exact role order, all 12 unique frozen prompt hashes and
request fields, cross-role realized token counts, structurally complete 512-token responses, the 135-minute
timed-bracket bound, per-index wall-clock FP8 interpolation, R type-7 tail quantiles, and a
deterministic 50,000-draw matched-index bootstrap. Thirteen unit tests pass, including the observed
invalid-flank-plus-failing-candidate precedence, isolated tail/TTFT failures, and every flank-drift
cap. Before measurement, three Claude Opus 5 analyzer-review rounds drove strict
uniqueness/type/positivity checks, a scoped verdict, contractual memory labeling, and the
contract-derived bracket bound; that analyzer review ended in `LGTM`.

## Result

| Role | Mean TPOT | Mean decode | Mean TTFT | Aggregate end-to-end | Derived acceptance | Derived target verify |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP8 pre | 3.564081 ms | 280.973 tok/s | 288.384 ms | 242.690 tok/s | 2.642695 | 9.390861 ms |
| E2M1 N14 | 3.817868 ms | 262.520 tok/s | 245.589 ms | 233.090 tok/s | 2.552059 | 9.702495 ms |
| H43 N14 | 3.867008 ms | 259.379 tok/s | 324.918 ms | 222.510 tok/s | 2.531149 | 9.746193 ms |
| FP8 post | 3.601179 ms | 277.987 tok/s | 293.194 ms | 239.987 tok/s | 2.642695 | 9.490308 ms |

- FP8-post versus FP8-pre mean drift was `+1.040872%` TPOT, `+1.668019%` TTFT, `0%`
  response acceptance, and `+1.058981%` target verification. Only TPOT exceeded its validity cap,
  by `0.040872` percentage point. The deterministic FP8 flanks returned byte-identical responses
  and chunk counts, so acceptance drift was structurally zero and target-verify drift was collinear
  with TPOT; only TPOT and TTFT supplied independent flank constraints in this campaign.
- Against per-index FP8 interpolated at the H43 measurement midpoint, H43 was `+7.716439%` mean
  TPOT with deterministic matched-bootstrap 95% interval `[+2.882239%, +12.925275%]`,
  `+11.371174%` mean TTFT, `-4.220938%` response acceptance, and `+3.021919%` target-verification
  time. Its per-index TPOT ratio p95 was `1.214025` and maximum `1.216297`; both tail limits fail.
- E2M1 attribution was `+6.744321%` mean TPOT, `-15.317973%` mean TTFT, `-3.429695%`
  response acceptance, and `+2.949197%` target-verification time. H43 adds `0.972118` percentage
  point of matched TPOT penalty over E2M1; most of the **mean-TPOT** loss is already present in the
  14-layer compressed-cache topology. That statement does not apply to TTFT: H43 is `+32.30%`
  versus same-layer E2M1 N14 (`324.918` versus `245.589 ms`), an independent, unexplained H43
  prefill/startup-path blocker that speculative acceptance cannot explain.
- All four roles completed 12/12 structurally valid requests and all ten measured requests produced
  exactly 512 completion tokens. Frozen prompt hashes match; canonical payload equality is
  harness-attested by the frozen manifest but cannot be re-derived from raw records that carry only
  prompt hashes and scalar request fields. The timed bracket lasted 3,537.133 seconds,
  well inside the 8,100-second bound. Runtime attestations, container identities, clocks, ECC,
  Xid/SXid, OOM, and sticky-fault gates passed. Shutdown-time `SystemExit: 0`/`CancelledError` on
  CT13 and dummy-worker startup cancellation on CT14 occurred only after the coordinator issued
  SIGTERM with zero requests remaining; pre-stop role logs and machine gates were clean.
- The analyzer's formal performance verdict is `NO_DECISION (fp8-flank-drift)` because control
  validity has precedence over the candidate bands. Independently, H43's observed mean TPOT, TTFT,
  and tail bands are all `FAIL_PERF`. No rerun, resampling, post-hoc tuning, or broader quality
  window was performed. The TPOT matched-bootstrap interval crosses the 5% gate
  (`[+2.882239%, +12.925275%]`) and does not alone exclude a within-gate population mean; the
  engineering rejection rests on the joint observed mean, TTFT, both tails, and the E2M1 N14 floor.
- The memory claim remains real and no-shadow: `0.778770447 GiB/rank` or `6.230163574 GiB` over
  TP8 versus FP8. H43's incremental persistent saving over incumbent E2M1 is
  `0.160217285 GiB/rank` or `1.281738281 GiB` over TP8. These are byte-accounting claims from the
  sealed contract, not measurements made by the timing analyzer.
- Restoration preserved every experimental container, restarted the exact accepted E2M1 N2
  service IDs `03b4ce2c...efa976` and `a64d7605...bc15d`, produced fresh successful restore
  markers, passed health on both nodes and an independent completion returning `7`, and left all
  eight GPUs at P0 1965/4000 MHz with zero volatile uncorrectable ECC and no Xid/SXid. The
  coordinator state machine entered `DISARMED` at 04:24:37 UTC. Independent systemd safeguards
  were still armed in the 04:25:47/49 `restore-final-ct13/14.txt` captures, then both nodes stopped
  the FP8-post and absolute timers at 04:25:58, before the 04:39:23/24 phase deadlines. The
  timestamped 04:54 `coordinator/post-deadline-timer-proof.txt` shows all timers inactive/dead,
  empty last-trigger fields, and both deadline services never started with no journal entries.
  Rank-local accepted Docker image IDs differ because this historical accepted deployment uses
  separately pinned node-local images and mutable overlays. Pre-campaign evidence attests that the
  restore contract pins each node's exact image ID, configured image, full container ID, and overlay
  digests; the timed restore evidence itself shows image/container IDs, name, and `StartedAt`.

Decision: **do not promote H43 and do not run the broader quality gate**. The formal frozen campaign
is `NO_DECISION`, while the available engineering evidence is adverse enough to reject this exact
N14 H43/DFlash endpoint configuration under the 5% owner gate. A repeated identical bracket would
address the 0.0409-point control-validity miss but would not explain the joint TPOT, TTFT, tail, and
same-layer E2M1 results; the next experiment must add mechanism evidence rather than resample this
configuration opportunistically.

## Reflection and next hypothesis

The recorded acceptance and target-verification values are stream-derived proxies, not independent
server telemetry: per request, `target_verify / acceptance * 512/511` is exactly TPOT. Their split
is therefore an algebraic decomposition and motivates a hypothesis; it does not prove that TQ
caused a DFlash acceptance loss or that the native material path is otherwise neutral. A
mean-of-means counterfactual using FP8-derived acceptance is `3.695 ms` (`+2.93%` mean TPOT), while
matched per-index recomputation is `3.699 ms` (`+3.03%`). Both clear only the 5% mean-TPOT gate;
H43's `+11.37%` TTFT and both observed tail limits would still fail.

The harness did capture one independent server signal outside the analyzer: fresh-role post-run
`/server_info` lifetime average accept length was `2.652` FP8-pre, `2.592` E2M1, `2.574` H43, and
`2.647` FP8-post. It corroborates the same ordering, but each value includes the coordinator smoke,
two warmups, and ten measured requests; the before snapshot was unavailable and no per-request
server fields were retained. It is therefore excluded from the verdict and does not prove
causation. The successor harness should preserve final-response `meta_info.spec_accept_length` or a
clean before/after server counter delta per measured request.

The next clean direction is a fixed-memory attribution experiment, not EAGLE3 yet. It must collect
independent server-side draft accept-length telemetry while screening predeclared selected-layer and
numerical-representation variants, and profile H43's separate prefill/TTFT delta in the same run.
Only then can acceptance recovery or H43 prefill-path work be selected without exceeding the row
budget or weakening the sealed component gates. A fresh cold FP8/TQ/FP8 endpoint bracket remains
mandatory for any successor. Aggregate throughput (`222.510 tok/s` for H43 here) must not be
compared with the historical approximately `297 tok/s` decode-only inverse-TPOT figure. That
historical figure belongs to the accepted E2M1 N2 deployment, which was not measured in this
bracket; only within-bracket comparisons are supported.
