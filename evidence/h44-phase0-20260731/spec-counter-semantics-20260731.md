# H44 speculative-counter semantics probe

Date: 2026-07-31
Host/service: CT13, accepted E2M1 N2 Kimi K2.6 service on port 30000
Source base: `f365d5c8bc7fb826d82b30b414a2d2ba24142686`
Mutation: none; ordinary c1 native generation requests only

## Hypothesis

The planned process counter and final native response metadata may use different
speculative-acceptance conventions at request boundaries. If so, equating either
quantity with emitted `completion_tokens` would invalidate otherwise correct
telemetry.

Expected mechanism: scheduler `update_spec_metrics()` counts accepted drafts plus
one target/bonus token per verification forward. Tokenizer response metadata
computes `spec_accept_length` from externally emitted completion tokens. The
initial prefill token and final speculative overshoot/truncation can make the two
integer numerators differ.

## Probe result

The accepted service was queried serially with maximum output lengths 8, 32, and
64. No concurrent request was active. This service predates the new cumulative
counters, so the table reconciles final response metadata only; it is not evidence
that a batch-level process counter is request-exact.

| Max output | Completion | Correct drafts | Verify count | Correct + verify | Reported accept length | Internal - emitted |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 8 | 1 | 6 | 7 | 1.3333333333333333 | -1 |
| 32 | 32 | 19 | 16 | 35 | 2.0 | +3 |
| 64 | 64 | 24 | 40 | 64 | 1.6 | 0 |

Every reported accept length equals `completion_tokens / spec_verify_ct`. It does
not consistently equal `(spec_num_correct_drafts + spec_verify_ct) /
spec_verify_ct`.

Initial source inspection confirmed both definitions:

- `scheduler_components/metrics_reporter.py::update_spec_metrics()` adds
  `num_correct_drafts + batch_size` to the legacy batch-level scheduler counter.
- `tokenizer_manager.py::_calculate_spec_decoding_metrics()` sets response
  `spec_accept_length` to `completion_tokens / spec_verify_ct`.
- `batch_result_processor.py` accumulates the per-request correct-draft and verify
  counts before the final output boundary is represented in response metadata.

For configured speculative draft budget 5, the exact full-request boundary range
is `[-1, 3]`: the lower edge represents the initial prefill token not counted by a
verification forward, and the upper edge represents a final five-token verified
run arriving when one emitted token remains.

An external code review then traced overlap scheduling and found a second,
independent boundary: one delayed verify batch for a request that has already
finished still reaches batch-level `update_spec_metrics()`, while the per-request
`spec_verify_ct` and correct-draft fields correctly skip it. Therefore a cumulative
counter incremented at that batch call site cannot be reconciled to final response
metadata even at concurrency one.

## Decision

Keep the two new cumulative counters, but update them once at final scheduler
output from the completed request's own `spec_verify_ct` and
`spec_num_correct_drafts`. Do not update them from the legacy batch-level path.
`finished_output` makes this final-output accounting idempotent under the delayed
overlap output attempt. The corrected H44 harness records three separate quantities:

1. internal accept length: `internal_accept_tokens / verify_ct`;
2. response accept length: `completion_tokens / verify_ct`; and
3. emitted inter-token-interval yield: `(completion_tokens - 1) / verify_ct`.

Use the third quantity for acceptance-restored TPOT algebra because TPOT measures
the same first-to-last-token intervals. Use internal acceptance only as a separate
implementation diagnostic. Exact qualification must reconcile all integer
identities and enforce the bounded boundary adjustment.

## Reflection

The failure was useful: it rules out using either SGLang field as an unqualified
synonym for "tokens emitted per verification," and it rules out batch-level legacy
counters as request attribution sources under overlap scheduling. The response
discrepancy is deterministic boundary accounting; the extra legacy batch is a
separate scheduler-work artifact. Catching both before an outage prevents a valid
Window A from being rejected by impossible counter identities and prevents final
overshoot or trailing overlap work from being mistaken for acceptance improvement.
