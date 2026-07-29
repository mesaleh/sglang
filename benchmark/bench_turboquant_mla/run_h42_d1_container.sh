#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?usage: run_h42_d1_container.sh VARIANT [trace-only]}"
MODE="${2:-timing}"
RUN_TAG="${H42_D1_RUN_TAG:?set a unique H42_D1_RUN_TAG}"

case "$VARIANT" in
  f-chain|s-chain|s-static|s-phased|s-chain-noapdl) ;;
  *)
    echo "unsupported H42 D1 variant: $VARIANT" >&2
    exit 2
    ;;
esac
case "$MODE" in
  timing)
    timing_arguments=(--warmups 100 --samples 20 --replays-per-sample 100)
    ;;
  trace-only)
    timing_arguments=(--trace-only --warmups 5 --samples 1 --replays-per-sample 1)
    ;;
  *)
    echo "unsupported H42 D1 mode: $MODE" >&2
    exit 2
    ;;
esac

if [[ ! "$RUN_TAG" =~ ^[a-z0-9][a-z0-9_.-]{0,30}$ ]]; then
  echo "invalid H42_D1_RUN_TAG: $RUN_TAG" >&2
  exit 2
fi
if [[ "$(hostname -s)" != *ct13 ]]; then
  echo "H42 D1 isolated execution is allowed only on CT13" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_BENCHMARK_SHA256='bea0de4e9c655cd08fd1cef75cf206ed2446b09b31f6f4759b73bb5bb7f39801'
EXPECTED_ANALYZER_SHA256='0fda02add6d37e35d0630c60155befc1f503d6618241da69d6bdef637835e0bb'
if [[ "$(sha256sum "$SCRIPT_DIR/bench_h41_i1_integrated.py" | awk '{print $1}')" != \
  "$EXPECTED_BENCHMARK_SHA256" ]]; then
  echo 'H42 D1 benchmark source mismatch' >&2
  exit 2
fi
if [[ "$(sha256sum "$SCRIPT_DIR/analyze_h42_long_dependency.py" | awk '{print $1}')" != \
  "$EXPECTED_ANALYZER_SHA256" ]]; then
  echo 'H42 D1 analyzer source mismatch' >&2
  exit 2
fi

H41_I1_RUN_TAG="h42-${VARIANT}-${RUN_TAG}" \
  "$SCRIPT_DIR/run_h41_i1_container.sh" integrated \
    --context 37932 \
    --split-kv 40 \
    --allocation-order control-first \
    --seed 20260729 \
    --variant "$VARIANT" \
    "${timing_arguments[@]}"
