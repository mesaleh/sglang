#!/usr/bin/env bash
set -euo pipefail

RUN_TAG="${H41_I1_RUN_TAG:?set a unique H41_I1_RUN_TAG}"
MODE="${1:?usage: run_h41_i1_container.sh MODE [arguments ...]}"
shift

if [[ ! "$RUN_TAG" =~ ^[a-z0-9][a-z0-9_.-]{0,50}$ ]]; then
  echo "invalid H41_I1_RUN_TAG: $RUN_TAG" >&2
  exit 2
fi
if [[ "$(hostname -s)" != *ct13 ]]; then
  echo "H41 I1 isolated execution is allowed only on CT13" >&2
  exit 2
fi

ROOT='/opt/kimi-gb200/experiments/tq-h41-i1-20260729'
SGLANG_PACKAGE="$ROOT/source/sglang/python/sglang"
TOKENSPEED_PACKAGE="$ROOT/source/tokenspeed_mla"
BENCHMARK_ROOT="$ROOT/benchmark"
CACHE_ROOT="$ROOT/cache"
IMAGE='gitlab.aws.omniva.com:5050/acls/security-analytics/sglang:kimi-k26-tq-gate-candidate-20260727-7a1-db01-admissionfix@sha256:cc6fa1f338da513f9da20ee9310ba8dafbc964e2d9a6b7d528bdef9500c1920c'
EXPECTED_IMAGE_ID='sha256:cc6fa1f338da513f9da20ee9310ba8dafbc964e2d9a6b7d528bdef9500c1920c'
EXPECTED_SGLANG_TREE='5d66a8ab990e4867b0bb1a009b0582b52b6fe603a7b9fec38fd861c7729ac53b'
EXPECTED_TOKENSPEED_TREE='113bcc8535b9eef5cdeef6f65d3978cede610f8216a4bcdee70dcaf3a28ec985'

case "$MODE" in
  integrated)
    SCRIPT='/work/bench_turboquant_mla/bench_h41_i1_integrated.py'
    ;;
  roundtrip)
    SCRIPT='/work/bench_turboquant_mla/test_h41_i1_roundtrip.py'
    ;;
  w2-regression)
    SCRIPT='/work/bench_turboquant_mla/test_h41_w2_frontend.py'
    ;;
  h40-regression)
    SCRIPT='/work/tokenspeed-test/microbench_tq4_mla_decode.py'
    ;;
  h40-contract)
    SCRIPT='pytest-module'
    ;;
  sanitizer)
    SCRIPT='/work/bench_turboquant_mla/test_h41_w2_frontend.py'
    ;;
  *)
    echo "unsupported H41 I1 mode: $MODE" >&2
    exit 2
    ;;
esac

ENTRYPOINT='python3'
container_arguments=("$SCRIPT" "$@")
if [[ "$MODE" == 'h40-contract' ]]; then
  container_arguments=(
    -m pytest /work/tokenspeed-test/test_tq4_contract.py -q "$@"
  )
fi
if [[ "$MODE" == 'sanitizer' ]]; then
  ENTRYPOINT='compute-sanitizer'
  container_arguments=(
    --tool memcheck
    --target-processes all
    --kernel-name regex:tq_mla_frontend_kernel
    python3 "$SCRIPT" --mode sanitizer
  )
fi

tree_sha256() {
  local package_dir="$1"
  (
    cd "$package_dir"
    while IFS= read -r -d '' path; do
      if [[ -L "$path" ]]; then
        printf 'SYMLINK:%s  %s\n' "$(readlink "$path")" "$path"
      else
        sha256sum "$path"
      fi
    done < <(
      find . \( -type f -o -type l \) ! -path '*/__pycache__/*' -print0 |
        LC_ALL=C sort -z
    ) | sha256sum | awk '{print $1}'
  )
}

test -d "$SGLANG_PACKAGE"
test -d "$TOKENSPEED_PACKAGE"
test -f "$BENCHMARK_ROOT/bench_turboquant_mla/bench_h41_i1_integrated.py"
if [[ "$(tree_sha256 "$SGLANG_PACKAGE")" != "$EXPECTED_SGLANG_TREE" ]]; then
  echo 'H41 I1 SGLang source tree mismatch' >&2
  exit 2
fi
if [[ "$(tree_sha256 "$TOKENSPEED_PACKAGE")" != "$EXPECTED_TOKENSPEED_TREE" ]]; then
  echo 'H41 I1 TokenSpeed source tree mismatch' >&2
  exit 2
fi

if [[ "$(sudo docker image inspect --format '{{.Id}}' "$IMAGE")" != "$EXPECTED_IMAGE_ID" ]]; then
  echo 'H41 I1 immutable image mismatch' >&2
  exit 2
fi
if sudo docker ps -a --format '{{.Names}}' | grep -Fxq "h41-i1-$RUN_TAG"; then
  echo "container h41-i1-$RUN_TAG already exists" >&2
  exit 2
fi

compute_rows="$(
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
)"
if [[ -n "$compute_rows" ]]; then
  echo "H41 I1 requires every CT13 GPU to be idle; found: $compute_rows" >&2
  exit 2
fi

sudo nvidia-smi -pm 1 >/dev/null
sudo nvidia-smi -ac 4000,1965 >/dev/null
clock_rows="$(
  nvidia-smi \
    --query-gpu=clocks.applications.graphics,clocks.applications.memory \
    --format=csv,noheader,nounits
)"
if [[ "$(printf '%s\n' "$clock_rows" | grep -c '^1965, 4000$')" -ne 4 ]]; then
  echo 'H41 I1 application-clock check failed' >&2
  printf '%s\n' "$clock_rows" >&2
  exit 2
fi
health_rows="$(
  nvidia-smi \
    --query-gpu=index,ecc.errors.uncorrected.volatile.total,gpu_recovery_action,fabric.state,fabric.status \
    --format=csv,noheader,nounits
)"
if [[ "$(printf '%s\n' "$health_rows" | grep -cE '^[0-3], 0, None, state  Completed, status Success$')" -ne 4 ]]; then
  echo 'H41 I1 ECC, recovery-action, or fabric check failed' >&2
  printf '%s\n' "$health_rows" >&2
  exit 2
fi

sudo install -d "$CACHE_ROOT"/{flashinfer,torch,torch-extensions,triton,nv,cutlass}

sudo docker run --rm --name "h41-i1-$RUN_TAG" \
  --gpus '"device=0"' \
  --ipc=host \
  --privileged \
  --cap-add SYS_NICE \
  --shm-size=32g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "$ENTRYPOINT" \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONNOUSERSITE=1 \
  -e NVIDIA_IMEX_CHANNELS=0 \
  -e TORCH_EXTENSIONS_DIR=/root/.cache/torch_extensions \
  -e TRITON_CACHE_DIR=/root/.triton \
  -e PYTHONPATH=/sgl-workspace/sglang/python \
  -v "$SGLANG_PACKAGE:/sgl-workspace/sglang/python/sglang:ro" \
  -v "$TOKENSPEED_PACKAGE:/usr/local/lib/python3.12/dist-packages/tokenspeed_mla:ro" \
  -v "$BENCHMARK_ROOT:/work:ro" \
  -v "$CACHE_ROOT/flashinfer:/root/.cache/flashinfer" \
  -v "$CACHE_ROOT/torch:/root/.cache/torch" \
  -v "$CACHE_ROOT/torch-extensions:/root/.cache/torch_extensions" \
  -v "$CACHE_ROOT/triton:/root/.triton" \
  -v "$CACHE_ROOT/nv:/root/.nv" \
  -v "$CACHE_ROOT/cutlass:/root/.cache/cutlass" \
  "$IMAGE" \
  "${container_arguments[@]}"
