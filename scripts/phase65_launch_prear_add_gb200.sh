#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-omniva/sglang:gb200-phase65-prear-add-004962e54}"
NAME="${NAME:-kimi-gb200-best}"
MODEL_DIR="${MODEL_DIR:-/opt/kimi-gb200/models/Kimi-K2.6-2755962d}"
PORT="${PORT:-30000}"
TP="${TP:-4}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.93}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-16384}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-16384}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-256000}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-256000}"
FLASHINFER_CACHE="${FLASHINFER_CACHE:-/opt/kimi-gb200/flashinfer-cache/phase65-prear-add}"
EXTRA_SGLANG_ARGS="${EXTRA_SGLANG_ARGS:---kv-cache-dtype fp8_e4m3 --attention-backend tokenspeed_mla --page-size 32 --enable-fused-moe-sum-all-reduce --disable-cuda-graph-padding --enable-nccl-nvls}"

mkdir -p /opt/kimi-gb200/logs "${FLASHINFER_CACHE}"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

read -r -a EXTRA_ARGS <<< "${EXTRA_SGLANG_ARGS}"

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --network host \
  --ipc host \
  --shm-size 64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --cap-add SYS_NICE \
  -e PYTHONUNBUFFERED=1 \
  -e GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}" \
  -e NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}" \
  -e HF_HOME=/root/.cache/huggingface \
  -e SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-0}" \
  -e SGLANG_FLASHINFER_ALLREDUCE_FUSION_MAX_TOKENS="${SGLANG_FLASHINFER_ALLREDUCE_FUSION_MAX_TOKENS:-224}" \
  -e SGLANG_FLASHINFER_PRE_ALLREDUCE_ADD_FUSION="${SGLANG_FLASHINFER_PRE_ALLREDUCE_ADD_FUSION:-1}" \
  -e SGLANG_TQ_MLA_FUSED_DECODE="${SGLANG_TQ_MLA_FUSED_DECODE:-0}" \
  -e SGLANG_TQ_MLA_STAGED_FLASHMLA="${SGLANG_TQ_MLA_STAGED_FLASHMLA:-0}" \
  -e SGLANG_TQ_MLA_STAGED_FLASHMLA_ALLOW_JIT="${SGLANG_TQ_MLA_STAGED_FLASHMLA_ALLOW_JIT:-0}" \
  -e SGLANG_TQ_MLA_STAGED_FLASHMLA_THREADS="${SGLANG_TQ_MLA_STAGED_FLASHMLA_THREADS:-512}" \
  -e SGLANG_TQ_MLA_FAST_METADATA="${SGLANG_TQ_MLA_FAST_METADATA:-0}" \
  -e SGLANG_TQ_MLA_FUSED_METADATA_INDICES="${SGLANG_TQ_MLA_FUSED_METADATA_INDICES:-0}" \
  -e SGLANG_TQ_MLA_FUSED_KV_WRITE="${SGLANG_TQ_MLA_FUSED_KV_WRITE:-0}" \
  -e SGLANG_TQ_MLA_FUSED_ROPE_WRITE="${SGLANG_TQ_MLA_FUSED_ROPE_WRITE:-0}" \
  -e SGLANG_TQ_MLA_KV_WRITE_WORKSPACE_TOKENS="${SGLANG_TQ_MLA_KV_WRITE_WORKSPACE_TOKENS:-256}" \
  -v /opt/kimi-gb200/models:/models:ro \
  -v /opt/kimi-gb200/hf-cache:/root/.cache/huggingface \
  -v /opt/kimi-gb200/logs:/logs \
  -v "${FLASHINFER_CACHE}":/root/.cache/flashinfer \
  "${IMAGE}" \
  python3 -u -m sglang.launch_server \
    --model-path "/models/$(basename "${MODEL_DIR}")" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP}" \
    --trust-remote-code \
    --context-length "${CONTEXT_LENGTH}" \
    --max-total-tokens "${MAX_TOTAL_TOKENS}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS}" \
    --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}" \
    --reasoning-parser kimi_k2 \
    --tool-call-parser kimi_k2 \
    "${EXTRA_ARGS[@]}"

echo "Started ${NAME} on port ${PORT}"
