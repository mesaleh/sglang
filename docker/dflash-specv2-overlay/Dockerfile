ARG BASE_IMAGE
FROM ${BASE_IMAGE}

WORKDIR /sgl-workspace/sglang

COPY python/sglang/srt/environ.py python/sglang/srt/environ.py
COPY python/sglang/srt/server_args.py python/sglang/srt/server_args.py
COPY python/sglang/srt/layers/lm_head_gemm.py python/sglang/srt/layers/lm_head_gemm.py
COPY python/sglang/srt/layers/logits_processor.py python/sglang/srt/layers/logits_processor.py
COPY python/sglang/srt/managers/scheduler.py python/sglang/srt/managers/scheduler.py
COPY python/sglang/srt/models/deepseek_v2.py python/sglang/srt/models/deepseek_v2.py
COPY python/sglang/srt/models/dflash.py python/sglang/srt/models/dflash.py
COPY python/sglang/srt/speculative/spec_info.py python/sglang/srt/speculative/spec_info.py
COPY python/sglang/srt/speculative/dflash_info.py python/sglang/srt/speculative/dflash_info.py
COPY python/sglang/srt/speculative/dflash_utils.py python/sglang/srt/speculative/dflash_utils.py
COPY python/sglang/srt/speculative/dflash_worker.py python/sglang/srt/speculative/dflash_worker.py
COPY python/sglang/srt/speculative/dflash_info_v2.py python/sglang/srt/speculative/dflash_info_v2.py
COPY python/sglang/srt/speculative/dflash_worker_v2.py python/sglang/srt/speculative/dflash_worker_v2.py
COPY python/sglang/srt/speculative/triton_ops/dflash_prepare_block.py python/sglang/srt/speculative/triton_ops/dflash_prepare_block.py
COPY python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py

RUN python3 -m py_compile \
    python/sglang/srt/environ.py \
    python/sglang/srt/server_args.py \
    python/sglang/srt/layers/lm_head_gemm.py \
    python/sglang/srt/layers/logits_processor.py \
    python/sglang/srt/managers/scheduler.py \
    python/sglang/srt/models/deepseek_v2.py \
    python/sglang/srt/models/dflash.py \
    python/sglang/srt/speculative/spec_info.py \
    python/sglang/srt/speculative/dflash_info.py \
    python/sglang/srt/speculative/dflash_utils.py \
    python/sglang/srt/speculative/dflash_worker.py \
    python/sglang/srt/speculative/dflash_info_v2.py \
    python/sglang/srt/speculative/dflash_worker_v2.py \
    python/sglang/srt/speculative/triton_ops/dflash_prepare_block.py \
    python/sglang/srt/speculative/triton_ops/fused_kv_materialize.py
