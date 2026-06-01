# syntax=docker/dockerfile:1.7

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG SKIP_PY_COMPILE=0

WORKDIR /sgl-workspace/sglang

COPY python/sglang/srt/environ.py python/sglang/srt/environ.py
COPY python/sglang/srt/layers/attention/flashinfer_mla_backend.py python/sglang/srt/layers/attention/flashinfer_mla_backend.py
COPY python/sglang/srt/layers/attention/tokenspeed_mla_backend.py python/sglang/srt/layers/attention/tokenspeed_mla_backend.py
COPY python/sglang/srt/layers/attention/tokenspeed_workspace.py python/sglang/srt/layers/attention/tokenspeed_workspace.py
COPY python/sglang/srt/layers/attention/trtllm_mla_backend.py python/sglang/srt/layers/attention/trtllm_mla_backend.py
COPY python/sglang/srt/managers/utils.py python/sglang/srt/managers/utils.py
COPY python/sglang/srt/mem_cache/memory_pool.py python/sglang/srt/mem_cache/memory_pool.py
COPY python/sglang/srt/model_executor/cuda_graph_runner.py python/sglang/srt/model_executor/cuda_graph_runner.py
COPY python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py
COPY python/sglang/srt/server_args.py python/sglang/srt/server_args.py
COPY python/sglang/srt/speculative/eagle_info_v2.py python/sglang/srt/speculative/eagle_info_v2.py
COPY python/sglang/srt/speculative/eagle_worker.py python/sglang/srt/speculative/eagle_worker.py
COPY python/sglang/srt/speculative/eagle_worker_v2.py python/sglang/srt/speculative/eagle_worker_v2.py

COPY --from=tokenspeed_mla_src . /usr/local/lib/python3.12/dist-packages/tokenspeed_mla

RUN if [ "${SKIP_PY_COMPILE}" != "1" ]; then \
      python3 -m py_compile \
        python/sglang/srt/environ.py \
        python/sglang/srt/layers/attention/flashinfer_mla_backend.py \
        python/sglang/srt/layers/attention/tokenspeed_mla_backend.py \
        python/sglang/srt/layers/attention/tokenspeed_workspace.py \
        python/sglang/srt/layers/attention/trtllm_mla_backend.py \
        python/sglang/srt/managers/utils.py \
        python/sglang/srt/mem_cache/memory_pool.py \
        python/sglang/srt/model_executor/cuda_graph_runner.py \
        python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py \
        python/sglang/srt/server_args.py \
        python/sglang/srt/speculative/eagle_info_v2.py \
        python/sglang/srt/speculative/eagle_worker.py \
        python/sglang/srt/speculative/eagle_worker_v2.py \
        /usr/local/lib/python3.12/dist-packages/tokenspeed_mla/*.py; \
    fi
