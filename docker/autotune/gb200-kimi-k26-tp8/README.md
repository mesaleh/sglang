# GB200 Kimi K2.6 TP8 FlashInfer tactics

These eight rank-specific FlashInfer 0.6.8.post1 tactics were selected by the
immutable production control image on ct13+ct14 during the 2026-07-16 stock-sync
promotion window. They remove startup-time tactic variance from the final image.

Runtime contract:

- NVIDIA GB200, SM100, unified TP8 across two nodes
- official Kimi K2.6 target with compressed-tensors MxInt4 MoE
- FlashInfer 0.6.8.post1, cache key `6372c823c48e8df1`
- global TP ranks 0-3 on ct13 and 4-7 on ct14

Do not reuse these files for a different GPU architecture, FlashInfer version,
model quantization, or parallel topology. `SHA256SUMS` records the promoted
files. The image copies this directory to `/root/.cache/sglang`.
