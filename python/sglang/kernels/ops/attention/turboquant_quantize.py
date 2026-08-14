"""Fused TurboQuant quantize: WHT (CUDA) + searchsorted+pack (Triton).

The WHT transform uses SGLang's existing CUDA hadamard kernel (can't fuse).
The searchsorted + centroid lookup + quant_norm + bit-pack is fused into
a single Triton kernel for both 4-bit and 2-bit, replacing ~8 separate
PyTorch ops per call.

Total: 3 kernel launches (batched norm+normalize + WHT rotation + batched pack+store).
"""

import triton
import triton.language as tl


@triton.jit
def _fused_pack_4bit_kernel(
    Y,          # (tokens, heads, dim) float32 — WHT-rotated unit vectors
    Packed,     # (tokens, heads, packed_dim) uint8 output
    DScale,     # (tokens, heads) bf16 output — dequant scale = norm / max(qnorm, eps)
    Norms,      # (tokens, heads) float32 — L2 norms
    Boundaries, # (N_BOUNDARIES,) float32
    Centroids,  # (N_CENTROIDS,) float32
    stride_y_t,
    stride_y_h,
    stride_p_t,
    stride_p_h,
    stride_ds_t,
    stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_half: tl.constexpr,
):
    """Fused searchsorted + centroid gather + quant_norm + 4-bit nibble pack.

    One program per (token, head). Processes dim/2 pairs of elements.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_pair = tl.arange(0, BLOCK_PACKED)
    mask_pair = offs_pair < Lk_half

    # Load even/odd elements of the rotated vector
    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_even = tl.load(y_base + offs_pair * 2, mask=mask_pair, other=0.0)
    y_odd = tl.load(y_base + offs_pair * 2 + 1, mask=mask_pair, other=0.0)

    # Searchsorted: count boundaries less than y (linear scan, N_BOUNDARIES is small: 7 or 15)
    idx_even = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_odd = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_even += tl.where(y_even > bound, 1, 0).to(tl.int32)
        idx_odd += tl.where(y_odd > bound, 1, 0).to(tl.int32)

    # Centroid lookup (gather from small table)
    c_even = tl.load(Centroids + idx_even, mask=mask_pair, other=0.0)
    c_odd = tl.load(Centroids + idx_odd, mask=mask_pair, other=0.0)

    # Compute quant_norm and dequant scale in one shot
    qnorm_sq = tl.sum(c_even * c_even, axis=0) + tl.sum(c_odd * c_odd, axis=0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)
    tl.store(DScale + pid_t * stride_ds_t + pid_h, dscale)

    # 4-bit pack: (idx_odd << 4) | idx_even
    packed = ((idx_odd & 0xF) << 4) | (idx_even & 0xF)
    p_ptr = Packed + pid_t * stride_p_t + pid_h * stride_p_h + offs_pair
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_pair)


@triton.jit
def _fused_pack_2bit_kernel(
    Y,          # (tokens, heads, dim) float32 — WHT-rotated unit vectors
    Packed,     # (tokens, heads, packed_dim) uint8 output
    DScale,     # (tokens, heads) bf16 output — dequant scale
    Norms,      # (tokens, heads) float32 — L2 norms
    Boundaries, # (N_BOUNDARIES,) float32
    Centroids,  # (N_CENTROIDS,) float32
    stride_y_t,
    stride_y_h,
    stride_p_t,
    stride_p_h,
    stride_ds_t,
    stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_quarter: tl.constexpr,
):
    """Fused searchsorted + centroid gather + quant_norm + 2-bit pack.

    One program per (token, head). Processes dim/4 groups of 4 elements.
    Packing: byte = (idx3 << 6) | (idx2 << 4) | (idx1 << 2) | idx0
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_group = tl.arange(0, BLOCK_PACKED)
    mask_group = offs_group < Lk_quarter

    # Load 4 elements per group
    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_0 = tl.load(y_base + offs_group * 4, mask=mask_group, other=0.0)
    y_1 = tl.load(y_base + offs_group * 4 + 1, mask=mask_group, other=0.0)
    y_2 = tl.load(y_base + offs_group * 4 + 2, mask=mask_group, other=0.0)
    y_3 = tl.load(y_base + offs_group * 4 + 3, mask=mask_group, other=0.0)

    # Searchsorted: count boundaries less than y (N_BOUNDARIES = 3 for 2-bit)
    idx_0 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_1 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_2 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_3 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_0 += tl.where(y_0 > bound, 1, 0).to(tl.int32)
        idx_1 += tl.where(y_1 > bound, 1, 0).to(tl.int32)
        idx_2 += tl.where(y_2 > bound, 1, 0).to(tl.int32)
        idx_3 += tl.where(y_3 > bound, 1, 0).to(tl.int32)

    # Centroid lookup
    c_0 = tl.load(Centroids + idx_0, mask=mask_group, other=0.0)
    c_1 = tl.load(Centroids + idx_1, mask=mask_group, other=0.0)
    c_2 = tl.load(Centroids + idx_2, mask=mask_group, other=0.0)
    c_3 = tl.load(Centroids + idx_3, mask=mask_group, other=0.0)

    # Compute quant_norm and dequant scale in one shot
    qnorm_sq = (
        tl.sum(c_0 * c_0, axis=0) + tl.sum(c_1 * c_1, axis=0)
        + tl.sum(c_2 * c_2, axis=0) + tl.sum(c_3 * c_3, axis=0)
    )
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)
    tl.store(DScale + pid_t * stride_ds_t + pid_h, dscale)

    # 2-bit pack: (idx3 << 6) | (idx2 << 4) | (idx1 << 2) | idx0
    packed = (
        ((idx_3 & 0x03) << 6)
        | ((idx_2 & 0x03) << 4)
        | ((idx_1 & 0x03) << 2)
        | (idx_0 & 0x03)
    )
    p_ptr = Packed + pid_t * stride_p_t + pid_h * stride_p_h + offs_group
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_group)


@triton.jit
def _fused_norm_normalize_kernel(
    X,          # (tokens, heads, dim) bf16/fp16/fp32
    Out,        # (tokens, heads, dim) float32 — unit vectors
    Norms,      # (tokens, heads) float32
    stride_x_t,
    stride_x_h,
    stride_o_t,
    stride_o_h,
    stride_n_t,
    BLOCK_DIM: tl.constexpr,
    Lk: tl.constexpr,
):
    """Fused L2 norm + normalize: out = x / max(||x||, eps), norms = ||x||.

    One program per (token, head). Replaces torch.linalg.norm + where + div.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < Lk

    # Load x
    x_ptr = X + pid_t * stride_x_t + pid_h * stride_x_h + offs_d
    x = tl.load(x_ptr, mask=mask_d, other=0.0).to(tl.float32)

    # L2 norm
    x_sq = x * x
    norm_sq = tl.sum(x_sq, axis=0)
    norm = tl.sqrt(norm_sq)

    # Normalize (safe division)
    safe_norm = tl.where(norm > 0.0, norm, 1.0)
    x_unit = x / safe_norm

    # Store
    out_ptr = Out + pid_t * stride_o_t + pid_h * stride_o_h + offs_d
    tl.store(out_ptr, x_unit, mask=mask_d)
    tl.store(Norms + pid_t * stride_n_t + pid_h, norm)


@triton.jit
def _fused_norm_normalize_kv_kernel(
    X_K,        # (tokens, heads, dim) — K input
    X_V,        # (tokens, heads, dim) — V input
    Out,        # (2*tokens, heads, dim) float32 — unit vectors [K; V]
    Norms,      # (2*tokens, heads) float32 — norms [K; V]
    stride_k_t,
    stride_k_h,
    stride_v_t,
    stride_v_h,
    stride_o_t,
    stride_o_h,
    stride_n_t,
    TOKENS: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
    Lk: tl.constexpr,
):
    """Batched norm+normalize for K and V in a single kernel launch.

    pid_t in [0, 2*TOKENS): first TOKENS programs process K, rest process V.
    Output is contiguous: [K_unit; V_unit] in Out, [K_norms; V_norms] in Norms.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_d = tl.arange(0, BLOCK_DIM)
    mask_d = offs_d < Lk

    # Select K or V input based on pid_t
    is_v = pid_t >= TOKENS
    local_t = pid_t - TOKENS * tl.where(is_v, 1, 0)

    k_base = X_K + local_t * stride_k_t + pid_h * stride_k_h
    v_base = X_V + local_t * stride_v_t + pid_h * stride_v_h
    x_ptr = tl.where(is_v, v_base, k_base) + offs_d
    x = tl.load(x_ptr, mask=mask_d, other=0.0).to(tl.float32)

    # L2 norm
    norm = tl.sqrt(tl.sum(x * x, axis=0))

    # Normalize (safe division)
    safe_norm = tl.where(norm > 0.0, norm, 1.0)
    x_unit = x / safe_norm

    # Store to contiguous output (indexed by pid_t directly)
    out_ptr = Out + pid_t * stride_o_t + pid_h * stride_o_h + offs_d
    tl.store(out_ptr, x_unit, mask=mask_d)
    tl.store(Norms + pid_t * stride_n_t + pid_h, norm)


@triton.jit
def _fused_pack_store_4bit_kernel(
    Y,              # (tokens, heads, dim) float32 — WHT-rotated unit vectors
    Norms,          # (tokens, heads) float32 — L2 norms
    Loc,            # (tokens,) int64 — pool slot indices
    KBuffer,        # (pool_size, heads, packed_dim) uint8 — destination
    DScaleBuffer,   # (pool_size, heads) bf16 — destination
    RopeSrc,        # optional (tokens, heads, rope_dim) bf16
    RopeBuffer,     # optional (pool_size, heads, rope_dim) bf16
    Boundaries,
    Centroids,
    stride_y_t,
    stride_y_h,
    stride_kb_s,    # KBuffer stride for pool_size dim
    stride_kb_h,
    stride_ds_s,    # DScaleBuffer stride for pool_size dim
    stride_rope_src_t,
    stride_rope_src_h,
    stride_rope_dst_s,
    stride_rope_dst_h,
    stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_half: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    STORE_ROPE: tl.constexpr,
):
    """Fused searchsorted + pack + dscale + optional RoPE scatter store."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Get scatter destination
    pool_slot = tl.load(Loc + pid_t)

    offs_pair = tl.arange(0, BLOCK_PACKED)
    mask_pair = offs_pair < Lk_half

    # Load WHT-rotated unit vector
    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_even = tl.load(y_base + offs_pair * 2, mask=mask_pair, other=0.0)
    y_odd = tl.load(y_base + offs_pair * 2 + 1, mask=mask_pair, other=0.0)

    # Searchsorted
    idx_even = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_odd = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_even += tl.where(y_even > bound, 1, 0).to(tl.int32)
        idx_odd += tl.where(y_odd > bound, 1, 0).to(tl.int32)

    # Centroid lookup + quant_norm + dscale
    c_even = tl.load(Centroids + idx_even, mask=mask_pair, other=0.0)
    c_odd = tl.load(Centroids + idx_odd, mask=mask_pair, other=0.0)
    qnorm_sq = tl.sum(c_even * c_even, axis=0) + tl.sum(c_odd * c_odd, axis=0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    # Pack and scatter store directly to KV pool
    packed = ((idx_odd & 0xF) << 4) | (idx_even & 0xF)
    p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_pair
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_pair)
    tl.store(DScaleBuffer + pool_slot * stride_ds_s + pid_h, dscale)

    if STORE_ROPE:
        rope_mask = offs_pair < ROPE_DIM
        rope = tl.load(
            RopeSrc + pid_t * stride_rope_src_t + pid_h * stride_rope_src_h + offs_pair,
            mask=rope_mask,
            other=0.0,
        )
        rope_ptr = (
            RopeBuffer
            + pool_slot * stride_rope_dst_s
            + pid_h * stride_rope_dst_h
            + offs_pair
        )
        tl.store(rope_ptr, rope, mask=rope_mask)


@triton.jit
def _fused_pack_store_native_e2m1_mla_kernel(
    Y,              # (tokens, 1, 512) float32 — WHT-rotated unit vectors
    Norms,          # (tokens, 1) float32 — pre-rotation L2 norms
    Loc,            # (tokens,) int32/int64 — pool slot indices
    PackedBuffer,   # (pool_size, 1, 256) uint8 — hardware E2M1 nibbles
    DScaleBuffer,   # (pool_size, 1) bf16 — accepted N8 token scale
    RopeSrc,        # (tokens, 1, 64) bf16/fp16/fp32
    RopeBuffer,     # (pool_size, 1, 64) fp8_e4m3fn
    Boundaries,     # (14,) float32 — midpoint boundaries in raw-level units
    Levels,         # (15,) float32 — sorted E2M1 reconstruction values
    Codes,          # (15,) uint8 — sorted-index to hardware-code map
    stride_y_t,
    stride_y_h,
    stride_p_s,
    stride_p_h,
    stride_ds_s,
    stride_rope_src_t,
    stride_rope_src_h,
    stride_rope_dst_s,
    stride_rope_dst_h,
    stride_n_t,
    GRID: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    LORA_HALF: tl.constexpr,
    ROPE_DIM: tl.constexpr,
):
    """Select native E2M1 codes, pack them, and scatter the complete MLA row."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    # Widen before multiplying by row strides: the production pool can exceed
    # 2**31 / 256 rows, so int32 pointer arithmetic would wrap even though the
    # slot index itself still fits in int32.
    pool_slot = tl.load(Loc + pid_t).to(tl.int64)

    offs_pair = tl.arange(0, BLOCK_PACKED)
    pair_mask = offs_pair < LORA_HALF
    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_even = tl.load(y_base + offs_pair * 2, mask=pair_mask, other=0.0)
    y_odd = tl.load(y_base + offs_pair * 2 + 1, mask=pair_mask, other=0.0)

    # The accepted oracle divides by the frozen grid before bucketization.
    # Keep that operation order instead of folding the grid into boundaries.
    scaled_even = y_even / GRID
    scaled_odd = y_odd / GRID
    idx_even = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_odd = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(14):
        boundary = tl.load(Boundaries + b)
        idx_even += tl.where(scaled_even > boundary, 1, 0).to(tl.int32)
        idx_odd += tl.where(scaled_odd > boundary, 1, 0).to(tl.int32)

    raw_even = tl.load(Levels + idx_even, mask=pair_mask, other=0.0)
    raw_odd = tl.load(Levels + idx_odd, mask=pair_mask, other=0.0)
    code_even = tl.load(Codes + idx_even, mask=pair_mask, other=0).to(tl.int32)
    code_odd = tl.load(Codes + idx_odd, mask=pair_mask, other=0).to(tl.int32)

    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    is_zero = norm <= 0.0
    raw_even = tl.where(is_zero, 0.0, raw_even)
    raw_odd = tl.where(is_zero, 0.0, raw_odd)
    code_even = tl.where(is_zero, 0, code_even)
    code_odd = tl.where(is_zero, 0, code_odd)

    # Freeze the N8 scale operation order: norm(raw * grid), then
    # (input_norm / quant_norm) * grid, and finally BF16 rounding.
    quant_even = raw_even * GRID
    quant_odd = raw_odd * GRID
    quant_norm_sq = tl.sum(quant_even * quant_even, axis=0) + tl.sum(
        quant_odd * quant_odd, axis=0
    )
    quant_norm = tl.sqrt(quant_norm_sq)
    safe_quant_norm = tl.where(quant_norm > 0.0, quant_norm, 1.0)
    dscale = tl.where(is_zero, 0.0, (norm / safe_quant_norm) * GRID)

    packed = ((code_odd & 0xF) << 4) | (code_even & 0xF)
    packed_ptr = (
        PackedBuffer
        + pool_slot * stride_p_s
        + pid_h * stride_p_h
        + offs_pair
    )
    tl.store(packed_ptr, packed.to(tl.uint8), mask=pair_mask)
    tl.store(
        DScaleBuffer + pool_slot * stride_ds_s + pid_h,
        dscale.to(tl.bfloat16),
    )

    rope_mask = offs_pair < ROPE_DIM
    rope = tl.load(
        RopeSrc
        + pid_t * stride_rope_src_t
        + pid_h * stride_rope_src_h
        + offs_pair,
        mask=rope_mask,
        other=0.0,
    )
    rope_ptr = (
        RopeBuffer
        + pool_slot * stride_rope_dst_s
        + pid_h * stride_rope_dst_h
        + offs_pair
    )
    tl.store(rope_ptr, rope, mask=rope_mask)


@triton.jit
def _fused_pack_store_2bit_kernel(
    Y, Norms, Loc,
    KBuffer, DScaleBuffer,
    Boundaries, Centroids,
    stride_y_t, stride_y_h,
    stride_kb_s, stride_kb_h,
    stride_ds_s, stride_n_t,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_quarter: tl.constexpr,
):
    """Fused searchsorted + pack + dscale + scatter store (2-bit)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pool_slot = tl.load(Loc + pid_t)

    offs_group = tl.arange(0, BLOCK_PACKED)
    mask_group = offs_group < Lk_quarter

    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_0 = tl.load(y_base + offs_group * 4, mask=mask_group, other=0.0)
    y_1 = tl.load(y_base + offs_group * 4 + 1, mask=mask_group, other=0.0)
    y_2 = tl.load(y_base + offs_group * 4 + 2, mask=mask_group, other=0.0)
    y_3 = tl.load(y_base + offs_group * 4 + 3, mask=mask_group, other=0.0)

    idx_0 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_1 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_2 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_3 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_0 += tl.where(y_0 > bound, 1, 0).to(tl.int32)
        idx_1 += tl.where(y_1 > bound, 1, 0).to(tl.int32)
        idx_2 += tl.where(y_2 > bound, 1, 0).to(tl.int32)
        idx_3 += tl.where(y_3 > bound, 1, 0).to(tl.int32)

    c_0 = tl.load(Centroids + idx_0, mask=mask_group, other=0.0)
    c_1 = tl.load(Centroids + idx_1, mask=mask_group, other=0.0)
    c_2 = tl.load(Centroids + idx_2, mask=mask_group, other=0.0)
    c_3 = tl.load(Centroids + idx_3, mask=mask_group, other=0.0)

    qnorm_sq = tl.sum(c_0*c_0, 0) + tl.sum(c_1*c_1, 0) + tl.sum(c_2*c_2, 0) + tl.sum(c_3*c_3, 0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    packed = ((idx_3 & 0x03) << 6) | ((idx_2 & 0x03) << 4) | ((idx_1 & 0x03) << 2) | (idx_0 & 0x03)
    p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_group
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_group)
    tl.store(DScaleBuffer + pool_slot * stride_ds_s + pid_h, dscale)


@triton.jit
def _fused_pack_store_4bit_kv_kernel(
    Y,              # (2*tokens, heads, dim) float32 — [K_y; V_y] contiguous
    Norms,          # (2*tokens, heads) float32 — [K_norms; V_norms]
    Loc,            # (tokens,) int64 — pool slot indices
    KBuffer,        # (pool_size, heads, packed_dim) uint8
    VBuffer,        # (pool_size, heads, packed_dim) uint8
    KDScale,        # (pool_size, heads) bf16
    VDScale,        # (pool_size, heads) bf16
    Boundaries,
    Centroids,
    stride_y_t,
    stride_y_h,
    stride_kb_s,
    stride_kb_h,
    stride_vb_s,
    stride_vb_h,
    stride_kds,
    stride_vds,
    stride_n_t,
    TOKENS: tl.constexpr,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_half: tl.constexpr,
):
    """Batched 4-bit pack+store for K and V in a single launch (shared codebook).

    pid_t in [0, 2*TOKENS): first TOKENS programs store to KBuffer, rest to VBuffer.
    """
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    is_v = pid_t >= TOKENS
    local_t = pid_t - TOKENS * tl.where(is_v, 1, 0)
    pool_slot = tl.load(Loc + local_t)

    offs_pair = tl.arange(0, BLOCK_PACKED)
    mask_pair = offs_pair < Lk_half

    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_even = tl.load(y_base + offs_pair * 2, mask=mask_pair, other=0.0)
    y_odd = tl.load(y_base + offs_pair * 2 + 1, mask=mask_pair, other=0.0)

    idx_even = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_odd = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_even += tl.where(y_even > bound, 1, 0).to(tl.int32)
        idx_odd += tl.where(y_odd > bound, 1, 0).to(tl.int32)

    c_even = tl.load(Centroids + idx_even, mask=mask_pair, other=0.0)
    c_odd = tl.load(Centroids + idx_odd, mask=mask_pair, other=0.0)
    qnorm_sq = tl.sum(c_even * c_even, axis=0) + tl.sum(c_odd * c_odd, axis=0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    packed = ((idx_odd & 0xF) << 4) | (idx_even & 0xF)

    # Select output buffer: KBuffer or VBuffer
    k_p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_pair
    v_p_ptr = VBuffer + pool_slot * stride_vb_s + pid_h * stride_vb_h + offs_pair
    p_ptr = tl.where(is_v, v_p_ptr, k_p_ptr)
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_pair)

    k_ds_ptr = KDScale + pool_slot * stride_kds + pid_h
    v_ds_ptr = VDScale + pool_slot * stride_vds + pid_h
    ds_ptr = tl.where(is_v, v_ds_ptr, k_ds_ptr)
    tl.store(ds_ptr, dscale)


@triton.jit
def _fused_pack_store_2bit_kv_kernel(
    Y, Norms, Loc,
    KBuffer, VBuffer,
    KDScale, VDScale,
    Boundaries, Centroids,
    stride_y_t, stride_y_h,
    stride_kb_s, stride_kb_h,
    stride_vb_s, stride_vb_h,
    stride_kds, stride_vds,
    stride_n_t,
    TOKENS: tl.constexpr,
    N_BOUNDARIES: tl.constexpr,
    BLOCK_PACKED: tl.constexpr,
    Lk_quarter: tl.constexpr,
):
    """Batched 2-bit pack+store for K and V in a single launch (shared codebook)."""
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    is_v = pid_t >= TOKENS
    local_t = pid_t - TOKENS * tl.where(is_v, 1, 0)
    pool_slot = tl.load(Loc + local_t)

    offs_group = tl.arange(0, BLOCK_PACKED)
    mask_group = offs_group < Lk_quarter

    y_base = Y + pid_t * stride_y_t + pid_h * stride_y_h
    y_0 = tl.load(y_base + offs_group * 4, mask=mask_group, other=0.0)
    y_1 = tl.load(y_base + offs_group * 4 + 1, mask=mask_group, other=0.0)
    y_2 = tl.load(y_base + offs_group * 4 + 2, mask=mask_group, other=0.0)
    y_3 = tl.load(y_base + offs_group * 4 + 3, mask=mask_group, other=0.0)

    idx_0 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_1 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_2 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    idx_3 = tl.zeros([BLOCK_PACKED], dtype=tl.int32)
    for b in tl.static_range(N_BOUNDARIES):
        bound = tl.load(Boundaries + b)
        idx_0 += tl.where(y_0 > bound, 1, 0).to(tl.int32)
        idx_1 += tl.where(y_1 > bound, 1, 0).to(tl.int32)
        idx_2 += tl.where(y_2 > bound, 1, 0).to(tl.int32)
        idx_3 += tl.where(y_3 > bound, 1, 0).to(tl.int32)

    c_0 = tl.load(Centroids + idx_0, mask=mask_group, other=0.0)
    c_1 = tl.load(Centroids + idx_1, mask=mask_group, other=0.0)
    c_2 = tl.load(Centroids + idx_2, mask=mask_group, other=0.0)
    c_3 = tl.load(Centroids + idx_3, mask=mask_group, other=0.0)

    qnorm_sq = tl.sum(c_0*c_0, 0) + tl.sum(c_1*c_1, 0) + tl.sum(c_2*c_2, 0) + tl.sum(c_3*c_3, 0)
    qnorm = tl.sqrt(qnorm_sq)
    norm = tl.load(Norms + pid_t * stride_n_t + pid_h)
    safe_qnorm = tl.where(qnorm > 1e-10, qnorm, 1.0)
    dscale = (norm / safe_qnorm).to(tl.bfloat16)

    packed = ((idx_3 & 0x03) << 6) | ((idx_2 & 0x03) << 4) | ((idx_1 & 0x03) << 2) | (idx_0 & 0x03)

    k_p_ptr = KBuffer + pool_slot * stride_kb_s + pid_h * stride_kb_h + offs_group
    v_p_ptr = VBuffer + pool_slot * stride_vb_s + pid_h * stride_vb_h + offs_group
    p_ptr = tl.where(is_v, v_p_ptr, k_p_ptr)
    tl.store(p_ptr, packed.to(tl.uint8), mask=mask_group)

    k_ds_ptr = KDScale + pool_slot * stride_kds + pid_h
    v_ds_ptr = VDScale + pool_slot * stride_vds + pid_h
    ds_ptr = tl.where(is_v, v_ds_ptr, k_ds_ptr)
    tl.store(ds_ptr, dscale)


def fused_turboquant_quantize_and_store(
    x, signs1, signs2, centroids, boundaries, bit_width,
    kv_buffer, dscale_buffer, loc,
    pre_unit=None,
    pre_norms=None,
    pre_y=None,
    rope_src=None,
    rope_buffer=None,
):
    """Fused quantize + scatter store: norm → normalize → WHT → pack+dscale → scatter to KV pool.

    Eliminates temp tensors and scatter store kernels.
    """
    import torch
    from sglang.kernels.ops.quantization.hadamard import hadamard_transform_with_signs

    tokens, heads, dim = x.shape
    use_workspace = (
        pre_unit is not None
        and pre_norms is not None
        and pre_y is not None
        and pre_unit.dim() == 3
        and pre_norms.dim() == 2
        and pre_y.dim() == 3
        and pre_unit.shape[0] >= tokens
        and pre_unit.shape[1] >= heads
        and pre_unit.shape[2] >= dim
        and pre_norms.shape[0] >= tokens
        and pre_norms.shape[1] >= heads
        and pre_y.shape[0] >= tokens
        and pre_y.shape[1] >= heads
        and pre_y.shape[2] >= dim
        and pre_unit.dtype == torch.float32
        and pre_norms.dtype == torch.float32
        and pre_y.dtype == torch.float32
        and pre_unit.device == x.device
        and pre_norms.device == x.device
        and pre_y.device == x.device
    )

    # Step 1: Fused norm + normalize (1 Triton kernel)
    BLOCK_DIM = triton.next_power_of_2(dim)
    if use_workspace:
        x_unit = pre_unit[:tokens, :heads, :dim]
        norms = pre_norms[:tokens, :heads]
        y_out = pre_y[:tokens, :heads, :dim]
    else:
        x_unit = torch.empty(tokens, heads, dim, dtype=torch.float32, device=x.device)
        norms = torch.empty(tokens, heads, dtype=torch.float32, device=x.device)
        y_out = None

    grid_nn = (tokens, heads)
    _fused_norm_normalize_kernel[grid_nn](
        x, x_unit, norms,
        x.stride(0), x.stride(1),
        x_unit.stride(0), x_unit.stride(1),
        norms.stride(0),
        BLOCK_DIM=BLOCK_DIM, Lk=dim, num_warps=4,
    )

    # Step 2: Fused WHT rotation (1 CUDA kernel)
    wht_scale = 1.0 / (dim ** 0.5)
    y = hadamard_transform_with_signs(
        x_unit, signs1, signs2, scale=wht_scale, out=y_out
    )

    # Step 3: Fused pack + dscale + scatter store (1 Triton kernel)
    store_rope = rope_src is not None and rope_buffer is not None
    if (rope_src is None) != (rope_buffer is None):
        raise ValueError("rope_src and rope_buffer must be provided together")
    if store_rope and bit_width != 4:
        raise ValueError("Fused RoPE write is currently supported only for 4-bit")

    if bit_width == 4:
        packed_dim = dim // 2
        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        rope_dim = rope_src.shape[-1] if store_rope else 0
        if store_rope and rope_dim > packed_dim:
            raise ValueError(
                f"Fused RoPE write requires rope_dim <= packed_dim, got "
                f"{rope_dim} > {packed_dim}"
            )
        grid_ps = (tokens, heads)
        _fused_pack_store_4bit_kernel[grid_ps](
            y, norms, loc,
            kv_buffer, dscale_buffer,
            rope_src if store_rope else x,
            rope_buffer if store_rope else kv_buffer,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            kv_buffer.stride(0), kv_buffer.stride(1),
            dscale_buffer.stride(0),
            rope_src.stride(0) if store_rope else 0,
            rope_src.stride(1) if store_rope else 0,
            rope_buffer.stride(0) if store_rope else 0,
            rope_buffer.stride(1) if store_rope else 0,
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_half=packed_dim,
            ROPE_DIM=rope_dim,
            STORE_ROPE=store_rope,
            num_warps=4,
        )
    elif bit_width == 2:
        packed_dim = dim // 4
        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        grid_ps = (tokens, heads)
        _fused_pack_store_2bit_kernel[grid_ps](
            y, norms, loc,
            kv_buffer, dscale_buffer,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            kv_buffer.stride(0), kv_buffer.stride(1),
            dscale_buffer.stride(0),
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_quarter=packed_dim,
            num_warps=4,
        )
    else:
        raise ValueError(f'Unsupported bit_width: {bit_width}')


def fused_native_e2m1_mla_quantize_and_store(
    x,
    signs1,
    signs2,
    boundaries,
    levels,
    codes,
    grid,
    packed_buffer,
    dscale_buffer,
    loc,
    rope_src,
    rope_buffer,
    *,
    pre_unit,
    pre_norms,
    pre_y,
):
    """Write the complete native-E2M1 MLA row with bounded scratch storage.

    The persistent representation is deliberately different from stock
    ``turboquant_4bit``: nibbles are hardware E2M1 codes, the scale follows
    the accepted N8 operation order, and RoPE is stored as E4M3 FP8.  Scratch
    capacity is fixed by ``pre_*``; larger writes are processed in chunks and
    never allocate capacity- or batch-sized temporaries.

    ``loc`` is expected to contain one physical destination per source row.
    Unsorted locations are supported.  As with SGLang's other scatter writers,
    concurrent duplicate destinations must carry identical rows; conflicting
    duplicate writes have no ordering contract.
    """
    import torch
    from sglang.kernels.ops.quantization.hadamard import (
        hadamard_transform_with_signs,
    )

    expected_dim = 512
    expected_rope_dim = 64
    if x.dim() != 3 or x.shape[1:] != (1, expected_dim):
        raise ValueError(
            "Native E2M1 MLA writer expects x shaped (tokens, 1, 512), "
            f"got {tuple(x.shape)}."
        )
    if rope_src.dim() != 3 or rope_src.shape != (
        x.shape[0],
        1,
        expected_rope_dim,
    ):
        raise ValueError(
            "Native E2M1 MLA writer expects rope_src shaped (tokens, 1, 64), "
            f"got {tuple(rope_src.shape)}."
        )
    if loc.dim() != 1 or loc.numel() != x.shape[0]:
        raise ValueError(
            "Native E2M1 MLA writer requires one flat location per token, "
            f"got loc={tuple(loc.shape)} for {x.shape[0]} tokens."
        )
    if loc.dtype not in (torch.int32, torch.int64):
        raise TypeError(
            f"Native E2M1 MLA locations must be int32 or int64, got {loc.dtype}."
        )
    if x.dtype != torch.bfloat16 or rope_src.dtype != torch.bfloat16:
        raise TypeError(
            "Native E2M1 MLA writer requires BF16 latent and RoPE inputs, "
            f"got x={x.dtype}, rope={rope_src.dtype}."
        )
    if x.stride(-1) != 1 or rope_src.stride(-1) != 1:
        raise ValueError(
            "Native E2M1 MLA writer requires contiguous feature dimensions."
        )
    if loc.stride(0) != 1:
        raise ValueError("Native E2M1 MLA locations must be contiguous.")

    device = x.device
    named_tensors = {
        "signs1": signs1,
        "signs2": signs2,
        "boundaries": boundaries,
        "levels": levels,
        "codes": codes,
        "packed_buffer": packed_buffer,
        "dscale_buffer": dscale_buffer,
        "loc": loc,
        "rope_src": rope_src,
        "rope_buffer": rope_buffer,
        "pre_unit": pre_unit,
        "pre_norms": pre_norms,
        "pre_y": pre_y,
    }
    for name, tensor in named_tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"Native E2M1 MLA {name} is on {tensor.device}, expected {device}."
            )

    if signs1.shape != (expected_dim,) or signs2.shape != (expected_dim,):
        raise ValueError("Native E2M1 MLA signs must each have shape (512,).")
    if boundaries.shape != (14,) or levels.shape != (15,) or codes.shape != (15,):
        raise ValueError(
            "Native E2M1 MLA requires 14 boundaries, 15 levels, and 15 codes."
        )
    for name, tensor in (
        ("signs1", signs1),
        ("signs2", signs2),
        ("boundaries", boundaries),
        ("levels", levels),
    ):
        if tensor.dtype != torch.float32 or not tensor.is_contiguous():
            raise TypeError(
                f"Native E2M1 MLA {name} must be contiguous float32, "
                f"got dtype={tensor.dtype}, contiguous={tensor.is_contiguous()}."
            )
    if codes.dtype != torch.uint8:
        raise TypeError(f"Native E2M1 MLA codes must be uint8, got {codes.dtype}.")
    if not codes.is_contiguous():
        raise ValueError("Native E2M1 MLA codes must be contiguous.")
    if packed_buffer.dim() != 3 or packed_buffer.shape[1:] != (1, 256):
        raise ValueError(
            "Native E2M1 MLA packed_buffer must have shape (pool, 1, 256)."
        )
    if packed_buffer.dtype != torch.uint8:
        raise TypeError(
            f"Native E2M1 MLA packed_buffer must be uint8, got {packed_buffer.dtype}."
        )
    if packed_buffer.stride(-1) != 1:
        raise ValueError(
            "Native E2M1 MLA packed_buffer requires a contiguous feature dimension."
        )
    if dscale_buffer.shape != packed_buffer.shape[:2]:
        raise ValueError(
            "Native E2M1 MLA dscale_buffer must have shape (pool, 1)."
        )
    if dscale_buffer.dtype != torch.bfloat16:
        raise TypeError(
            f"Native E2M1 MLA dscale_buffer must be BF16, got {dscale_buffer.dtype}."
        )
    if rope_buffer.shape != (
        packed_buffer.shape[0],
        1,
        expected_rope_dim,
    ):
        raise ValueError(
            "Native E2M1 MLA rope_buffer must have shape (pool, 1, 64)."
        )
    if rope_buffer.dtype != torch.float8_e4m3fn:
        raise TypeError(
            "Native E2M1 MLA rope_buffer must be float8_e4m3fn, "
            f"got {rope_buffer.dtype}."
        )
    if rope_buffer.stride(-1) != 1:
        raise ValueError(
            "Native E2M1 MLA rope_buffer requires a contiguous feature dimension."
        )

    if pre_unit.dim() != 3 or pre_unit.shape[1:] != (1, expected_dim):
        raise ValueError("pre_unit must have shape (workspace_tokens, 1, 512).")
    if pre_y.shape != pre_unit.shape:
        raise ValueError("pre_y must have the same shape as pre_unit.")
    if pre_norms.shape != pre_unit.shape[:2]:
        raise ValueError("pre_norms must have shape (workspace_tokens, 1).")
    if any(t.dtype != torch.float32 for t in (pre_unit, pre_norms, pre_y)):
        raise TypeError("Native E2M1 MLA writer workspaces must be float32.")
    if not all(t.is_contiguous() for t in (pre_unit, pre_norms, pre_y)):
        raise ValueError("Native E2M1 MLA writer workspaces must be contiguous.")
    workspace_tokens = pre_unit.shape[0]
    if workspace_tokens <= 0:
        raise ValueError(
            "Native E2M1 MLA writer workspace must hold at least one token."
        )

    tokens = x.shape[0]
    if tokens == 0:
        return

    block_dim = triton.next_power_of_2(expected_dim)
    block_packed = triton.next_power_of_2(expected_dim // 2)
    wht_scale = 1.0 / (expected_dim**0.5)
    for begin in range(0, tokens, workspace_tokens):
        end = min(begin + workspace_tokens, tokens)
        chunk_tokens = end - begin
        x_chunk = x[begin:end]
        rope_chunk = rope_src[begin:end]
        loc_chunk = loc[begin:end]
        x_unit = pre_unit[:chunk_tokens]
        norms = pre_norms[:chunk_tokens]
        y_out = pre_y[:chunk_tokens]

        _fused_norm_normalize_kernel[(chunk_tokens, 1)](
            x_chunk,
            x_unit,
            norms,
            x_chunk.stride(0),
            x_chunk.stride(1),
            x_unit.stride(0),
            x_unit.stride(1),
            norms.stride(0),
            BLOCK_DIM=block_dim,
            Lk=expected_dim,
            num_warps=4,
        )
        rotated = hadamard_transform_with_signs(
            x_unit, signs1, signs2, scale=wht_scale, out=y_out
        )
        _fused_pack_store_native_e2m1_mla_kernel[(chunk_tokens, 1)](
            rotated,
            norms,
            loc_chunk,
            packed_buffer,
            dscale_buffer,
            rope_chunk,
            rope_buffer,
            boundaries,
            levels,
            codes,
            rotated.stride(0),
            rotated.stride(1),
            packed_buffer.stride(0),
            packed_buffer.stride(1),
            dscale_buffer.stride(0),
            rope_chunk.stride(0),
            rope_chunk.stride(1),
            rope_buffer.stride(0),
            rope_buffer.stride(1),
            norms.stride(0),
            GRID=float(grid),
            BLOCK_PACKED=block_packed,
            LORA_HALF=expected_dim // 2,
            ROPE_DIM=expected_rope_dim,
            num_warps=4,
        )


def fused_turboquant_quantize_and_store_kv(
    cache_k, cache_v,
    signs1, signs2,
    k_centroids, k_boundaries, k_bit_width,
    v_centroids, v_boundaries, v_bit_width,
    k_buffer, k_dscale_buffer,
    v_buffer, v_dscale_buffer,
    loc,
    pre_kv_unit=None, pre_kv_norms=None,
):
    """Batched K+V quantize: shares norm+normalize, WHT, and pack+store launches.

    3 kernel launches total (when K/V share bit_width):
    1. Batched norm+normalize for [K, V] (1 Triton kernel)
    2. Batched WHT rotation for [K_unit, V_unit] (1 CUDA kernel)
    3. Batched pack+store for K and V (1 Triton kernel)
    """
    import torch
    from sglang.kernels.ops.quantization.hadamard import hadamard_transform_with_signs

    tokens, heads, dim = cache_k.shape
    BLOCK_DIM = triton.next_power_of_2(dim)
    wht_scale = 1.0 / (dim ** 0.5)

    # Use pre-allocated buffers if available (avoids torch.empty inside CUDA graph)
    if pre_kv_unit is not None and pre_kv_unit.shape[0] >= 2 * tokens:
        kv_unit = pre_kv_unit[:2 * tokens, :heads, :dim]
        kv_norms = pre_kv_norms[:2 * tokens, :heads]
    else:
        kv_unit = torch.empty(2 * tokens, heads, dim, dtype=torch.float32, device=cache_k.device)
        kv_norms = torch.empty(2 * tokens, heads, dtype=torch.float32, device=cache_k.device)

    # Step 1: Batched norm+normalize K and V (1 Triton kernel)
    grid_nn = (2 * tokens, heads)
    _fused_norm_normalize_kv_kernel[grid_nn](
        cache_k, cache_v, kv_unit, kv_norms,
        cache_k.stride(0), cache_k.stride(1),
        cache_v.stride(0), cache_v.stride(1),
        kv_unit.stride(0), kv_unit.stride(1),
        kv_norms.stride(0),
        TOKENS=tokens,
        BLOCK_DIM=BLOCK_DIM, Lk=dim, num_warps=4,
    )

    # Step 2: Batched WHT for K+V together (1 CUDA kernel)
    kv_y = hadamard_transform_with_signs(kv_unit, signs1, signs2, scale=wht_scale)

    # Step 3: Batched pack+store (1 Triton kernel when K/V share bit_width)
    if k_bit_width == v_bit_width:
        if k_bit_width == 4:
            packed_dim = dim // 2
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (2 * tokens, heads)
            _fused_pack_store_4bit_kv_kernel[grid_ps](
                kv_y, kv_norms, loc,
                k_buffer, v_buffer,
                k_dscale_buffer, v_dscale_buffer,
                k_boundaries, k_centroids,
                kv_y.stride(0), kv_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                k_dscale_buffer.stride(0), v_dscale_buffer.stride(0),
                kv_norms.stride(0),
                TOKENS=tokens,
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_half=packed_dim, num_warps=4,
            )
        elif k_bit_width == 2:
            packed_dim = dim // 4
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (2 * tokens, heads)
            _fused_pack_store_2bit_kv_kernel[grid_ps](
                kv_y, kv_norms, loc,
                k_buffer, v_buffer,
                k_dscale_buffer, v_dscale_buffer,
                k_boundaries, k_centroids,
                kv_y.stride(0), kv_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                k_dscale_buffer.stride(0), v_dscale_buffer.stride(0),
                kv_norms.stride(0),
                TOKENS=tokens,
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_quarter=packed_dim, num_warps=4,
            )
    else:
        # Asymmetric bit widths: fall back to separate K and V pack+store launches
        k_y = kv_y[:tokens]
        v_y = kv_y[tokens:]
        k_norms = kv_norms[:tokens]
        v_norms = kv_norms[tokens:]

        if k_bit_width == 4:
            packed_dim = dim // 2
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_4bit_kernel[grid_ps](
                k_y, k_norms, loc,
                k_buffer, k_dscale_buffer,
                cache_k, k_buffer,
                k_boundaries, k_centroids,
                k_y.stride(0), k_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                k_dscale_buffer.stride(0),
                0, 0, 0, 0,
                k_norms.stride(0),
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_half=packed_dim,
                ROPE_DIM=0, STORE_ROPE=False, num_warps=4,
            )
        elif k_bit_width == 2:
            packed_dim = dim // 4
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_2bit_kernel[grid_ps](
                k_y, k_norms, loc,
                k_buffer, k_dscale_buffer,
                k_boundaries, k_centroids,
                k_y.stride(0), k_y.stride(1),
                k_buffer.stride(0), k_buffer.stride(1),
                k_dscale_buffer.stride(0), k_norms.stride(0),
                N_BOUNDARIES=k_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_quarter=packed_dim, num_warps=4,
            )

        if v_bit_width == 4:
            packed_dim = dim // 2
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_4bit_kernel[grid_ps](
                v_y, v_norms, loc,
                v_buffer, v_dscale_buffer,
                cache_v, v_buffer,
                v_boundaries, v_centroids,
                v_y.stride(0), v_y.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                v_dscale_buffer.stride(0),
                0, 0, 0, 0,
                v_norms.stride(0),
                N_BOUNDARIES=v_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_half=packed_dim,
                ROPE_DIM=0, STORE_ROPE=False, num_warps=4,
            )
        elif v_bit_width == 2:
            packed_dim = dim // 4
            BLOCK_PACKED = triton.next_power_of_2(packed_dim)
            grid_ps = (tokens, heads)
            _fused_pack_store_2bit_kernel[grid_ps](
                v_y, v_norms, loc,
                v_buffer, v_dscale_buffer,
                v_boundaries, v_centroids,
                v_y.stride(0), v_y.stride(1),
                v_buffer.stride(0), v_buffer.stride(1),
                v_dscale_buffer.stride(0), v_norms.stride(0),
                N_BOUNDARIES=v_boundaries.shape[0],
                BLOCK_PACKED=BLOCK_PACKED, Lk_quarter=packed_dim, num_warps=4,
            )


def fused_turboquant_quantize(x, signs1, signs2, centroids, boundaries, bit_width):
    """Fused TurboQuant quantize: WHT (CUDA) + pack (Triton).

    Returns: (packed, norms, quant_norms) — same interface as batched_quantize.
    """
    import torch

    tokens, heads, dim = x.shape

    # --- PyTorch ops (norm + normalize + WHT) ---
    from sglang.kernels.ops.quantization.hadamard import hadamard_transform_with_signs

    # Fused norm + normalize: 1 Triton kernel instead of 3 PyTorch ops
    BLOCK_DIM = triton.next_power_of_2(dim)
    x_unit = torch.empty(tokens, heads, dim, dtype=torch.float32, device=x.device)
    norms = torch.empty(tokens, heads, dtype=torch.float32, device=x.device)
    grid = (tokens, heads)
    _fused_norm_normalize_kernel[grid](
        x, x_unit, norms,
        x.stride(0), x.stride(1),
        x_unit.stride(0), x_unit.stride(1),
        norms.stride(0),
        BLOCK_DIM=BLOCK_DIM,
        Lk=dim,
        num_warps=4,
    )

    wht_scale = 1.0 / (dim ** 0.5)
    y = hadamard_transform_with_signs(x_unit, signs1, signs2, scale=wht_scale)

    # --- Fused Triton kernel (searchsorted + gather + qnorm + pack): 1 launch ---
    if bit_width == 4:
        packed_dim = dim // 2
        packed = torch.empty(tokens, heads, packed_dim, dtype=torch.uint8, device=x.device)
        dscale = torch.empty(tokens, heads, dtype=torch.bfloat16, device=x.device)

        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        grid = (tokens, heads)
        _fused_pack_4bit_kernel[grid](
            y, packed, dscale, norms,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            packed.stride(0), packed.stride(1),
            dscale.stride(0),
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_half=packed_dim,
            num_warps=4,
        )
        return packed, dscale
    elif bit_width == 2:
        packed_dim = dim // 4
        packed = torch.empty(tokens, heads, packed_dim, dtype=torch.uint8, device=x.device)
        dscale = torch.empty(tokens, heads, dtype=torch.bfloat16, device=x.device)

        BLOCK_PACKED = triton.next_power_of_2(packed_dim)
        grid = (tokens, heads)
        _fused_pack_2bit_kernel[grid](
            y, packed, dscale, norms,
            boundaries, centroids,
            y.stride(0), y.stride(1),
            packed.stride(0), packed.stride(1),
            dscale.stride(0),
            norms.stride(0),
            N_BOUNDARIES=boundaries.shape[0],
            BLOCK_PACKED=BLOCK_PACKED,
            Lk_quarter=packed_dim,
            num_warps=4,
        )
        return packed, dscale
    else:
        raise ValueError(f"Unsupported bit_width: {bit_width}. Only 2 and 4 are supported.")
