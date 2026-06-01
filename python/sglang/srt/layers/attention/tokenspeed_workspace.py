"""Dependency-light helpers for TokenSpeed MLA workspace sizing."""

# Workspace upper bound for tokenspeed_mla_decode:
#   B * H_eff * S_eff * split_kv * (kv_lora_rank + 1) * sizeof(float32)
# TokenSpeed's split planner keeps B * split_kv <= num_sms, so an upper bound is:
#   num_sms * H_eff * S_eff * (kv_lora_rank + 1) * sizeof(float32)
# For Kimi's small-head tree verify path, TokenSpeed folds q chunks of 8 tokens:
# q16/q24/q32 become H_eff=128, S_eff=2/3/4. In all supported modes,
# H_eff * S_eff is bounded by num_heads * raw_q_len, with a floor of the
# standard Blackwell M tile (128 rows).
_TOKENSPEED_MIN_M_ROWS = 128


def tokenspeed_workspace_bytes(
    num_sms: int, num_heads: int, kv_lora_rank: int, q_len: int
) -> int:
    query_rows = max(_TOKENSPEED_MIN_M_ROWS, num_heads * max(1, q_len))
    return num_sms * query_rows * (kv_lora_rank + 1) * 4
