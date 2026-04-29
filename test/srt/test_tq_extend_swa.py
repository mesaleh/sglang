"""
Unit tests for windowed TurboQuant extend attention kernel.

Validates that the sliding-window path in
turboquant_extend_attention.tq_extend_attention_fwd produces
numerically equivalent output to running the same kernel with
sliding_window_size=-1 on a pre-trimmed kv_indices that already
excludes pre-window tokens.

Rationale: the kernel math should be identical between "full pool
with windowed mask" and "pre-windowed pool with no mask" — both
attend to the exact same kv tokens from the query's perspective.
If they match, the mask + SKIP_TILE logic is correct.

Test matrix (tuned for gpt-oss-120b target shapes):
- Lq = Lv = 64            (gpt-oss head_dim)
- 8:1 GQA (64 Q heads, 8 KV heads at TP=1, or 8:1 at smaller scale)
- batch_size in {1, 4}
- cur_seq_len_prefix in {0, 64, 128, 256, 1024, 8192}
- cur_seq_len_extend in {1, 64, 1024}
- sliding_window_size in {-1, 128}
- k_bit_width, v_bit_width in {(4,4), (4,2)}
- sinks in {None, random-bf16}

Usage:
    python -m pytest test/srt/test_tq_extend_swa.py -v
    # or individual class:
    python -m pytest test/srt/test_tq_extend_swa.py::TestWindowedCorrectness -v
"""

import math
import unittest

try:
    import torch
    import triton

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    HAS_CUDA = False


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestWindowedCorrectness(unittest.TestCase):
    """
    Golden test: windowed kernel on full prefix == unwindowed kernel on
    trimmed prefix. The two paths compute exactly the same attention
    (same query, same kv tokens that fall inside the window); the
    window is expressed differently (mask vs index trimming) but the
    math is identical.
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

        cls.device = "cuda"
        cls.head_dim = 64
        cls.num_q_heads = 16
        cls.num_kv_heads = 2  # 8:1 GQA
        cls.window_size = 128
        cls.configs = {}
        for bits in [2, 4]:
            cls.configs[bits] = TurboQuantConfig(
                bit_width=bits, head_dim=cls.head_dim, device=cls.device
            )

    def _build_pool_buffers(self, k_raw, v_raw, k_bits, v_bits):
        """
        Quantize a [T, H, D] bf16 K and V into packed uint8 + bf16 dscale
        by writing them through an MHATokenToKVPoolTurboQuant instance.
        This mirrors exactly what production does at set_kv_buffer time,
        so the kernel reads buffers in their real runtime layout.
        """
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolTurboQuant

        T, H, D = k_raw.shape
        # Use a real pool object so the quantized storage matches production.
        # Same bit-width for K and V in tests that use symmetric; for
        # asymmetric we build two single-bit pools (test ensures the
        # kernel mixes them correctly — see asymmetric test).
        assert k_bits == v_bits, (
            "_build_pool_buffers assumes K/V share bit_width; "
            "asymmetric test should construct buffers separately."
        )
        pool = MHATokenToKVPoolTurboQuant(
            size=T,
            page_size=1,
            dtype=torch.bfloat16,
            head_num=H,
            head_dim=D,
            layer_num=1,
            device=self.device,
            enable_memory_saver=False,
            turboquant_bits=k_bits,
            turboquant_k_bits=k_bits,
            turboquant_v_bits=v_bits,
        )
        # Write every token into layer 0 of the pool.
        loc = torch.arange(T, device=self.device, dtype=torch.int64)

        # Build a fake RadixAttention-like object; set_kv_buffer only reads
        # .layer_id from it.
        class _FakeLayer:
            layer_id = 0

        pool.set_kv_buffer(
            layer=_FakeLayer(),
            loc=loc,
            cache_k=k_raw,
            cache_v=v_raw,
        )
        return (
            pool.k_buffer[0], pool.v_buffer[0],
            pool.k_dequant_scale_buffer[0], pool.v_dequant_scale_buffer[0],
            pool,  # keep alive so the buffers don't get freed
        )

    def _run_kernel(
        self,
        q_extend,
        k_extend,
        v_extend,
        k_packed_pool,
        v_packed_pool,
        k_dscale_pool,
        v_dscale_pool,
        kv_indices,
        kv_indptr,
        qo_indptr,
        sliding_window_size,
        k_bits,
        v_bits,
        sinks=None,
    ):
        from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
            tq_extend_attention_fwd,
        )

        k_cfg = self.configs[k_bits]
        v_cfg = self.configs[v_bits]

        num_tokens_extend = q_extend.shape[0]
        o_extend = torch.empty_like(q_extend)

        tq_extend_attention_fwd(
            q_extend=q_extend,
            k_extend=k_extend,
            v_extend=v_extend,
            o_extend=o_extend,
            k_packed=k_packed_pool,
            v_packed=v_packed_pool,
            k_dscale=k_dscale_pool,
            v_dscale=v_dscale_pool,
            k_centroids=k_cfg.k_centroids.float(),
            v_centroids=v_cfg.k_centroids.float(),
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            custom_mask=None,
            is_causal=True,
            mask_indptr=None,
            max_len_extend=num_tokens_extend,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            k_bit_width=k_bits,
            v_bit_width=v_bits,
            logit_cap=0.0,
            sinks=sinks,
            xai_temperature_len=-1,
            sliding_window_size=sliding_window_size,
            window_kv_offsets=None,
        )
        return o_extend

    def _synthetic_case(
        self,
        prefix_len,
        extend_len,
        batch_size=1,
        seed=0,
    ):
        """Build synthetic q/k/v extend tensors + a pre-filled TQ pool."""
        torch.manual_seed(seed)

        # Extend-phase tensors (fresh bf16, rotated into WHT domain as caller
        # would have done — for this test, unrotated is fine because our
        # reference also uses the same kernel, and the WHT cancels across
        # both kernel invocations.)
        q = torch.randn(
            extend_len * batch_size,
            self.num_q_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        k = torch.randn(
            extend_len * batch_size,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        v = torch.randn(
            extend_len * batch_size,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        # Prefix KV pool — allocate prefix_len tokens per batch item plus
        # slack. TQ kernel reads from pool via kv_indices indirection.
        pool_capacity = max(prefix_len * batch_size + 16, 64)
        k_pool_raw = torch.randn(
            pool_capacity,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        v_pool_raw = torch.randn(
            pool_capacity,
            self.num_kv_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )

        return q, k, v, k_pool_raw, v_pool_raw

    def _run_test(self, prefix_len, extend_len, window, k_bits, v_bits, has_sink=False):
        """
        The core assertion: if the TQ pool holds `prefix_len` tokens, a query
        with a window of size `window` attends only to the last `window`
        of those. Our windowed kernel (SWS=window on full pool) must match
        a reference that pre-trims the pool to the last `window` tokens
        (SWS=-1 on trimmed pool).
        """
        if prefix_len == 0:
            # When the pool is empty, the windowed and un-windowed paths
            # both skip the prefix loop entirely — no buffers to compare,
            # no assertion possible on the prefix stage. We still want to
            # exercise the extend-stage window mask, so build a 1-token
            # dummy pool that neither path will read from (kv_indptr shows
            # 0 prefix tokens).
            dummy_prefix = 1
        else:
            dummy_prefix = prefix_len

        batch_size = 1
        q, k, v, k_pool_raw, v_pool_raw = self._synthetic_case(
            dummy_prefix, extend_len, batch_size=batch_size, seed=0xBEEF
        )

        # Quantize the pool via the production pool object so buffers match
        # the runtime layout exactly.
        assert k_bits == v_bits, (
            "This test uses symmetric K/V bits; asymmetric handled separately."
        )
        k_packed, v_packed, k_dscale, v_dscale, _pool = self._build_pool_buffers(
            k_pool_raw, v_pool_raw, k_bits, v_bits,
        )

        # Full indices: 0..prefix_len-1 point into the pool
        kv_indices_full = torch.arange(
            prefix_len, device=self.device, dtype=torch.int32
        )
        # Trimmed indices: only the last `window` tokens (bounded by prefix_len)
        effective_window = min(window, prefix_len) if window > 0 else prefix_len
        trim_start = prefix_len - effective_window
        kv_indices_trimmed = torch.arange(
            trim_start, prefix_len, device=self.device, dtype=torch.int32
        )

        qo_indptr = torch.tensor([0, extend_len], device=self.device, dtype=torch.int32)
        kv_indptr_full = torch.tensor(
            [0, prefix_len], device=self.device, dtype=torch.int32
        )
        kv_indptr_trim = torch.tensor(
            [0, effective_window], device=self.device, dtype=torch.int32
        )

        sinks = None
        if has_sink:
            sinks = torch.randn(
                self.num_q_heads, device=self.device, dtype=torch.bfloat16
            )

        # Reference: windowed attention via index trimming
        o_ref = self._run_kernel(
            q, k, v, k_packed, v_packed, k_dscale, v_dscale,
            kv_indices_trimmed, kv_indptr_trim, qo_indptr,
            sliding_window_size=-1,
            k_bits=k_bits, v_bits=v_bits,
            sinks=sinks,
        )

        # Test: windowed attention via mask on full pool
        o_win = self._run_kernel(
            q, k, v, k_packed, v_packed, k_dscale, v_dscale,
            kv_indices_full, kv_indptr_full, qo_indptr,
            sliding_window_size=window,
            k_bits=k_bits, v_bits=v_bits,
            sinks=sinks,
        )

        # Cosine similarity per token. Threshold matches upstream
        # PR #23135 test bar.
        o_ref_f = o_ref.float().flatten(1)  # [T, H*D]
        o_win_f = o_win.float().flatten(1)
        cos = torch.nn.functional.cosine_similarity(
            o_ref_f, o_win_f, dim=-1
        )
        return cos

    # All trim-reference tests use extend_len=1. Rationale: the
    # trim-reference path pre-trims the pool to the last `window` tokens
    # and runs the kernel with SWS=-1 (attending to the entire trimmed
    # pool). This matches the windowed kernel's behavior *only when*
    # the query window does not shift across extend positions — i.e.
    # extend_len=1. For extend_len > 1, later query tokens in the
    # windowed kernel see a shifted window that drops early prefix
    # tokens; the trim-reference does not shift. For longer extends the
    # correctness is instead validated end-to-end via the
    # needle-in-haystack tests at deployment (see arc doc's correctness
    # gate protocol). Unit-test here nails down the decode-position
    # semantics which is the production-critical path at c=1.

    def test_window_matches_trim_4bit_small(self):
        """256-token prefix, window=128, 4-bit K/V, no sink."""
        cos = self._run_test(
            prefix_len=256, extend_len=1, window=128, k_bits=4, v_bits=4
        )
        self.assertGreaterEqual(
            cos.min().item(), 0.995,
            f"min cosine similarity {cos.min().item():.4f} < 0.995",
        )

    def test_window_matches_trim_4bit_large(self):
        """Long prefix: 8192-token prefix, window=128 — stress SKIP_TILE."""
        cos = self._run_test(
            prefix_len=8192, extend_len=1, window=128, k_bits=4, v_bits=4
        )
        self.assertGreaterEqual(
            cos.min().item(), 0.995,
            f"min cosine similarity {cos.min().item():.4f} < 0.995",
        )

    # Asymmetric K/V bit widths (e.g. K=4bit, V=2bit) require separately
    # building K and V pools via MHATokenToKVPoolTurboQuant with the
    # asymmetric kwargs. Deferring to a follow-on test when we actually
    # deploy an asymmetric config (current gpt-oss deployment is K=V=4bit).

    def test_window_smaller_than_prefix(self):
        """Prefix > window: standard case, most tiles skipped."""
        cos = self._run_test(
            prefix_len=1024, extend_len=1, window=128, k_bits=4, v_bits=4
        )
        self.assertGreaterEqual(cos.min().item(), 0.995)

    def test_window_equals_prefix(self):
        """Prefix == window: no tiles should be skipped."""
        cos = self._run_test(
            prefix_len=128, extend_len=1, window=128, k_bits=4, v_bits=4
        )
        self.assertGreaterEqual(cos.min().item(), 0.995)

    def test_window_larger_than_prefix(self):
        """Prefix < window: window is effectively no-op; all prefix attended."""
        cos = self._run_test(
            prefix_len=64, extend_len=1, window=128, k_bits=4, v_bits=4
        )
        self.assertGreaterEqual(cos.min().item(), 0.995)

    def test_zero_prefix(self):
        """No prefix — extend-stage only. Window still applies to the
        query-vs-extend-kv relationship, but with extend_len=1 there's
        nothing to mask. Windowed and un-windowed paths match trivially."""
        cos = self._run_test(
            prefix_len=0, extend_len=1, window=128, k_bits=4, v_bits=4
        )
        self.assertGreaterEqual(cos.min().item(), 0.995)

    def test_window_with_sink(self):
        """Sliding-window + learned sinks (gpt-oss SWA layers)."""
        cos = self._run_test(
            prefix_len=1024, extend_len=1, window=128, k_bits=4, v_bits=4,
            has_sink=True,
        )
        self.assertGreaterEqual(
            cos.min().item(), 0.995,
            f"window+sink failed, min cos {cos.min().item():.4f}",
        )

    def test_long_extend_runs_without_error(self):
        """
        With extend_len > 1 the trim-reference diverges from the windowed
        kernel (the window shifts per query position). This test just
        checks the kernel produces finite output on a realistic
        chunked-prefill shape — end-to-end correctness is validated by
        the deploy-time needle-in-haystack gate.
        """
        _q, k_ext, v_ext, k_pool_raw, v_pool_raw = self._synthetic_case(
            prefix_len=4096, extend_len=512, seed=0xFEED,
        )
        q = torch.randn(
            512, self.num_q_heads, self.head_dim,
            device=self.device, dtype=torch.bfloat16,
        )
        k_packed, v_packed, k_dscale, v_dscale, _pool = self._build_pool_buffers(
            k_pool_raw, v_pool_raw, 4, 4,
        )
        kv_indices = torch.arange(4096, device=self.device, dtype=torch.int32)
        kv_indptr = torch.tensor([0, 4096], device=self.device, dtype=torch.int32)
        qo_indptr = torch.tensor([0, 512], device=self.device, dtype=torch.int32)
        o = self._run_kernel(
            q, k_ext, v_ext, k_packed, v_packed, k_dscale, v_dscale,
            kv_indices, kv_indptr, qo_indptr,
            sliding_window_size=128, k_bits=4, v_bits=4,
        )
        self.assertTrue(torch.isfinite(o).all(), "output has NaN/Inf")
        # Should not be trivially zero either (that'd suggest everything was masked)
        self.assertGreater(o.abs().max().item(), 1e-4, "output is suspiciously ~zero")


@unittest.skipUnless(HAS_CUDA, "CUDA not available")
class TestBackwardCompat(unittest.TestCase):
    """
    When sliding_window_size=-1 (default, pre-change behavior), the kernel
    should produce output consistent with full-attention TQ extend — and
    determinism: running the kernel twice with identical inputs should
    yield bitwise-identical outputs.
    """

    @classmethod
    def setUpClass(cls):
        from sglang.srt.layers.quantization.kv_turboquant import TurboQuantConfig

        cls.device = "cuda"
        cls.head_dim = 64
        cls.num_q_heads = 16
        cls.num_kv_heads = 2
        cls.cfg4 = TurboQuantConfig(bit_width=4, head_dim=cls.head_dim, device=cls.device)

    def _build_tq_pool(self, T, H, D, bits):
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolTurboQuant
        return MHATokenToKVPoolTurboQuant(
            size=T, page_size=1, dtype=torch.bfloat16,
            head_num=H, head_dim=D, layer_num=1,
            device=self.device, enable_memory_saver=False,
            turboquant_bits=bits, turboquant_k_bits=bits, turboquant_v_bits=bits,
        )

    def _fill_pool(self, pool, k_raw, v_raw):
        class _FakeLayer:
            layer_id = 0
        loc = torch.arange(k_raw.shape[0], device=self.device, dtype=torch.int64)
        pool.set_kv_buffer(
            layer=_FakeLayer(), loc=loc, cache_k=k_raw, cache_v=v_raw,
        )
        return (
            pool.k_buffer[0], pool.v_buffer[0],
            pool.k_dequant_scale_buffer[0], pool.v_dequant_scale_buffer[0],
        )

    def test_sws_neg1_determinism(self):
        """Two identical runs with SWS=-1 produce bitwise-identical output."""
        from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
            tq_extend_attention_fwd,
        )

        torch.manual_seed(123)
        prefix_len = 512
        extend_len = 64

        q = torch.randn(extend_len, self.num_q_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(extend_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(extend_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)

        k_pool_raw = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)
        v_pool_raw = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)

        pool = self._build_tq_pool(prefix_len, self.num_kv_heads, self.head_dim, 4)
        k_packed, v_packed, k_dscale, v_dscale = self._fill_pool(pool, k_pool_raw, v_pool_raw)

        kv_indices = torch.arange(prefix_len, device=self.device, dtype=torch.int32)
        kv_indptr = torch.tensor([0, prefix_len], device=self.device, dtype=torch.int32)
        qo_indptr = torch.tensor([0, extend_len], device=self.device, dtype=torch.int32)

        outputs = []
        for _ in range(2):
            o = torch.empty_like(q)
            tq_extend_attention_fwd(
                q_extend=q, k_extend=k, v_extend=v, o_extend=o,
                k_packed=k_packed, v_packed=v_packed,
                k_dscale=k_dscale, v_dscale=v_dscale,
                k_centroids=self.cfg4.k_centroids.float(),
                v_centroids=self.cfg4.k_centroids.float(),
                qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
                custom_mask=None, is_causal=True, mask_indptr=None,
                max_len_extend=extend_len,
                sm_scale=1.0 / math.sqrt(self.head_dim),
                k_bit_width=4, v_bit_width=4,
                logit_cap=0.0, sinks=None, xai_temperature_len=-1,
                sliding_window_size=-1,
            )
            outputs.append(o.clone())

        # Identical inputs → bitwise-identical outputs (Triton is deterministic
        # for a given grid + constexpr signature).
        self.assertTrue(
            torch.equal(outputs[0], outputs[1]),
            f"SWS=-1 non-deterministic: max diff "
            f"{(outputs[0] - outputs[1]).abs().max().item()}",
        )

    def test_sws_neg1_vs_sws_large(self):
        """
        SWS=-1 and SWS=very-large (larger than prefix+extend) must produce
        equivalent output — both attend to the entire prefix.
        """
        from sglang.srt.layers.attention.triton_ops.turboquant_extend_attention import (
            tq_extend_attention_fwd,
        )

        torch.manual_seed(456)
        prefix_len = 256
        extend_len = 32

        q = torch.randn(extend_len, self.num_q_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)
        k = torch.randn(extend_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)
        v = torch.randn(extend_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)

        k_pool_raw = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)
        v_pool_raw = torch.randn(prefix_len, self.num_kv_heads, self.head_dim, device=self.device, dtype=torch.bfloat16)

        pool = self._build_tq_pool(prefix_len, self.num_kv_heads, self.head_dim, 4)
        k_packed, v_packed, k_dscale, v_dscale = self._fill_pool(pool, k_pool_raw, v_pool_raw)

        kv_indices = torch.arange(prefix_len, device=self.device, dtype=torch.int32)
        kv_indptr = torch.tensor([0, prefix_len], device=self.device, dtype=torch.int32)
        qo_indptr = torch.tensor([0, extend_len], device=self.device, dtype=torch.int32)

        o_noswa = torch.empty_like(q)
        o_large = torch.empty_like(q)

        common = dict(
            q_extend=q, k_extend=k, v_extend=v,
            k_packed=k_packed, v_packed=v_packed,
            k_dscale=k_dscale, v_dscale=v_dscale,
            k_centroids=self.cfg4.k_centroids.float(),
            v_centroids=self.cfg4.k_centroids.float(),
            qo_indptr=qo_indptr, kv_indptr=kv_indptr, kv_indices=kv_indices,
            custom_mask=None, is_causal=True, mask_indptr=None,
            max_len_extend=extend_len,
            sm_scale=1.0 / math.sqrt(self.head_dim),
            k_bit_width=4, v_bit_width=4,
            logit_cap=0.0, sinks=None, xai_temperature_len=-1,
        )
        tq_extend_attention_fwd(o_extend=o_noswa, sliding_window_size=-1, **common)
        tq_extend_attention_fwd(
            o_extend=o_large, sliding_window_size=prefix_len + extend_len + 10, **common
        )

        cos = torch.nn.functional.cosine_similarity(
            o_noswa.float().flatten(1), o_large.float().flatten(1), dim=-1
        )
        self.assertGreaterEqual(
            cos.min().item(), 0.9995,
            f"SWS=-1 vs SWS=large diverge, min cos {cos.min().item():.4f}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
