import types
import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.layers import flashinfer_comm_fusion as fusion
from sglang.srt.layers import communicator
from sglang.srt.layers import layernorm
from sglang.srt.runtime_context import get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-h100")
register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-b200")
register_cuda_ci(est_time=30, stage="base-c", runner_config="4-gpu-gb300")


class _FakeWorkspace:
    def __init__(self, backend, world_size):
        self.backend = backend
        self.world_size = world_size

    def is_buffer_size_sufficient(self, **_kwargs):
        return True


class _FakeFlashInferComm:
    class AllReduceFusionPattern:
        kARResidualRMSNorm = object()

    def __init__(self):
        self.calls = []

    def create_allreduce_fusion_workspace(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeWorkspace(kwargs["backend"], kwargs["world_size"])

    def allreduce_fusion(
        self,
        *,
        input,
        workspace,
        residual_out,
        norm_out,
        residual_in,
        rms_gamma,
        rms_eps,
        **_kwargs,
    ):
        allreduced = input * workspace.world_size
        expected_residual = allreduced + residual_in
        variance = expected_residual.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        expected_norm = (
            expected_residual.to(torch.float32)
            * torch.rsqrt(variance + rms_eps)
            * rms_gamma.to(torch.float32)
        ).to(input.dtype)
        residual_out.copy_(expected_residual)
        norm_out.copy_(expected_norm)


def _torch_allreduce_residual_rmsnorm_baseline(
    input_tensor, residual, weight, world_size, eps
):
    allreduced = input_tensor * world_size
    residual_out = allreduced + residual
    variance = residual_out.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    norm_out = (
        residual_out.to(torch.float32)
        * torch.rsqrt(variance + eps)
        * weight.to(torch.float32)
    ).to(input_tensor.dtype)
    return norm_out, residual_out


class TestFlashInferCommFusion(unittest.TestCase):
    def test_prepare_attn_fallback_restores_deferred_pre_allreduce_addition(self):
        layer_communicator = communicator.LayerCommunicator.__new__(
            communicator.LayerCommunicator
        )
        norm_out = torch.tensor([[11.0, 12.0]])
        residual_out = torch.tensor([[13.0, 14.0]])
        layer_communicator.input_layernorm = MagicMock(
            return_value=(norm_out, residual_out)
        )
        layer_communicator._context = object()
        layer_communicator.qkv_latent_func = None
        layer_communicator._communicate_simple_fn = MagicMock(
            side_effect=lambda hidden_states, **_kwargs: hidden_states
        )
        hidden_states = torch.tensor([[1.0, 2.0]])
        pre_add = torch.tensor([[3.0, 4.0]])
        residual = torch.tensor([[5.0, 6.0]])
        hidden_states._sglang_needs_allreduce_fusion = True
        hidden_states._sglang_pre_allreduce_addition = pre_add

        with (
            patch.object(
                communicator,
                "get_attn_tp_context",
                return_value=types.SimpleNamespace(input_scattered=False),
            ),
            patch.object(
                communicator,
                "apply_flashinfer_allreduce_fusion",
                return_value=False,
            ),
            patch.object(
                communicator,
                "apply_aiter_all_reduce_fusion",
                return_value=False,
            ),
            patch.object(
                communicator,
                "moe_tensor_model_parallel_all_reduce",
                side_effect=lambda value: value * 4,
            ) as all_reduce,
        ):
            actual_norm, actual_residual = layer_communicator.prepare_attn(
                hidden_states,
                residual,
                forward_batch=MagicMock(),
            )

        torch.testing.assert_close(all_reduce.call_args.args[0], hidden_states + pre_add)
        layer_communicator.input_layernorm.assert_called_once()
        reduced, norm_residual = layer_communicator.input_layernorm.call_args.args
        torch.testing.assert_close(reduced, (hidden_states + pre_add) * 4)
        torch.testing.assert_close(norm_residual, residual)
        self.assertIs(actual_norm, norm_out)
        self.assertIs(actual_residual, residual_out)

    def test_legacy_pre_allreduce_add_capability_is_backend_specific(self):
        trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=1
        )
        mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=1
        )

        with (
            patch.object(
                fusion, "_flashinfer_allreduce_supports_pre_allreduce_add", True
            ),
            patch.object(
                fusion,
                "_flashinfer_comm",
                types.SimpleNamespace(),
            ),
            patch.object(fusion, "is_sm100_supported", return_value=True),
        ):
            self.assertTrue(fusion.supports_flashinfer_pre_allreduce_add(trtllm))
            self.assertFalse(fusion.supports_flashinfer_pre_allreduce_add(mnnvl))

        with patch.object(
            fusion, "_flashinfer_allreduce_supports_pre_allreduce_add", False
        ):
            self.assertFalse(fusion.supports_flashinfer_pre_allreduce_add(trtllm))

    def test_advertised_mnnvl_pre_allreduce_add_capability(self):
        mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=2
        )
        advertised_comm = types.SimpleNamespace(
            AllReduceFusionPattern=types.SimpleNamespace(
                kARResidualRMSNorm=object()
            ),
            supports_pre_allreduce_add=lambda backend, _pattern: backend == "mnnvl",
        )

        with (
            patch.object(
                fusion, "_flashinfer_allreduce_supports_pre_allreduce_add", True
            ),
            patch.object(fusion, "_flashinfer_comm", advertised_comm),
            patch.object(fusion, "is_sm100_supported", return_value=True),
        ):
            self.assertTrue(fusion.supports_flashinfer_pre_allreduce_add(mnnvl))

    def test_declined_fusion_preserves_pre_allreduce_addition(self):
        norm = types.SimpleNamespace(
            variance_epsilon=1e-6,
            forward=MagicMock(return_value=(object(), object())),
        )
        x = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])
        post_add = torch.tensor([[5.0, 6.0]])
        pre_add = torch.tensor([[7.0, 8.0]])

        def fake_moe_all_reduce(value):
            return value * 4

        with (
            patch.object(layernorm, "_use_aiter", False),
            patch.object(
                fusion,
                "flashinfer_allreduce_residual_rmsnorm",
                return_value=(None, None),
            ),
            patch(
                "sglang.srt.distributed.moe_tensor_model_parallel_all_reduce",
                side_effect=fake_moe_all_reduce,
            ) as all_reduce,
            get_parallel().override(moe_ep_size=1, moe_tp_size=4),
        ):
            result = layernorm._forward_with_allreduce_fusion(
                norm_module=norm,
                x=x,
                residual=residual,
                post_residual_addition=post_add,
                weight=torch.ones(2),
                use_attn_tp_group=False,
                pre_allreduce_addition=pre_add,
            )

        self.assertEqual(result, norm.forward.return_value)
        torch.testing.assert_close(all_reduce.call_args.args[0], x + pre_add)
        reduced, adjusted_residual, deferred_post = norm.forward.call_args.args
        torch.testing.assert_close(reduced, (x + pre_add) * 4)
        torch.testing.assert_close(adjusted_residual, residual + post_add)
        self.assertIsNone(deferred_post)

    def test_full_tp_uses_torch_dist_for_rendezvous_but_default_runtime_group(self):
        device_group = object()
        cpu_group = object()
        created_backends = []

        class _FakeTorchDistBackend:
            def __init__(self, device_group, cpu_group):
                self.device_group = device_group
                self.cpu_group = cpu_group
                created_backends.append(self)

        fake_comm = _FakeFlashInferComm()
        manager = fusion.FlashInferWorkspaceManager()
        with (
            patch.object(fusion, "_flashinfer_comm", fake_comm),
            patch.object(
                fusion,
                "_create_allreduce_fusion_workspace",
                fake_comm.create_allreduce_fusion_workspace,
            ),
            patch.object(fusion, "_TorchDistBackend", _FakeTorchDistBackend),
            patch.object(fusion, "_flashinfer_create_workspace_supports_group", True),
            patch.object(
                fusion, "_flashinfer_create_workspace_supports_comm_backend", True
            ),
            patch.object(
                fusion, "_preflight_check_workspace_memory", return_value=True
            ) as preflight,
            patch.object(
                fusion, "in_the_same_node_as", return_value=[True, True, True, True]
            ),
        ):
            manager.initialize(
                world_size=4,
                rank=0,
                max_token_num=8,
                hidden_dim=16,
                backend="mnnvl",
                device_group=None,
                cpu_group=None,
                comm_device_group=device_group,
                comm_cpu_group=cpu_group,
            )

        self.assertEqual(len(created_backends), 1)
        self.assertIs(created_backends[0].device_group, device_group)
        self.assertIs(created_backends[0].cpu_group, cpu_group)
        preflight.assert_called_once_with(
            world_size=4,
            max_token_num=8,
            hidden_dim=16,
            dtype=None,
            cpu_group=cpu_group,
        )
        self.assertIs(fake_comm.calls[0]["comm_backend"], created_backends[0])
        self.assertIsNone(fake_comm.calls[0]["group"])
        self.assertEqual(fake_comm.calls[0]["gpus_per_node"], 4)
        self.assertEqual(manager.group, (None, None))

    def test_full_tp_keeps_groups_for_workspace_rendezvous(self):
        device_group = object()
        cpu_group = object()
        coordinator = types.SimpleNamespace(
            device_group=device_group,
            cpu_group=cpu_group,
            world_size=4,
        )
        manager = MagicMock()
        manager.initialized = False

        with (
            patch.object(fusion, "_flashinfer_allreduce_unavailable", False),
            patch.object(fusion, "is_flashinfer_available", return_value=True),
            patch.object(fusion, "_flashinfer_comm", object()),
            patch.object(fusion, "get_attn_tp_group", return_value=coordinator),
            patch.object(fusion, "get_tp_group", return_value=coordinator),
            patch.object(
                fusion, "get_global_server_args", return_value=types.SimpleNamespace()
            ),
            patch.object(fusion, "_get_workspace_manager", return_value=manager),
            patch.object(
                fusion,
                "resolve_flashinfer_allreduce_fusion_backend",
                return_value="mnnvl",
            ),
            patch.object(fusion, "_sync_allreduce_unavailable_across_tp"),
            get_parallel().override(attn_tp_size=4, attn_tp_rank=0),
        ):
            fusion.ensure_workspace_initialized()

        manager.initialize.assert_called_once()
        kwargs = manager.initialize.call_args.kwargs
        self.assertIsNone(kwargs["device_group"])
        self.assertIsNone(kwargs["cpu_group"])
        self.assertIs(kwargs["comm_device_group"], device_group)
        self.assertIs(kwargs["comm_cpu_group"], cpu_group)

    def test_auto_backend_resolves_by_arch(self):
        single_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=1
        )
        multi_node = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="auto", nnodes=2
        )

        # Blackwell: mnnvl on both single-node and multi-node.
        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node), "mnnvl"
            )

        # SM90: auto uses trtllm on single-node, multi-node is unsupported.
        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

        # Architectures outside SM90/SM10X are unsupported. Both pre-SM90
        # and post-SM10X devices (e.g. SM120) must fail closed.
        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(single_node)
                with self.assertRaises(ValueError):
                    fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node)

    def test_explicit_backend_validation(self):
        single_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=1
        )
        multi_node_mnnvl = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="mnnvl", nnodes=2
        )
        single_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=1
        )
        multi_node_trtllm = types.SimpleNamespace(
            flashinfer_allreduce_fusion_backend="trtllm", nnodes=2
        )

        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_mnnvl),
                "mnnvl",
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(single_node_trtllm),
                "trtllm",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl)
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        with patch.object(fusion, "is_sm100_supported", return_value=True):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_mnnvl),
                "mnnvl",
            )
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        with (
            patch.object(fusion, "is_sm100_supported", return_value=True),
            fusion.envs.SGLANG_FLASHINFER_TRTLLM_MULTINODE.override(True),
        ):
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(
                    multi_node_trtllm
                ),
                "trtllm",
            )
            auto = types.SimpleNamespace(
                flashinfer_allreduce_fusion_backend="auto", nnodes=2
            )
            self.assertEqual(
                fusion.resolve_flashinfer_allreduce_fusion_backend(auto), "mnnvl"
            )

        with (
            patch.object(fusion, "is_sm100_supported", return_value=False),
            patch.object(fusion, "is_sm90_supported", return_value=True),
            fusion.envs.SGLANG_FLASHINFER_TRTLLM_MULTINODE.override(True),
        ):
            with self.assertRaises(ValueError):
                fusion.resolve_flashinfer_allreduce_fusion_backend(multi_node_trtllm)

        for arch in ("pre_sm90", "post_sm10x"):
            with (
                self.subTest(arch=arch),
                patch.object(fusion, "is_sm100_supported", return_value=False),
                patch.object(fusion, "is_sm90_supported", return_value=False),
            ):
                for args in (
                    single_node_mnnvl,
                    multi_node_mnnvl,
                    single_node_trtllm,
                    multi_node_trtllm,
                ):
                    with self.subTest(backend=args.flashinfer_allreduce_fusion_backend):
                        with self.assertRaises(ValueError):
                            fusion.resolve_flashinfer_allreduce_fusion_backend(args)

    def test_allreduce_fusion_backends_match_torch_baseline(self):
        fake_comm = _FakeFlashInferComm()
        original_comm = fusion._flashinfer_comm
        original_create = fusion._create_allreduce_fusion_workspace
        original_manager = fusion._attn_tp_workspace_manager
        original_unavailable = fusion._flashinfer_allreduce_unavailable
        try:
            fusion._flashinfer_comm = fake_comm
            fusion._create_allreduce_fusion_workspace = (
                fake_comm.create_allreduce_fusion_workspace
            )
            fusion._flashinfer_allreduce_unavailable = False

            for backend in ("trtllm", "mnnvl"):
                with self.subTest(backend=backend):
                    world_size = 4
                    manager = fusion.FlashInferWorkspaceManager()
                    manager.workspace = _FakeWorkspace(backend, world_size)
                    manager.initialized = True
                    fusion._attn_tp_workspace_manager = manager
                    if not torch.cuda.is_available():
                        self.skipTest("FlashInfer allreduce custom op is CUDA-only")
                    device = torch.device("cuda")
                    torch.manual_seed(0)
                    input_tensor = torch.randn(4, 8, dtype=torch.float32, device=device)
                    residual = torch.randn(4, 8, dtype=torch.float32, device=device)
                    weight = torch.randn(8, dtype=torch.float32, device=device)
                    eps = 1e-6

                    expected_norm, expected_residual = (
                        _torch_allreduce_residual_rmsnorm_baseline(
                            input_tensor, residual, weight, world_size, eps
                        )
                    )

                    with (
                        patch.object(
                            fusion, "is_flashinfer_available", return_value=True
                        ),
                        get_parallel().override(attn_tp_size=world_size),
                        patch.object(
                            fusion, "ensure_workspace_initialized", return_value=True
                        ),
                    ):
                        norm_out, residual_out = (
                            fusion.flashinfer_allreduce_residual_rmsnorm(
                                input_tensor=input_tensor,
                                residual=residual,
                                weight=weight,
                                eps=eps,
                                max_token_num=8,
                            )
                        )

                    torch.testing.assert_close(norm_out, expected_norm)
                    torch.testing.assert_close(residual_out, expected_residual)
        finally:
            fusion._flashinfer_comm = original_comm
            fusion._create_allreduce_fusion_workspace = original_create
            fusion._attn_tp_workspace_manager = original_manager
            fusion._flashinfer_allreduce_unavailable = original_unavailable

    def test_mnnvl_workspace_declines_pre_allreduce_add(self):
        if not torch.cuda.is_available():
            self.skipTest("FlashInfer allreduce custom op is CUDA-only")
        fake_comm = MagicMock()
        fake_comm.AllReduceFusionPattern.kARResidualRMSNorm = object()
        fake_comm.supports_pre_allreduce_add = None
        manager = types.SimpleNamespace(
            workspace=_FakeWorkspace("mnnvl", 4), initialized=True
        )
        input_tensor = torch.ones(2, 4, device="cuda")

        with (
            patch.object(fusion, "is_flashinfer_available", return_value=True),
            patch.object(fusion, "_flashinfer_comm", fake_comm),
            patch.object(
                fusion, "_flashinfer_allreduce_supports_pre_allreduce_add", True
            ),
            patch.object(fusion, "ensure_workspace_initialized", return_value=True),
            patch.object(fusion, "_get_workspace_manager", return_value=manager),
            get_parallel().override(attn_tp_size=4),
        ):
            result = fusion.flashinfer_allreduce_residual_rmsnorm(
                input_tensor=input_tensor,
                residual=torch.ones_like(input_tensor),
                weight=torch.ones(4, device="cuda"),
                pre_allreduce_addition=torch.ones_like(input_tensor),
            )

        self.assertEqual(result, (None, None))
        fake_comm.allreduce_fusion.assert_not_called()

    def test_mnnvl_workspace_accepts_advertised_pre_allreduce_add(self):
        if not torch.cuda.is_available():
            self.skipTest("FlashInfer allreduce custom op is CUDA-only")
        fake_comm = MagicMock()
        fake_comm.AllReduceFusionPattern.kARResidualRMSNorm = object()
        fake_comm.supports_pre_allreduce_add.side_effect = (
            lambda backend, _pattern: backend == "mnnvl"
        )
        manager = types.SimpleNamespace(
            workspace=_FakeWorkspace("mnnvl", 4), initialized=True
        )
        input_tensor = torch.ones(2, 4, device="cuda")
        pre_add = torch.full_like(input_tensor, 2)

        with (
            patch.object(fusion, "is_flashinfer_available", return_value=True),
            patch.object(fusion, "_flashinfer_comm", fake_comm),
            patch.object(
                fusion, "_flashinfer_allreduce_supports_pre_allreduce_add", True
            ),
            patch.object(fusion, "ensure_workspace_initialized", return_value=True),
            patch.object(fusion, "_get_workspace_manager", return_value=manager),
            get_parallel().override(attn_tp_size=4),
        ):
            fusion.flashinfer_allreduce_residual_rmsnorm(
                input_tensor=input_tensor,
                residual=torch.ones_like(input_tensor),
                weight=torch.ones(4, device="cuda"),
                pre_allreduce_addition=pre_add,
            )

        self.assertIs(
            fake_comm.allreduce_fusion.call_args.kwargs["pre_allreduce_add"],
            pre_add,
        )


if __name__ == "__main__":
    unittest.main()
