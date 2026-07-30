from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.utils import GenerationBatchResult


def test_generation_result_h43_sticky_status_fails_closed():
    result = GenerationBatchResult(
        tq_mla_fault_status=torch.zeros(1, dtype=torch.int32)
    )
    result.raise_for_tq_mla_fault()

    result.tq_mla_fault_status.fill_(1)
    with pytest.raises(RuntimeError, match="coordinated rank restart"):
        result.raise_for_tq_mla_fault()


def test_scheduler_attaches_rank_local_h43_status():
    status = torch.zeros(1, dtype=torch.int32)
    scheduler = object.__new__(Scheduler)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(
        get_kvcache=lambda: SimpleNamespace(tq_mla_frontend_fault_status=status)
    )
    result = GenerationBatchResult()

    scheduler._attach_tq_mla_fault_status(result)

    assert result.tq_mla_fault_status is status
