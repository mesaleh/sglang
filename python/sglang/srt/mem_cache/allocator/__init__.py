"""Token-to-KV-slot allocators. One file per allocation strategy."""

from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.paged import (
    PagedTokenToKVPoolAllocator,
    StaticTieredPagedTokenToKVPoolAllocator,
    alloc_extend_naive,
)
from sglang.srt.mem_cache.allocator.token import TokenToKVPoolAllocator

__all__ = [
    "BaseTokenToKVPoolAllocator",
    "PagedTokenToKVPoolAllocator",
    "StaticTieredPagedTokenToKVPoolAllocator",
    "TokenToKVPoolAllocator",
    "alloc_extend_naive",
]
