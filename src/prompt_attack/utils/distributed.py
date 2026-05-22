"""Small helpers for optional single-node distributed execution."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class DistributedContext:
    """Runtime distributed state derived from torchrun environment variables."""

    rank: int
    local_rank: int
    world_size: int
    device: str
    backend: str

    @classmethod
    def from_env(cls, requested_device: str) -> "DistributedContext":
        """Create and initialize distributed state when torchrun is active."""
        import torch
        import torch.distributed as dist

        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if world_size <= 1:
            return cls(
                rank=0,
                local_rank=0,
                world_size=1,
                device=requested_device,
                backend="none",
            )

        if requested_device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = f"cuda:{local_rank}"
            backend = "nccl"
        else:
            device = requested_device
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend)
        return cls(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            backend=backend,
        )

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_rank0(self) -> bool:
        return self.rank == 0

    def shard(self, records: Sequence[T]) -> list[T]:
        """Return this rank's deterministic shard of a sequence."""
        if not self.is_distributed:
            return list(records)
        return list(records[self.rank :: self.world_size])

    def barrier(self) -> None:
        """Synchronize ranks when distributed execution is active."""
        if not self.is_distributed:
            return
        import torch.distributed as dist

        dist.barrier()

    def broadcast_tensor(self, tensor) -> None:
        """Broadcast a tensor in-place from rank 0."""
        if not self.is_distributed:
            return
        import torch.distributed as dist

        dist.broadcast(tensor, src=0)

    def all_reduce_sum(self, tensor) -> None:
        """Sum a tensor in-place across ranks."""
        if not self.is_distributed:
            return
        import torch.distributed as dist

        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def close(self) -> None:
        """Tear down the process group if this process initialized one."""
        if not self.is_distributed:
            return
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
