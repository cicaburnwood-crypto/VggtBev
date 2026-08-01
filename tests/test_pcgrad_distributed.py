from __future__ import annotations

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from vggt_bev_method1.cli_train_metric import (
    _all_reduce_gradients,
    _apply_gradient_correction,
    _direct_priority_gradient_correction,
)


class _ToyHead(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor([1.0, 1.0]))

    def forward(self) -> torch.Tensor:
        return self.value


def _distributed_pcgrad_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
) -> None:
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
    )
    try:
        head = DistributedDataParallel(_ToyHead())
        parameter = head.module.value
        magnitude = float(rank + 1)
        with head.no_sync():
            value = head()
            direct_loss = magnitude * value[0]
            guessed_loss = magnitude * (-value[0] + value[1])
            correction, metrics = _direct_priority_gradient_correction(
                direct_loss,
                guessed_loss,
                [parameter],
            )
            (direct_loss + guessed_loss).backward()
        _all_reduce_gradients([parameter])
        _apply_gradient_correction([parameter], correction)

        # Rank-averaged direct=[1.5,0], guess=[-1.5,1.5]. Projection preserves
        # direct and removes only guess's opposing x component.
        assert torch.allclose(parameter.grad, torch.tensor([1.5, 1.5]))
        assert torch.allclose(correction[0], torch.tensor([1.5, 0.0]))
        assert float(metrics["pcgrad_conflict"]) == 1.0
        assert abs(float(metrics["pcgrad_projected_cosine"])) < 1e-6
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("P1B_RUN_DISTRIBUTED_TESTS") != "1",
    reason="requires permission to create a local Gloo transport",
)
def test_distributed_pcgrad_uses_global_task_gradients() -> None:
    descriptor, rendezvous = tempfile.mkstemp(prefix="p1b-pcgrad-")
    os.close(descriptor)
    try:
        mp.spawn(
            _distributed_pcgrad_worker,
            args=(2, rendezvous),
            nprocs=2,
            join=True,
        )
    finally:
        if os.path.exists(rendezvous):
            os.unlink(rendezvous)
