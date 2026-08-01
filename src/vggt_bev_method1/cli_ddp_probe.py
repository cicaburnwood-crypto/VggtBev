from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist

from vggt_bev_method1.nccl import configure_and_validate_nccl


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    require_same_numa = os.environ.get(
        "METHOD1_REQUIRE_SAME_NUMA",
        "1",
    ).strip().lower() not in ("0", "false", "no")
    preflight = configure_and_validate_nccl(
        world_size=world_size,
        require_same_numa=require_same_numa,
        p2p_level="AUTO",
    )
    dist.init_process_group("nccl", device_id=device)
    value = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(value)
    identity = torch.cuda.get_device_properties(local_rank)
    print(
        json.dumps(
            {
                "rank": rank,
                "local_rank": local_rank,
                "device_name": identity.name,
                "device_uuid": str(identity.uuid),
                "all_reduce_sum": float(value.cpu()),
                "nccl_preflight": preflight,
            }
        ),
        flush=True,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
