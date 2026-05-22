from prompt_attack.utils.distributed import DistributedContext


def test_distributed_context_shards_records_by_rank() -> None:
    rank0 = DistributedContext(rank=0, local_rank=0, world_size=2, device="cpu", backend="gloo")
    rank1 = DistributedContext(rank=1, local_rank=1, world_size=2, device="cpu", backend="gloo")

    assert rank0.shard([0, 1, 2, 3, 4]) == [0, 2, 4]
    assert rank1.shard([0, 1, 2, 3, 4]) == [1, 3]
