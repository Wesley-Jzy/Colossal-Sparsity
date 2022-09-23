from sparsity.spmm import distributed_spmm
from sparsity.tensor import SparseTensor, DenseTensor

from colossalai.utils import free_port
from functools import partial

import torch
import colossalai
import torch.multiprocessing as mp

x = torch.tensor([
        [1, 2, 3],
        [4, 5, 6]
    ])

y = torch.tensor([
        [7, 8],
        [9, 10],
        [11, 12]
    ])

def run_spmm(world_size):
    A = SparseTensor(x)
    B = DenseTensor(y)
    hw_manager = HardwareManager(num_gpu=world_size)
    C = distributed_spmm(A, B, hw_manager)

def run_dist(rank, world_size, port):
    print(f"rank: {rank} initialized.")
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_spmm(world_size)

def test_spmm(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_spmm(4)