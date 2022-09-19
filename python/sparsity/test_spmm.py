from .spmm import distributed_spmm

from colossalai.utils import free_port
from functools import partial

import colossalai
import torch.multiprocessing as mp

def run_spmm(world_size):
    A = SparseTensor()
    B = DenseTensor()
    hw_manager = HardwareManager(num_gpu=world_size)
    C = distributed_spmm(A, B, hw_manager)

def run_dist(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host='localhost', port=port, backend='nccl')
    run_spmm(world_size)

def test_gpt(world_size):
    run_func = partial(run_dist, world_size=world_size, port=free_port())
    mp.spawn(run_func, nprocs=world_size)

if __name__ == '__main__':
    test_spmm(4)