# Distributed A · B = C
# A: SparseTensor | B: DenseTensor | C: DenseTensor

from .tensor import DenseTensor, SparseTensor
from .scheduler import DistributedSPMMScheduler
from .distributed_manager import DistributedSPMMManager
from .hardware_monitor import HardwareManager

def distributed_spmm(A:SparseTensor, B:DenseTensor, hw_manager:HardwareManager) -> DenseTensor:
    # 1. Analysis the best distributed policy through given A & B & Hardware.
    num_gpu = hw_manager.get_num_gpu()

    scheduler = DistributedSPMMScheduler(A, B, num_gpu)
    scheduler.analysis_policy()
    distributed_policy = scheduler.get_recommended_policy()

    # 2. Arrange A & B into distributed nodes.
    dist_spmm_manager = DistributedSPMMManager(distributed_policy, A, B)
    dist_spmm_manager.pre()

    # 3. Do local spmm.
    dist_spmm_manager.run_task()

    # 4. Merge the result.
    dist_spmm_manager.post()

    C = dist_spmm_manager.get_res()

    return C


