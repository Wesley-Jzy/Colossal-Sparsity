# Distributed A · B = C
# A: SparseTensor | B: DenseTensor | C: DenseTensor

from sparsity.tensor import DenseTensor, SparseTensor
from sparsity.scheduler import DistributedSPMMScheduler
from sparsity.distributed_manager import DistributedSPMMManager
from sparsity.hardware_manager import HardwareManager

def distributed_spmm(A:SparseTensor, B:DenseTensor, hw_manager:HardwareManager) -> DenseTensor:
    # 1. Analysis the best distributed policy through given A & B & Hardware.
    num_gpu = hw_manager.get_num_gpu()

    scheduler = DistributedSPMMScheduler(A, B, num_gpu)
    scheduler.analysis_policy()
    distributed_policy = scheduler.get_recommended_policy()

    # 2. Manage distributed input Tensor A B & output Tensor C.
    dist_spmm_manager = DistributedSPMMManager(distributed_policy)
    dist_spmm_manager.start_distribution()

    dist_A = distributed_policy.distribute_A(A, device_row_index = 0, device_col_index = 0)
    dist_B = distributed_policy.distribute_B(B, device_row_index = 0, device_col_index = 0)
    dist_C = distributed_policy.distribute_C(device_row_index = 0, device_col_index = 0)
    # 3. Do distributed spmm.
    distributed_policy.do_spmm(dist_A, dist_B, dist_C, device_row_index = 0, device_col_index = 0)
    C = distributed_policy.collect(dist_C, device_row_index = 0, device_col_index = 0)

    dist_spmm_manager.end_distribution()
    print('distributed_spmm finished.')

    return C


