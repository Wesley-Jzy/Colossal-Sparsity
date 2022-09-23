from sparsity.device_mesh import DeviceMesh
from sparsity.tensor import DenseTensor, SparseTensor
import torch

# Sparse A(m x n) · Dense B(n x k) = Dense C(m x k)
class DistributedSPMMPolicy(object):
    def __init__(self, device_num_row, device_num_col, m, n, k):
        self._device_mesh = DeviceMesh(device_num_row, device_num_col)
        self._m = m
        self._n = n
        self._k = k

    def do_spmm(self, device_row_index = 0, device_col_index = 0, A:SparseTensor, B:DenseTensor, C:DenseTensor):
        pass

    def collect(self, device_row_index = 0, device_col_index = 0, C:DenseTensor):
        pass

    def eval_memory_storage(self):
        pass

    def eval_computation_cost(self):
        pass

    def eval_communication_cost(self):
        pass
    
    def distribute_A(self, A:SparseTensor, device_row_index = 0, device_col_index = 0) -> SparseTensor:
        pass
    
    def distribute_B(self, B:DenseTensordevice_row_index = 0, device_col_index = 0) -> DenseTensor:
        pass
    
    def distribute_C(self, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        pass

# Do spmm in the first node only to test the system
class TestPolicy(DistributedSPMMPolicy):
    def __init__(self, device_num_row, device_num_col, m, n, k):
        super(TestPolicy, self).__init__(device_num_row, device_num_col, m, n, k)

    def __str__(self):
        return self.__class__.__name__

    def do_spmm(self, device_row_index = 0, device_col_index = 0, A:SparseTensor, B:DenseTensor, C:DenseTensor):
        C._local_tensor = torch.matmul(A._local_tensor, B._local_tensor)
        return

    def collect(self, device_row_index = 0, device_col_index = 0, C:DenseTensor):
        return C

    def eval_memory_storage(self):
        return 10.0

    def eval_computation_cost(self):
        return 10.0

    def eval_communication_cost(self):
        return 10.0
    
    def distribute_A(self, A:SparseTensor, device_row_index = 0, device_col_index = 0) -> SparseTensor:
        return A
    
    def distribute_B(self, B:DenseTensordevice_row_index = 0, device_col_index = 0) -> DenseTensor:
        return B
    
    def distribute_C(self, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        return DenseTensor(torch.empty(self._m, self._k), self._device_mesh)