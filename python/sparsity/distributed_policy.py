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

    def do_spmm(self, A:SparseTensor, B:DenseTensor, C:DenseTensor, device_row_index = 0, device_col_index = 0):
        pass

    def collect(self, C:DenseTensor, device_row_index = 0, device_col_index = 0):
        pass

    def eval_memory_storage(self):
        pass

    def eval_computation_cost(self):
        pass

    def eval_communication_cost(self):
        pass
    
    def distribute_A(self, A:SparseTensor, device_row_index = 0, device_col_index = 0) -> SparseTensor:
        pass
    
    def distribute_B(self, B:DenseTensor, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        pass
    
    def distribute_C(self, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        pass

# Do spmm in the first node only to test the system
class TestPolicy(DistributedSPMMPolicy):
    def __init__(self, device_num_row, device_num_col, m, n, k):
        super(TestPolicy, self).__init__(device_num_row, device_num_col, m, n, k)

    def __str__(self):
        return self.__class__.__name__

    def do_spmm(self, A:SparseTensor, B:DenseTensor, C:DenseTensor, device_row_index = 0, device_col_index = 0):
        C._local_tensor = torch.matmul(A._local_tensor, B._local_tensor)
        return

    def collect(self, C:DenseTensor, device_row_index = 0, device_col_index = 0):
        return C

    def eval_memory_storage(self):
        return 10.0

    def eval_computation_cost(self):
        return 10.0

    def eval_communication_cost(self):
        return 10.0
    
    def distribute_A(self, A:SparseTensor, device_row_index = 0, device_col_index = 0) -> SparseTensor:
        return A
    
    def distribute_B(self, B:DenseTensor, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        return B
    
    def distribute_C(self, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        return DenseTensor(torch.empty(self._m, self._k), self._device_mesh)

class BStationary1DPolicy(DistributedSPMMPolicy):
    def __init__(self, device_num_row, device_num_col, m, n, k):
        super(TestPolicy, self).__init__(device_num_row, device_num_col, m, n, k)

    def __str__(self):
        return self.__class__.__name__

    def do_spmm(self, A:SparseTensor, B:DenseTensor, C:DenseTensor, device_row_index = 0, device_col_index = 0):
        C._local_tensor = torch.matmul(A._local_tensor, B._local_tensor)
        return

    def collect(self, C:DenseTensor, device_row_index = 0, device_col_index = 0):
        return C

    def eval_memory_storage(self):
        return 10.0

    def eval_computation_cost(self):
        return 10.0

    def eval_communication_cost(self):
        return 10.0
    
    def distribute_A(self, A:SparseTensor, device_row_index = 0, device_col_index = 0) -> SparseTensor:
        num_row_shard = self._device_mesh.num_row * self._device_mesh.num_col
        assert (self._m * self._n) % num_row_shard == 0

        row_len = (self._m * self._n) / num_row_shard
        row_offset = device_row_index * self._device_mesh.num_col + device_col_index


        return A
    
    def distribute_B(self, B:DenseTensor, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        return B
    
    def distribute_C(self, device_row_index = 0, device_col_index = 0) -> DenseTensor:
        return DenseTensor(torch.empty(self._m, self._k), self._device_mesh)