from .device_mesh import DeviceMesh
from .tensor import DenseTensor, SparseTensor

# Sparse A(m x n) · Dense B(n x k) = Dense C(m x k)
class DistributedSPMMPolicy(object):
    def __init__(self, device_num_row, device_num_col):
        self._device_mesh = DeviceMesh(device_num_row, device_num_col)

    def distribute(self, device_row_index = 0, device_col_index = 0, A:SparseTensor, B:DenseTensor, C:DenseTensor):
        self._check_device_index()
        self._distribute_A(A)
        self._distribute_B(B)
        self._distribute_C(C)

    def do_spmm(self, device_row_index = 0, device_col_index = 0, A:SparseTensor, B:DenseTensor, C:DenseTensor):
        pass

    def collect(self, device_row_index = 0, device_col_index = 0, A:SparseTensor, B:DenseTensor, C:DenseTensor):
        pass

    def eval_memory_storage(self):
        pass

    def eval_computation_cost(self):
        pass

    def eval_communication_cost(self):
        pass

    def _check_device_index(self, device_row_index, device_col_index):
        assert device_row_index < self._device_mesh.num_row and device_col_index < self._device_mesh.num_col \
            and device_row_index >= 0 and device_col_index >= 0, \
            'Device index should be among the range of device mesh.'
    
    def _distribute_A(self, A:SparseTensor):
        pass
    
    def _distribute_B(self, B:DenseTensor):
        pass
    
    def _distribute_C(self, C:DenseTensor):
        pass

# Do spmm in the first node only to test the system
class TestPolicy(DistributedSPMMPolicy):
    def __init__(self, device_num_row, device_num_col):
        self._device_mesh = DeviceMesh(device_num_row, device_num_col)

    def __str__(self):
        return self.__class__.__name__

    def do_spmm(self, device_row_index = 0, device_col_index = 0, A:SparseTensor, B:DenseTensor, C:DenseTensor):
        return

    def collect(self, device_row_index = 0, device_col_index = 0, A:SparseTensor, B:DenseTensor, C:DenseTensor):
        return

    def eval_memory_storage(self):
        return 10.0

    def eval_computation_cost(self):
        return 10.0

    def eval_communication_cost(self):
        return 10.0
    
    def _distribute_A(self, A:SparseTensor):
        return
    
    def _distribute_B(self, B:DenseTensor):
        return
    
    def _distribute_C(self, C:DenseTensor):
        return