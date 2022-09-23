import torch
from sparsity.device_mesh import DeviceMesh
from sparsity.shard import Shard

class GlobalTensorSpec(object):
    def __init__(self, device_mesh:DeviceMesh=None):
        self._device_mesh = device_mesh
        self._shards = []

    def add_shard(self, shard:Shard):
        self._shards.append(shard)

class Tensor(object):
    def __init__(self, tensor, device_mesh:DeviceMesh=None):
        self._global = GlobalTensorSpec(device_mesh)
        self._local_tensor = tensor

    def get_dim_size(self, dim = 0):
        return self._local_tensor.size(dim = dim)

DenseTensor = Tensor
SparseTensor = Tensor

#class DenseTensor(Tensor):
#    pass

#class SparseTensor(Tensor):
#    pass