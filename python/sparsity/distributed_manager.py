from .distributed_policy import DistributedSPMMPolicy
from .tensor import DenseTensor, SparseTensor

class DistributedSPMMManager(object):
    def __init__(self, dist_policy:DistributedSPMMPolicy,
                    A:SparseTensor, B:DenseTensor):
        pass

    def pre(self):
        pass
    
    def run_task(self):
        pass
    
    def post(self):
        pass
    
    def get_res(self):
        pass
