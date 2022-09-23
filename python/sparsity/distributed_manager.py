from .distributed_policy import DistributedSPMMPolicy
from .tensor import DenseTensor, SparseTensor

class DistributedSPMMManager(object):
    def __init__(self, dist_policy:DistributedSPMMPolicy):
        pass

    def start_distribution(self):
        pass
    
    def end_distribution(self):
        pass