from dataclasses import dataclass
from sparsity.tensor import DenseTensor, SparseTensor
from sparsity.distributed_policy import DistributedSPMMPolicy, TestPolicy, BStationary1DPolicy

import math

@dataclass
class _PolicyCandidate(object):
    policy:DistributedSPMMPolicy
    memory_cost:float = 0.0
    time_cost:float = 0.0

    def __str__(self):
        return f'PolicyCandidate: [{str(self.policy)} | memory {self.memory_cost} | time {self.time_cost}]'

class DistributedSPMMScheduler(object):
    def __init__(self, A:SparseTensor, B:DenseTensor, num_gpu:int):
        self._policy_candidates = []
        self._num_gpu = num_gpu
        self._A = A
        self._B = B
    
    def analysis_policy(self):
        # TODO Add real analysis code
        row_dim = int(math.sqrt(self._num_gpu))
        col_dim = row_dim
        policy = BStationary1DPolicy(row_dim, col_dim, self._A.get_dim_size(dim=0), self._A.get_dim_size(dim=1), self._B.get_dim_size(dim=1))
        candidate = _PolicyCandidate(policy, policy.eval_memory_storage(), policy.eval_computation_cost())
        self._policy_candidates.append(candidate)
        print(str(_PolicyCandidate))

    def get_recommended_policy(self) -> DistributedSPMMPolicy:
        return self._policy_candidates[0].policy