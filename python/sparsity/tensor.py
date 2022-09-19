import torch

class DenseTensor(torch.Tensor):
    def __new__(cls, data: torch.Tensor, spec: ColoTensorSpec) -> 'ColoTensor':
        if data is None:
            data = torch.empty(0)
        return torch.Tensor._make_subclass(cls, data, data.requires_grad)

    def __init__(self, data: torch.Tensor, spec: Optional[ColoTensorSpec] = None) -> None:
        # If not set spec, use a DP process group and replicate dist spec
        if spec is None:
            self.has_initialized = False
            self.dist_spec = ReplicaSpec()
            self.compute_spec = None
            self.process_group = ProcessGroup()
        else:
            self.has_initialized = True
            self.dist_spec = spec.dist_attr
            self.compute_spec = spec.compute_attr
            if spec.pg is None:
                self.process_group = ProcessGroup()
            else:
                self.process_group = spec.pg

        self._type = TensorType.NONMODEL
        self._graph_node = None

class SparseTensor:
    pass