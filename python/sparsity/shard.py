class DimShard(object):
    def __init__(self, offset = 0, len = 0):
        self.offset = []
        self.len = []

class Shard(object):
    def __init__(self):
        self._dims = {}

    def is_sharded(self):
        pass

    def set_dimshards(self, dim, dimshard:DimShard):
        pass

    def get_dimshards(self, dim):
        pass

class Shard_1D(Shard):
    def __init__(self):
        self._dims = {}
    
    def is_sharded(self):
        return len(len(self._dims) > 1)

    def add_dimshards(self, dim, dimshard:DimShard):
        if not self._dims.has_key(dim):
            self._dims[dim] = []
        self._dims[dim].append(dimshard)

    def get_dimshards(self, dim):
        return self._dims[dim]