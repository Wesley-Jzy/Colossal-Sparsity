class _DimShard(object):
    def __init__(self, offset = 0, len = 0):
        self.offset = offset
        self.len = len

class Shard(object):
    def __init__(self):
        self._dims = {}

    def get_shard(self, dim = 0):
        return self._dims[dim]

class Shard_1D(Shard):
    def __init__(self, dim = 0, offset = 0, len = 0):
        self._dims = {}
        self._dims[dim] = _DimShard(offset = offset, len = len)

    def get_shard(self, dim = 0):
        return self._dims[dim]