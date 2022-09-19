class HardwareManager(object):
    def __init__(self, num_gpu:int):
        # describe the GPU nodes
        self._num_gpu = num_gpu
    
    def get_num_gpu(self):
        return self._num_gpu
