class DeviceMesh(object):
    """
    Describes the device mesh.
    Args:
        num_row(int): Number of mesh row.
        num_col(int): Number of mesh column. num_row * num_col = num_gpu.
    """
    def __init__(self, device_num_row = 0, device_num_col = 0):
        self.num_row: int = device_num_row
        self.num_col: int = device_num_col

    def check_device_index(self, device_row_index, device_col_index):
        assert device_row_index < self.num_row and device_col_index < self.num_col \
            and device_row_index >= 0 and device_col_index >= 0, \
            'Device index should be among the range of device mesh.'