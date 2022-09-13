from dataclasses import dataclass

@dataclass
class DeviceMesh(object):
    """
    Describes the device mesh.
    Args:
        num_row(int): Number of mesh row.
        num_col(int): Number of mesh column. num_row * num_col = num_gpu.
    """
    __slots__ = ['num_row', 'num_col']

    num_row: int = 0
    num_col: int = 0