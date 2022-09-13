from dataclasses import dataclass, field
from typing import List

@dataclass
class GPUNode(object):
    pass

@dataclass
class GPUResource(object):
    nodes: List[GPUNode] = field(default_factory=list)