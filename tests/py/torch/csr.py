import torch

values = torch.tensor([2,4,5])
col_indices = torch.tensor([2,0,1])
row_ptr = torch.tensor([0,1,2,3])

csr = torch.sparse_csr_tensor(row_ptr, col_indices, values, size=(3,4), dtype=torch.double)
print(csr)

dense = csr.to_dense()
print(dense)

a = torch.tensor([[0, 0, 1, 0], [1, 2, 0, 0], [0, 0, 0, 0]], dtype = torch.float64)
sp = a.to_sparse_csr()
#print(sp)
#print(sp.to_dense())