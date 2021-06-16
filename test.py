import torch
import pdb

a = torch.tensor([[[5, 6], [7, 8], [9, 10]], [[1, 2], [11, 3], [4, 12]]])
out = torch.max(a, dim=1).values
pdb.set_trace()