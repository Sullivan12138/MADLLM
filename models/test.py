import numpy as np 
import torch
from torch.nn.functional import cosine_similarity

a = torch.arange(24)
a = a.reshape(2,3,4)
# b = torch.tensor([[[1,2],[1,3],[1,2]],
#                   [[1,2],[1,3],[2,3]]])
print(a)
c = a.expand(2, -1, -1, -1)
print(c)