import torch

a = torch.tensor([0,1,-1,0,1])
a[a == 0] = 1
print(a)
