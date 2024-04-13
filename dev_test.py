from policies import ContiniousPolicy001
from history import MyHistory
import torch
import numpy as np

h = [MyHistory({"ol": torch.randn(16), "op": torch.randn(3)}) for _ in range(10)]

p = ContiniousPolicy001()
ol, op = h[0].get_vectors()

ol = torch.tensor(np.reshape(ol, (1,4,16)))
op = torch.tensor(np.reshape(op, (1, 3)))
# raise
print(p.forward(ol, op))