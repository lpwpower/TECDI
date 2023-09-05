from cdt.metrics import SID
from numpy.random import randint
import numpy as np
tar = np.triu(randint(2, size=(10, 10)))
pred = np.triu(randint(2, size=(10, 10)))
print(SID(tar, pred))


import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())



