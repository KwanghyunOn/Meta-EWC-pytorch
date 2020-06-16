import os
import random
import numpy as np

n = 10
lr = [0.34 for _ in range(n)]
alpha = 10 ** np.linspace(-2, 2, num=n)
print(alpha)

for i in range(n):
    print(f"Test alpha = {alpha[i]:.3f}")
    os.system(f"python test_meta.py --lr_main={lr[i]:.3f} --alpha={alpha[i]:.3f} --name=meta-test-4 --n=1")
