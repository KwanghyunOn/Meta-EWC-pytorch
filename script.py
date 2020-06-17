import os
import random
import numpy as np

n = 5
alpha = 10 ** np.linspace(-2, -0.5, num=n)
print(alpha)

for i in range(n):
    print(f"Test alpha = {alpha[i]:.3f}")
    os.system(f"python test_meta.py --alpha={alpha[i]:.3f} --name=meta-4 --n=1")
