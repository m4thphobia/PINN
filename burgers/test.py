import numpy as np

x_ = np.linspace(-1, 1, 3)
t_ = np.linspace(0, 1, 3)
X, T = np.meshgrid(x_, t_, indexing='ij')
x_flat = X.flatten()
t_flat = T.flatten()

print(f'x_flat:{x_flat}')
print(f'x_flat.shape:{x_flat.shape}')


