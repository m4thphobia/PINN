import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import loadmat
import torch
import torch.nn as nn
import torch.optim as optim
from burgersnet import *
device = "cuda" if torch.cuda.is_available() else "cpu"


file_path = './burgers_shock.mat'
exact_data = loadmat(file_path)

## initial condition
initial_sample_size = 2000
x_initial = np.linspace(-1, 1, initial_sample_size)
X_initial = np.zeros([initial_sample_size, 2])
X_initial[:, 0] = x_initial
u_initial = -np.sin(np.pi*x_initial)

X_initial_t = torch.tensor(X_initial, requires_grad=True).float().to(device)
u_initial_t = torch.tensor(u_initial, requires_grad=True).float().to(device).unsqueeze(dim=1)

## boundary condition
x_boundary = np.array([-1.0, 1.0])
boundary_sample_size = 200
t_boundary = np.linspace(0, 1.0, boundary_sample_size)

X_boundary_m = np.zeros([boundary_sample_size, 2])
X_boundary_m[:, 0] = -1
X_boundary_m[:, 1] = t_boundary

X_boundary_p = np.zeros([boundary_sample_size, 2])
X_boundary_p[:, 0] = 1
X_boundary_p[:, 1] = t_boundary

X_boundary = np.vstack([X_boundary_m, X_boundary_p])
u_boundary = np.zeros(X_boundary.shape[0])

X_boundary_t = torch.tensor(X_boundary, requires_grad=True).float().to(device)
u_boundary_t = torch.tensor(u_boundary, requires_grad=True).float().to(device).unsqueeze(dim=1)

## sampling point
x_ = np.linspace(-1, 1, 100)
t_ = np.linspace(0, 1, 100)
X, T = np.meshgrid(x_, t_, indexing='ij')
x_flat = X.flatten()
t_flat = T.flatten()

sampling_size = 5000
random_idx = np.random.choice(np.arange(x_flat.shape[0]), size=sampling_size, replace=False)

x_sampled = x_flat[random_idx]
t_sampled = t_flat[random_idx]

X_sampled = np.zeros([sampling_size, 2])
X_sampled[:, 0] = x_sampled
X_sampled[:, 1] = t_sampled

X_sampled_t = torch.tensor(X_sampled, requires_grad=True).float().to(device)

net = PINN(activation='tanh').to(device)
optimizer = optim.Adam(net.parameters(), lr=0.0005)

episodes=9000

for epoch in range(episodes):
    optimizer.zero_grad()

    x_sampled = X_sampled_t[:, 0].unsqueeze(dim=-1) #.to(device)
    t_sampled = X_sampled_t[:, 1].unsqueeze(dim=-1) #.to(device)
    pinn_loss = pinn_loss(x_sampled, t_sampled, net)

    x_initial = X_initial_t[:, 0].unsqueeze(dim=-1) #.to(device)
    t_initial = X_initial_t[:, 1].unsqueeze(dim=-1) #.to(device)
    initial_loss = initial_condition_loss(x_initial, t_initial, net, u_initial_t)

    x_boundary = X_boundary_t[:, 0].unsqueeze(dim=-1) #.to(device)
    t_boundary = X_boundary_t[:, 1].unsqueeze(dim=-1) #.to(device)
    boundary_loss = boundary_condition_loss(x_boundary, t_boundary, net, u_boundary_t)

    loss = pinn_loss + initial_loss + boundary_loss
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        loss_ =  loss.item()
        pinn_loss_ = pinn_loss.item()
        initial_loss_ = initial_loss.item()
        boundary_loss_ = boundary_loss.item()

        print(f'epoch:{epoch:.3e}, loss:{loss_:.3e}, pinn:{pinn_loss_:.3e}, initial:{initial_loss_:.3e}, boundary:{boundary_loss_:.3e}')

if not os.path.exists("out"):
        os.makedirs("out")

SAVE_PATH = "../out/pinn_net.pth"
torch.save(net.state_dict(),SAVE_PATH)
