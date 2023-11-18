import numpy as np
import torch
import torch.nn as nn
device = "cuda" if torch.cuda.is_available() else "cpu"


def pinn_loss(x, t, net):
    u = net(x, t)
    loss = nn.MSELoss()

    u_t = torch.autograd.grad(
        u, t,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]

    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]


    pinn_loss = u_t + u*u_x - (0.01/np.pi)*u_xx
    zeros_t = torch.zeros(pinn_loss.size()).to(device)
    pinn_loss_ = loss(pinn_loss, zeros_t)
    return pinn_loss_

def initial_condition_loss(x, t, net, u_ini):
    u = net(x, t)
    loss = nn.MSELoss()
    initial_condition_loss = loss(u, u_ini)
    return initial_condition_loss

def boundary_condition_loss(x, t, net, u_bc):
    u = net(x, t)
    loss = nn.MSELoss()
    boundary_condition_loss = loss(u, u_bc)
    return boundary_condition_loss

class PINN(nn.Module):
    def __init__(self, activation):
        super().__init__()
        self.activation = activation

        self.linear1 = self.linear(2, 5)
        self.linear2 = self.linear(5, 20)
        self.linear3 = self.linear(20, 40)
        self.linear4 = self.linear(40, 40)
        self.linear5 = self.linear(40, 40)
        self.linear6 = self.linear(40, 20)
        self.linear7 = self.linear(20, 10)
        self.linear8 = self.linear(10, 5)
        self.regressor = nn.Linear(5, 1)

    def linear(self, in_features, out_features):
        layers = [nn.Linear(in_features, out_features)]
        if self.activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif self.activation == 'tanh':
            layers.append(nn.Tanh())
        else:
            layers.append(nn.Sigmoid())
        net = nn.Sequential(*layers)
        return net

    def forward(self, x, t):
        inputs = torch.cat([x, t], axis=1)
        h = self.linear1(inputs)
        h = self.linear2(h)
        h = self.linear3(h)
        h = self.linear4(h)
        h = self.linear5(h)
        h = self.linear6(h)
        h = self.linear7(h)
        h = self.linear8(h)
        h = self.regressor(h)
        return h
