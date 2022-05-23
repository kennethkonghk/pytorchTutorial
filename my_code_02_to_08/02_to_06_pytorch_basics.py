# -----------------------------------------------------------
# Note of learning pytorch basics
#
# author: Chun Ho Kong
# email: kennethkong.uni@gmail.com
#
# references:
# https://www.youtube.com/watch?v=c36lUUr864M
# -----------------------------------------------------------

############################################################
# Tensor Basics
############################################################
print("##### Tensor Basics #####")

import torch
import numpy as np
import sys

# create tensors
x = torch.empty(3)  # 1D
x = torch.empty(2,3)  # 2D
x = torch.empty(2,3,4)  # 3D

x = torch.rand(2,2)     # uniform distribution
x = torch.randn(2,2)     # normal distribution
x = torch.zeros(2,2)
x = torch.ones(2,2)

# check type
print(x.dtype)  # default: float32
x = torch.ones(2,2, dtype=torch.int)
x = torch.ones(2,2, dtype=torch.double)
x = torch.ones(2,2, dtype=torch.float16)

# check size
print(x.size())

x = torch.tensor([2.5, 0.1]) # list --> tensor

# operations (+,-,*,/)
x = torch.rand(2,2)
y = torch.rand(2,2)
y.add_(x)   # inplace

print(x[1,1].item())    # actual value, only for 1x1

# reshape
x = torch.rand(4,4)
y = x.view(16)
y = x.view(-1,8)

# numpy --> tensor
a = torch.ones(5)   # on cpu
b = a.numpy()   # share same memory with a
print(type(b))

a.add_(1)
print(f"a: {a}")
print(f"b: {b}")

# tensor --> numpy  
a = np.ones(5)
b = torch.from_numpy(a)
a += 1  # b will be modified too

# check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
# additional info if using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

if torch.cuda.is_available():
    device = torch.device("cuda")
#     x = torch.ones(5, device=device)  # slow to load
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x + y
#     # z.numpy()   # error, numpy can only handle CPU tensor
#     z = z.to("cpu")
#     z.numpy()   # now works

############################################################
# Autograd
############################################################
print("##### Autograd #####")

# tell pytorch a variable will need gradient for optimisation later
x = torch.randn(3, requires_grad=True)   # default: False
print(x)

y = x + 2
print(y)    # there's grad_fn attribute!
z = y * y * 2
z = z.mean()    # scalar
print(z)

z.backward()    # dz/dx, basically a J*v product, need requires_grad=True
print(x.grad)

z = y * y * 2   # vector
print(z)
v = torch.tensor([0.1, 1.0, 0.001], dtype=torch.float32)
z.backward(v)

# prevent from tracking the gradients
print(x)    # original
# 1st way
x.requires_grad_(False)
print(x)
# 2nd way
y = x.detach()
print(y)
# 3rd way
with torch.no_grad():
    y = x + 1
print(y)

# note that grad accumulates
weights = torch.ones(4, requires_grad=True)
for epoch in range(3):
    model_output = (weights*3).sum()
    model_output.backward()
    print(weights.grad)     # wrong if it's without next line
    weights.grad.zero_()    # empty the grad
# in practice
optimizer = torch.optim.SGD([weights], lr=0.01)
optimizer.step()
optimizer.zero_grad()

############################################################
# Backpropagation
############################################################
print("##### Backpropagation #####")

x = torch.tensor(1.0)
y = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)
# b = torch.tensor(2.0, requires_grad=True)

# forward pass and compute the loss
y_hat = w * x
# y_hat = w * x + b
loss = (y_hat - y)**2

print(loss)

# backward pass
loss.backward()
print(w.grad)
# print(b.grad)

# update weights
# next forward and backward pass

############################################################
# Gradient Descent / Training Pipeline
############################################################
print("##### Gradient Descent / Gradient Pipeline #####")

### manually

# f = w * x
# f = 2 * x
X = np.array([1,2,3,4], dtype=np.float32)
Y = np.array([2,4,6,8], dtype=np.float32)
w = 0.0

# model prediction
def forward(x):
    return w * x

# loss = MSE
def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N * 2*x * (w*x - y)
def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted-y).mean()

print(f"Prediction before training: f(5) = {forward(5):.3f}")

# Training
learning_rate =0.01
n_iters = 20

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    w -= learning_rate * dw

    if epoch % 2 == 0:
        print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")

print(f"Prediction after training: f(5) = {forward(5):.3f}")

############################################################
### using pytorch

# Training pipeline
# 1) Design model (input size, output size, forward pass)
# 2) Construct loss and optimizer
# 3) Training loop
#   - forward pass: compute prediction
#   - backward pass: gradients
#   - update weights

import torch.nn as nn

# f = w * x
# f = 2 * x
# X = torch.tensor([1,2,3,4], dtype=torch.float32)    # Size([4])
# Y = torch.tensor([2,4,6,8], dtype=torch.float32)
X = torch.tensor([[1],[2],[3],[4]], dtype=torch.float32)    # Size([4, 1])
Y = torch.tensor([[2],[4],[6],[8]], dtype=torch.float32)
# X = torch.tensor([[1,2,3,4]], dtype=torch.float32)    # Size([1, 4]), will not work
# Y = torch.tensor([[2,4,6,8]], dtype=torch.float32)
# w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
# n_features, n_samples = X.shape     # not working
print(f"n_samples: {n_samples}, n_features: {n_features}")

input_size = n_features
output_size = n_features

# model = nn.Linear(input_size, output_size)    # initialize weights randomly

class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        """
        Explanation of super(LinearRegression, self).__init__():

        [1st part - super(LinearRegression, self)]
        1st param: the subclass
        2nd param: an (instantiated) object / an instance of that subclass
        Why we need the 2nd param?
         - To ensure super() return a bound method (a method that is bound to the object)
        --> Literally: parent of that subclass (1st param)

        [2nd part - .__init__() method]
        --> Search for a matching method from the parent of that subclass
            Note: Technically, super() doesn’t return a method but a proxy object (an object that delegates calls to the correct class methods without making an additional object)

        OR super().__init__() (only in python 3, recommended)
        """
        # super(LinearRegression, self).__init__()
        super().__init__()
        # define layers
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

model = LinearRegression(input_size, output_size)

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")

# Training
learning_rate =0.01
n_iters = 100

loss = nn.MSELoss()
# optimizer = torch.optim.SGD([w], lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(Y, y_pred)

    # gradients = backward pass
    l.backward()    # dl/dw

    # update weights
    # with torch.no_grad():
    #     w -= learning_rate * w.grad
    optimizer.step()

    # zero gradients
    # w.grad.zero_()
    optimizer.zero_grad()

    if epoch % 10 == 0:
        # print(f"epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}")
        [w, b] = model.parameters()
        print(f"epoch {epoch+1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f"Prediction before training: f(5) = {model(X_test).item():.3f}")




