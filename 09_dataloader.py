'''
Terminology:

1. epoch = one forward and backward pass of ALL training samples
2. batch_size = number of training samples used in one forward/backward pass
3. number of iterations = number of passes, each pass (forward+backward) using [batch_size] number of sampes

- e.g : 100 samples, batch_size=20 -> 100/20=5 iterations for 1 epoch
- Gradient computation etc. not efficient for whole data set --> divide dataset into small batches

Training loop:
for epoch in range(num_epochs):
    for i in range(total_batches):  # loop over all batches
        batch_x, batch_y = ...

--> DataLoader can do the batch computation for us
'''

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

"""
Implement a custom Dataset:
1) inherit Dataset
2) implement __init__ , __getitem__ , and __len__
"""

class WineDataset(Dataset):

    def __init__(self):
        # data loading
        xy = np.loadtxt('./data/wine/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)

        # 1st column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

        self.n_samples = xy.shape[0]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]   # tuple

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset = WineDataset()

# get a sample and check
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# Load whole dataset with DataLoader
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,     # shuffle data, good for training
                          num_workers=2)    # faster loading with multiple subprocesses

# convert to an iterator and look at one random sample
data_iter = iter(train_loader)
data = data_iter.next()
features, labels = data
print(features, labels)

# Dummy training loop
num_epochs = 2
total_samples = len(dataset)
n_iterations = math.ceil(total_samples/4)
print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # training steps here...
        
        # 178 samples, batch_size = 4, n_iters=178/4=44.5 -> 45 iterations
        if (i+1) % 5 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Step {i+1}/{n_iterations}| Inputs {inputs.shape} | Labels {labels.shape}')


# some famous datasets are available in torchvision.datasets
# e.g. MNIST, Fashion-MNIST, CIFAR10, COCO

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                           train=True, 
                                           transform=torchvision.transforms.ToTensor(),  
                                           download=True)

train_loader = DataLoader(dataset=train_dataset, 
                          batch_size=3, 
                          shuffle=True)

# look at a random sample
data_iter = iter(train_loader)
data = data_iter.next()
inputs, targets = data
print(inputs.shape, targets.shape)


