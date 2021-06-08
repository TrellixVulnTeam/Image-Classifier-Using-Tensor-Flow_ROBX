#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 17:53:50 2021

@author: tathagat
"""

import torch, torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim

trainset = torchvision.datasets.MNIST(root = './data', train = True, transform = transforms.Compose([transforms.ToTensor()]), download = True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle = True)

for images, labels in trainloader:
    print(images.size(), labels.size())
    break
#64 is number of samples in each batch size (batch_size) 
#1 //grayscale
#28,28 is the size of images

batches = iter(trainloader)

one_batch = next(batches)

images, labels = one_batch
print(len(images))

plt.imshow(images[4].numpy().squeeze(), cmap = "Greys_r")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(28*28, 512)
        self.hidden2 = nn.Linear(512, 512)
        self.output = nn.Linear(512, 10)
        
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x):
        x = self.hidden(x)
        #x = self.sigmoid(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
model = Net()
print(model)

criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01)

for epoch in range(5):
    running_loss = 0
    for images, labels in trainloader:
        images = images.view(images.shape[0], - 1)
        optimizer.zero_grad()  #Starting from zero grads
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward() #generate gradients
        optimizer.step()
        running_loss += loss.item()
    else:
        print('The running loss is: {}'.format(running_loss/len(trainloader)))
        
#images, labels = next(iter(trainloader))    #Only for 1 epoch

images, labels = next(iter(trainloader))
img = images[0].view(1, -1)

with torch.no_grad():
    logprobs = model(img)

print(logprobs)     #Not the actual probs, just the output of them model

probs = torch.exp(logprobs)
print(probs)  #Actual probabilites 

print(torch.sum(probs))  #Probs should sum up to 1

print(torch.argmax(probs)) #Printing the number


        
        
        
        
        
        
        
        
        
