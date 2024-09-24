#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 19:03:14 2024
Reference: https://machinelearningmastery.com/activation-functions-in-pytorch/
@author: Oscar Herrera
"""

# importing the libraries, <<paquete>> has fractional optimizers

from paquete import FSGDTorch
import torch
import matplotlib.pyplot as plt
import numpy as np

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import time
start = time.time()




input_size = 784  #Funciona con MNIST
hidden_size = 128

#input_size = 3072 #Funciona con CIFAR10
#hidden_size = 128 #Para ver su funciona con CIFAR10


num_classes = 10

#OSCAR1
eps = 1e-10
caso = 44

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
np.random.seed(0)
inicial=1.0

num_epochs = 100
learning_rate = 0.001 #corridas hechas
#learning_rate = 0.01  #empatando con misteando
#learning_rate = 0.1  #empatando con misteando
#FSGDTorch.Cnnu.nnu=1.7
#optimizer = FSGDTorch.FSGD
optimizer = torch.optim.Adam
#mynu= 0.75# 
#mynu= -2.75
mynu= -2.0
mya= 1.0
myb= 0.0
#myp= 1.0
print("\tv\t", mynu, "\ta\t", mya, "\tb\t", myb)#, "\tp\t", myp)

casostr=["0relu", "1max0", "2max0**1-nu", "3absx**1-nu", "4(x+p2cosp1x)sigm(x)", "5Relu(x) + SUM pi D^nu(x)","6max0+-p1max(+-x+p2)", "7w0", "8w1", "9w2", "10coscotnu*w0", "11coscotnu*w1","12coscotnu*w2", "13cosxw0","14cosxw1","15cosxw2","16relu+p1D^nu(-x+p2)","17relu+SUM1_14pjD^nu(-x+pi)", "18relu+SUM1_7pjD^nuabs(+-x-i)", "19ws", "20triangle", "21triangle2nu", "22ReLU+r", "23rarbrcNU-1Variable", "24htbap2/p1-5+1_initnu2_gamma(2-nu)", "25xr", "26xrx", "27Mish", "28sigmoid", "29softplus", "30polysoftplus", "31swish", "32PolySwish", "33PolySigmoid", "34Haar", "35Triangle", "36Heaviside", "37Sigmoid19.459x", "38PolyMish", "39nu+nu-", "40ABSMish", "41ELU", "42SigmoidPowered","43BimodalSigmoid", "44xmorph", "45xmorphabc", "46PolyGeLU", "47GeLU", "48ELU", "49PolyELU", "50tanhExp", "51Elliot", "52LiSHT", "53E-tanh", "54x3", "55tanh"]   
print("CASE:\t" + casostr[caso] + "\tnu INICIAL =\t", inicial , "\toptimizer=\t" , optimizer.__name__ , "\tFSGDTorch.Cnnu.nnu=\t", FSGDTorch.Cnnu.nnu,"\t lr=", learning_rate) 



class gw(nn.Module):
    def __init__(self):
        super(gw, self).__init__()
    def forward(self, x, nu, p1,p2):#, p3): #nu es param0
      if caso==0:
        r = nn.ReLU()(x)
      elif caso==27:        #Mish         
        r=x*torch.tanh(torch.log(1+torch.exp(x)))        
      elif caso==28: #Sigmoid
        r= 1.0/(1.0+torch.exp(-x))
      elif caso==29: #softplus                         
        r= torch.log(1.0+torch.exp(x))   
      elif caso==31: #Swish        
        r= x * (1.0/(1.0+torch.exp(-x)))         
      elif caso==44:
        nu = torch.maximum (  torch.tensor(-10.0), torch.minimum(nu.clone().requires_grad_(True), torch.tensor(1.0-0.1))  )    
        #nu=torch.tensor(-3.0)
        #nu=torch.tensor(-1.0)
        #p1= torch.maximum(p1, torch.tensor(0)) + eps
        #htba=-x
        #htba =x# 
        #htba=(x-p2)*(1/p1)#*(x-p2) #inicializar p1 =2 p2=0
        htba=x
        #ra=torch.pow( (torch.maximum (torch.abs(htba+1.0)        ,torch.tensor(0)) +eps ).clone().requires_grad_(True), (1-nu).clone().requires_grad_(True) ) / torch.exp(torch.lgamma( (2.0-nu).clone().requires_grad_(True) ))
        #rb=torch.pow( (torch.maximum (torch.abs(htba) ,torch.tensor(0)) +eps ).clone().requires_grad_(True), (1-nu).clone().requires_grad_(True) ) / torch.exp(torch.lgamma( (2.0-nu).clone().requires_grad_(True) ))
        #rc=torch.pow( (torch.maximum (torch.abs(htba-1.0)   ,torch.tensor(0)) +eps ).clone().requires_grad_(True), (1-nu).clone().requires_grad_(True) ) / torch.exp(torch.lgamma( (2.0-nu).clone().requires_grad_(True) ))
        ra=torch.pow( (torch.maximum (         (htba+1.0)        ,torch.tensor(0)) +eps ).clone().requires_grad_(True), (1-nu).clone().requires_grad_(True) ) / torch.exp(torch.lgamma( (2.0-nu).clone().requires_grad_(True) ))
        rb=torch.pow( (torch.maximum (         (htba) ,torch.tensor(0)) +eps ).clone().requires_grad_(True), (1-nu).clone().requires_grad_(True) ) / torch.exp(torch.lgamma( (2.0-nu).clone().requires_grad_(True) ))
        rc=torch.pow( (torch.maximum (         (htba-1.0)   ,torch.tensor(0)) +eps ).clone().requires_grad_(True), (1-nu).clone().requires_grad_(True) ) / torch.exp(torch.lgamma( (2.0-nu).clone().requires_grad_(True) ))
        #p3=torch.where(p3<0,0,1)
        #p3=torch.pow(x, p3)
        r = (ra -2*rb + rc)   
      elif caso==52: #LiSHT    pinta buena  
        r = x*torch.tanh(x)        
      elif caso==55: #tanh
        r= torch.tanh(x)
      return r
#OSCAR1



# Load the MNIST dataset
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='data/', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='data/', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, activation_function):
        super(NeuralNetwork, self).__init__()

        self.additional_param0 = nn.Parameter(torch.tensor(mynu))
        self.register_parameter('additional_param0', self.additional_param0  )
        self.additional_param1 = nn.Parameter(torch.tensor(mya))
        self.register_parameter('additional_param1', self.additional_param1  )
        self.additional_param2 = nn.Parameter(torch.tensor(myb) )
        self.register_parameter('additional_param2', self.additional_param2  )        

        #self.additional_param3 = nn.Parameter(torch.tensor(myp))
        #self.register_parameter('additional_param3', self.additional_param3  )
        #self.additional_param4 = nn.Parameter(torch.tensor(inicial*1.0))
        #self.register_parameter('additional_param4', self.additional_param4  )
        #self.additional_param5 = nn.Parameter(torch.tensor(inicial*0.0))
        #self.register_parameter('additional_param5', self.additional_param5  )
        
        #self.additional_param6 = nn.Parameter(torch.tensor(inicial*1.0))
        #self.register_parameter('additional_param6', self.additional_param6  )        
        #self.additional_param7 = nn.Parameter(torch.tensor(inicial*(1.0)))
        #self.register_parameter('additional_param7', self.additional_param7  )
        #self.additional_param8 = nn.Parameter(torch.tensor(inicial*(0.0)))
        #self.register_parameter('additional_param8', self.additional_param8  )
        
        self.layer1 = nn.Linear(input_size, hidden_size)
        #self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_classes)
        self.activation_function = activation_function

    def forward(self, x):
        x=self.layer1(x)
        x = self.activation_function(x, self.additional_param0, self.additional_param1, self.additional_param2)#, self.additional_param3)                
        #x = self.layer2(x)
        #x = self.activation_function(x, self.additional_param3, self.additional_param4, self.additional_param5)  
        #x = self.activation_function(x, self.additional_param6, self.additional_param7, self.additional_param8)          
        x = self.layer3(x)
        return x
    
def train(network, data_loader, criterion, optimizer, device):#, scheduler):
   
    network.train()
    running_loss = 0.0    
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(data.shape[0], -1)

        optimizer.zero_grad()
        output = network(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()        
        #scheduler.step()
        #print("Learning Rate = ",optimizer.param_groups[0]["lr"])        
        running_loss += loss.item() * data.size(0)

    return running_loss / len(data_loader.dataset)

def test(network, data_loader, criterion, device):
    network.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)

            output = network(data)
            loss = criterion(output, target)
            test_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return test_loss / len(data_loader.dataset), 100 * correct / total

    


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

activation_functions = {
    #'ReLU': nn.ReLU(),
    
    #'ReLU': gw(),
    #'Sigmoid': gw(),
    #'Swish': gw(),
    #'Mish': gw(),
    #'Softplus': gw(),    
    'Morph': gw(),
    #'tanh': gw(),    
    #'Lisht':gw()
    
    #'Sigmoid': nn.Sigmoid(),
    #'Tanh': nn.Tanh(),
    #'LeakyReLU': nn.LeakyReLU()
}

results = {}

# Train and test the model with different activation functions, all of them belongs to Morph cases
for name, activation_function in activation_functions.items():
    print(f"Training with {name} activation function...")
    if name=="ReLU":
        caso=0 
    if name=="Sigmoid":
        caso=28
    if name=="Swish":
        caso=31        
    if name=="Mish":
        caso=27        
    if name=="Softplus":
        caso=29        
    if name=="Morph":
        caso=44         
    if name=="Lisht":
       caso=52
#    if name=="tanh":
#        caso =55    
       
    print("name= " , name ,  " caso = ", caso)
    model = NeuralNetwork(input_size, hidden_size, num_classes, activation_function).to(device)
    criterion = nn.CrossEntropyLoss()

    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizerr = optimizer(model.parameters(), lr=learning_rate)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerr, T_max=10, eta_min=0) 
    train_loss_history = []
    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(num_epochs):
        #print("Learning Rate = ",optimizerr.param_groups[0]["lr"], "\t",end="")        
        train_loss = train(model, train_loader, criterion, optimizerr, device)#,scheduler)        
        test_loss, test_accuracy = test(model, test_loader, criterion, device)

        train_loss_history.append(train_loss)
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss:\t{train_loss:.4f}\t, Test Loss:\t{test_loss:.4f}\t, Test Accuracy:\t{test_accuracy:.2f}\t%", end="")
        print("\tEpoch",(epoch+1),"\t model.additional_param0 =  \t" , model.additional_param0.data,"\t model.additional_param1 =  \t" , model.additional_param1.data, "\t model.additional_param2 =  \t" , model.additional_param2.data)# , "\t model.additional_param3 =  \t" , model.additional_param3.data)            
    results[name] = {
        'train_loss_history': train_loss_history,
        'test_loss_history': test_loss_history,
        'test_accuracy_history': test_accuracy_history
    }
    
    

# Plot the training loss
plt.figure()
for name, data in results.items():
    plt.plot(data['train_loss_history'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend()
plt.show()

# Plot the testing loss
plt.figure()
for name, data in results.items():
    plt.plot(data['test_loss_history'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Testing Loss')
plt.legend()
plt.show()

# Plot the testing accuracy
plt.figure()
for name, data in results.items():
    plt.plot(data['test_accuracy_history'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Testing Accuracy')
plt.legend()
plt.show()

end = time.time()
print("ELAPSED TIME= ", round(end - start,2), "\n")        