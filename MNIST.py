# Imports
import torch
import torch.nn as nn  # <- Neural Network Modules (nn.Linear, loss functions, etc.)
import torch.optim as optim # <- Optimization Algorithsm (SGD, Adam, etc.) 
from torch.utils.data import DataLoader # <- Easier dataset management: create minibatches to train on 
import torchvision.datasets as datasets # <- Access standard databases
import torchvision.transforms as transforms # <- Transformation on Dataset 

# Device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # use 'gpu' if available -> faster

# Hyperparameters
batch_size = 64
num_classes =10
learning_rate = 0.001
input_size = 784

# Load MNIST Data (28x28)
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# Create Fully Connected Network
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 300)
        self.fc2 = nn.Linear(300, num_classes)
        self.relu = nn.ReLU() 

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out 

# Initialize Network
model = NeuralNet(input_size, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss() # includes SoftMax -> probability array
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network 
num_epochs = 2
n_total_steps = len(train_loader)

for epoch in range(num_epochs):
    for batch_idx, (images, labels) in enumerate(train_loader): 
        # Reshape images: 64x28x28 -> 64x784
        images = images.reshape(-1, 784)

        # Forward
        outputs = model(images)
        #64 x 10

        loss = criterion(outputs, labels)

        # Backward & Update 
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx+1)%100 == 0:
            print(f'epoch: {epoch+1}/{num_epochs}, step: {batch_idx+1}/{n_total_steps}, loss = {loss}')

# Check accuracy
with torch.no_grad(): 
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader: 
        images = images.reshape(-1, 784) 
        outputs = model(images) 
        # output: 64 x 10, labels: 64 x 1
        
        _, prediction = torch.max(outputs, 1) # 64 x 1
        # print(f'pred: {prediction.shape}')

        n_correct += (prediction == labels).sum().item()
        n_samples += labels.shape[0]

    accuracy = n_correct / n_samples * 100.0
    print(f'got {n_correct} / {n_samples} with accuracy = {accuracy:.2f}%')



