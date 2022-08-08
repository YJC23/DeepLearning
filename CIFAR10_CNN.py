from select import KQ_NOTE_RENAME
from turtle import down
import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.utils as utils
import matplotlib.pyplot as plt
import numpy as np

# Device 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_epochs = 10
batch_size = 4
num_classes = 10
learning_rate = 0.001

# Dataset has PILImage images of range [0, 1] -> transform to Tensors of normalized range [-1, 1]
transform = transforms.Compose(
    [transforms.ToTensor(), # becomes tensor of the form (Channel, Height, Width) & (0,255) -> (0,1)
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) # transforms (0,1) -> (-1,1)

# Load CIFAR10 Data 
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # has data reshuffled at every epoch -> more robust/accurate
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True) 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# # Functions to show an image
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()

# # get some random training images
# dataiter = iter(train_loader)
# images, labels = dataiter.next()

# # show images
# imshow(utils.make_grid(images))

# print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

# Create Convolutional Neural Network 
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1) 
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1) 
        self.fc1 = nn.Linear(12*5*5, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, 10)
        self.relu = nn.ReLU()

    def forward(self, x): 
        # Image: 3 x 32 x 32 
        x = self.relu(self.pool(self.conv1(x))) # 3 x 28 x 28 -> 3 x 14 x 14
        x = self.relu(self.pool(self.conv2(x))) # 3 x 10 x 10 -> 3 x 5 x 5
        x = x.view(-1, 12*5*5)
        x = self.relu(self.fc1(x)) # 400 -> 120
        x = self.relu(self.fc2(x)) # 120 -> 80
        x = self.fc3(x) # 80 -> 10 
        return x

# Initialize Network 
model = ConvNet()

# Loss & Optimizer 
criterion = nn.CrossEntropyLoss() # includes softmax
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network 
n_total_steps = len(train_loader)

for epoch in range(num_epochs): 
    for batch_idx, (images, labels) in enumerate(train_loader): 
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward & Update
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad(())

        if (batch_idx+1) % 2000 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}, Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item()}')

# Check Accuracy 
with torch.no_grad(): 
    n_correct = 0
    n_samples = 0

    for images, labels in test_loader: 
        outputs = model(images) 
        
        _, prediction = torch.max(outputs, 1) 

        n_correct += (prediction == labels).sum().item()
        n_samples += labels.shape[0]

    accuracy = n_correct / n_samples * 100.0
    print(f'got {n_correct} / {n_samples} with accuracy = {accuracy:.2f}%')


