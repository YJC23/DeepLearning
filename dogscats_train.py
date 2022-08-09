# Using CNN 
import torch
import torch.nn as nn  # <- Neural Network Modules (nn.Linear, loss functions, etc.)
import torch.optim as optim # <- Optimization Algorithsm (SGD, Adam, etc.) 
from torch.utils.data import DataLoader # <- Easier dataset management: create minibatches to train on 
import torchvision.datasets as datasets # <- Access standard databases
import torchvision.transforms as transforms # <- Transformation on Dataset 

# device config (not necessary) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# print(device)

# hyperparameters
num_epochs = 3
num_classes = 2 
batch_size = 64
learning_rate = 0.001

# Path for training and testing directory 
train_path = '/Users/youngjuchoi/dogs_cats_data/train'
test_path = '/Users/youngjuchoi/dogs_cats_data/test1'
classes = ['cat', 'dog']

# Transforms
transformer = transforms.Compose([
    transforms.Resize((64, 64)), # resizing image to 150, 150
    transforms.ToTensor(), # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1], formula (x-mean)/std
                         [0.5,0.5,0.5]),    
])

# Load Datasets
dataset = datasets.ImageFolder(train_path, transform=transformer)
train_set, test_set = torch.utils.data.random_split(dataset, [20000, 5000])
train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# Create Neural Network 
class ConvNet(nn.Module): 
    def __init__(self): 
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1, padding=2) 
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(1024, 512) # 20x8x8
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)
        self.relu = nn.ReLU() 
    
    def forward(self, x): 
        # Image: 3 x 64 x 64
        x = self.relu(self.pool(self.conv1(x))) # 6 x 64 x 64 -> 6 x 32 x 32
        x = self.relu(self.pool(self.conv2(x))) # 10 x 32 x 32 -> 10 x 16 x 16
        x = self.relu(self.pool(self.conv3(x))) # 16 x 16 x 16 -> 16 x 8 x 8 
        x = x.view(-1, 16*8*8)
        x = self.relu(self.fc1(x)) # 1024 -> 512
        x = self.relu(self.fc2(x)) # 512 -> 128
        x = self.fc3(x) # 128 -> 2
        return x 

#Initialize Network
model = ConvNet()

# Optimizer & Loss 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training (with testing) 
n_total_steps = len(train_loader)
best_accuracy = 0.0

for epoch in range(num_epochs): 
    model.train()
    for batch_idx, (images, labels) in enumerate(train_loader): 
        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward & Update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (batch_idx+1) % 50 == 0: 
            print(f'Epoch [{epoch+1}/{num_epochs}, Step [{batch_idx+1}/{n_total_steps}], Loss: {loss.item()}')

    # Check Accuracy 
    model.eval()
    with torch.no_grad(): 
        n_correct = 0
        n_samples = 0

        for images, labels in test_loader: 
           outputs = model(images) 
                
           _, prediction = torch.max(outputs, 1) 
           n_correct += (prediction == labels).sum().item()
           n_samples += labels.shape[0]
           
           accuracy = n_correct / n_samples * 100.0
        
        # Save best model 
        if accuracy > best_accuracy: 
            print(f'got {n_correct} / {n_samples} with new best accuracy = {accuracy:.2f}%')
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'best_model.pth')
            

