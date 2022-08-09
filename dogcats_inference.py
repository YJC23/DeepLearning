import torch
import glob 
import torch.nn as nn  
import torch.optim as optim 
from torch.utils.data import DataLoader 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms 
from torchvision.models import squeezenet1_1
from PIL import Image 

# device config (not necessary) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
# print(device)

# hyperparameters
num_epochs = 2
num_classes = 2 
batch_size = 64
learning_rate = 0.001

# Path for training and testing directory 
pred_path = '/Users/youngjuchoi/dogs_cats_data/test1'
classes = ['cat', 'dog']

# Transforms
transformer = transforms.Compose([
    transforms.Resize((64, 64)), # resizing image to 150, 150
    transforms.ToTensor(), # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5,0.5,0.5], # 0-1 to [-1,1], formula (x-mean)/std
                         [0.5,0.5,0.5]),    
])

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

# Load Model 
FILE = "best_model.pth"
model = ConvNet()
model.load_state_dict(torch.load(FILE))
model.eval() 

# # Check accuracy 
# with torch.no_grad(): 
#     n_correct = 0
#     n_samples = 0

#     for images, labels in test_loader: 
#         outputs = model(images) 
        
#         _, prediction = torch.max(outputs, 1) 

#         n_correct += (prediction == labels).sum().item()
#         n_samples += labels.shape[0]

#     accuracy = n_correct / n_samples * 100.0
#     print(f'got {n_correct} / {n_samples} with accuracy = {accuracy:.2f}%')

# Predicting Individual Images
def prediction(img_path): 
    image = Image.open(img_path)
    image_tensor = transformer(image)

    image_tensor = image_tensor.unsqueeze(0) # unsqueeze adds the batch layer -> compatible with model 
    # image_tensor dim: 1 x 3 x 64 x 64

    output = model(image_tensor)
    index = output.data.numpy().argmax() #gives index of biggest value

    pred = classes[index] 
    return pred

images_path = glob.glob(pred_path+'/*.jpg') # finds all the pathnames mathcing a specified pattern 
pred_dict = {}
for images in images_path:
    pred_dict[images[images.rfind('/')+1:]] = prediction(images) # getting the name of images using '/' 

# sorting 
import re
for k, v in sorted(pred_dict.items(), key=lambda x: int(re.compile("\d+").search(x[0]).group(0))):
    print(f'{k}: {v}') #Q: How to sort? 