import json 
import numpy as np 
from nltk_utils import tokenize, stem, bag_of_words

import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
for intent in intents['intents']: 
    tag = intent['tag']
    tags.append(tag) 
    for pattern in intent['patterns']: 
        w = tokenize(pattern) 
        all_words.extend(w) 
        xy.append((w, tag)) # saves ["How", "are", "you"], "greeting"

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words)) # sort & remove duplicates 
tags = sorted(set(tags)) # not really necessary but JIC

# print(all_words)

X_train = []
y_train = []
for (pattern_sentence, tag) in xy: 
    bag = bag_of_words(pattern_sentence, all_words)  
    X_train.append(bag) # X_train has the one-hot encoded arrays

    label = tags.index(tag) # index of label 
    y_train.append(label) # CrossEntropyLoss 

# print(X_train[0])

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset): 
    def __init__(self): 
        self.n_samples = len(X_train) 
        self.x_data = X_train 
        self.y_data = y_train 

    def __getitem__(self, idx): 
        return self.x_data[idx], self.y_data[idx] 
    
    def __len__(self):
        return self.n_samples


# Hyperparameters
batch_size = 8 
input_size = len(X_train[0])
hidden_size = 8 
output_size = len(tags)
learning_rate = 0.001
num_epochs = 1000

# Load Data
dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Initialize Model 
model = NeuralNet(input_size, hidden_size, output_size) 

# Loss & Optimizer
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

for epoch in range(num_epochs): 
    for (words, labels) in train_loader: 
        
        # Forward pass
        outputs = model(words) 
        loss = criterion(outputs, labels)
        
        # Backward & optimizer step 
        loss.backward() 
        optimizer.step()
        optimizer.zero_grad()

    if (epoch+1) % 100 == 0: 
        print(f'epoch {epoch+1}/{num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size, 
    "hidden_size": hidden_size, 
    "output_size": output_size, 
    "all_words": all_words, 
    "tags": tags
}
FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}') 