# Imports
import torch
import torch.nn as nn 
import matplotlib.pyplot as plt 
from utils import ALL_LETTERS, N_LETTERS
from utils import load_data, line_to_tensor, random_training_example

# Hyperparameters 
category_lines, all_categories = load_data() # eg. category_lines['Arabic'][0] = Khoury
n_categories = len(all_categories) # 18
n_hidden = 128
learning_rate = 0.005

# Create Recurrent Neural Network 
class RNN(nn.Module):
    # implement RNN from scratch rather than using nn.RNN
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.input2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.input2output = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        
        hidden = self.input2hidden(combined)
        output = self.input2output(combined)
        output = self.softmax(output)
        return output, hidden
    
    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)
    
# Initialize Network 
rnn = RNN(N_LETTERS, n_hidden, n_categories)

# Loss & Optimizer 
criterion = nn.NLLLoss() 
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)


# Train Network 
def train(line_tensor, category_tensor):
    hidden = rnn.init_hidden()
    
    # Forward 
    for i in range(line_tensor.size()[0]): # size here is length of name 
        output, hidden = rnn(line_tensor[i], hidden)
    loss = criterion(output, category_tensor)
    
    # Backward & Step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return output, loss.item()

def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

# Check/test results  
current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
    # category_tensor = idx of category 

    output, loss = train(line_tensor, category_tensor)
    current_loss += loss 
    
    # plot & print
    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss / plot_steps)
        current_loss = 0
        
    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = "CORRECT" if guess == category else f"WRONG ({category})"
        print(f"{i+1} {(i+1)/n_iters*100:.4f}% {loss:.4f} {line} / {guess} {correct}")

torch.save(rnn.state_dict(), 'rnn.pth') 
    
plt.figure()
plt.plot(all_losses)
plt.show()

    
