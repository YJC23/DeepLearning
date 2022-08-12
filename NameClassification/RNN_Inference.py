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


# Load Model 
rnn = RNN(N_LETTERS, n_hidden, n_categories) # <- creating model object
rnn.load_state_dict(torch.load('rnn.pth'))
rnn.eval() 

# return prediction value 
def category_from_output(output):
    category_idx = torch.argmax(output).item()
    return all_categories[category_idx]

# Accuracy 
n_correct = 0
n_samples = 5000
rnn.eval() # let model know we are on evaluation mode 

with torch.no_grad():
    for i in range(n_samples):
        # Retrieve random name 
        category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
        # print(f'name: {line}')

         # Forward 
        hidden = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)

        guess = category_from_output(output)
        # print(f'{line}: prediction={guess}, answer={category}')

        if guess == category: 
            n_correct += 1
    
    print(f'accuracy: {n_correct/n_samples * 100.0}%')



# Predict
def predict(input_line):
    print(f"\n> {input_line}")
    with torch.no_grad():
        line_tensor = line_to_tensor(input_line)
        
        # Forward 
        hidden = rnn.init_hidden()
        for i in range(line_tensor.size()[0]):
            output, hidden = rnn(line_tensor[i], hidden)
        
        guess = category_from_output(output)
        print(guess)

# while True:
#     sentence = input("Input:")
#     if sentence == "quit":
#         break
#     predict(sentence)
