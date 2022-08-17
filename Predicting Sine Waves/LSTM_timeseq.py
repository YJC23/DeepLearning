import torch
import torch.nn as nn 
import numpy as np 
import torch.optim as optim 
import matplotlib.pyplot as plt

# parameters
N_samples = 100 # number of samples(waves)
L_samples = 1000 # length of each sample(waves)
T = 20 # width of sine wave (changes period like in math)

# Generate Sine Waves
x = np.empty((N_samples, L_samples), dtype=np.float32) # dimension: N_samples x L_samples
x[:] = np.array(range(L_samples)) + np.random.randint(low=-4*T, high=4*T, size=N_samples).reshape(N_samples,1) #
# print(x[:]) <-- increasing by 1 for each row
y = np.sin(x/1.0/T).astype(np.float32)
# print(y[:])

plt.figure(figsize=(10,8))
plt.title("Sine wave")
plt.xlabel("x")
plt.ylabel("y")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.plot(np.arange(x.shape[1]), y[0,:], 'r', linewidth=2.0) #x.shape[1] = length of each wave, y[0,:] = y[0][:] = first y-values 
# plt.show()

class LSTMPredictor(nn.Module): 
    def __init__(self, hidden_size=51):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        # lstm1, lstm2, linear: 2 layers of LSTM written manually 
        self.lstm1 = nn.LSTMCell(1, self.hidden_size) # using one x-cor as input
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size) 
        self.linear = nn.Linear(self.hidden_size, 1) # prediciting one y-cor as output 

    def forward(self, x, future=0):
        # N, 100
        outputs = [] 
        n_samples = x.size(0) # number of waves(?)
         
        # initilizing hidden and cell states 
        h_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32) 
        c_t = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32) 
        h_t2 = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32) 
        c_t2 = torch.zeros(n_samples, self.hidden_size, dtype=torch.float32) 

        for input_t in x.split(1, dim=1): # splitting each x-cordinate(?)
            # N, 1
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # use output(hidden state) of 1st cell as input of 2nd cell 
            output = self.linear(h_t2)
            outputs.append(output)
        
        for i in range(future): # if we should predict the future
            h_t, c_t = self.lstm1(output, (h_t, c_t)) 
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) 
            output = self.linear(h_t2)
            outputs.append(output) 

        outputs = torch.cat(outputs, dim=1) # concatanating the outputs vertically  
        return outputs 


if __name__ == "__main__":
    # y = dim(100, 1000)
    train_input = torch.from_numpy(y[3:, :-1]) # excluding last point, dim(97, 999)
    train_target = torch.from_numpy(y[3:, 1:]) # including last point (reaches 1 more value into the future), dim(97,999)
    test_input = torch.from_numpy(y[ :3 , :-1]) # dim(3, 999)
    test_target = torch.from_numpy(y[ :3, :-1]) # dim(3, 999)


    model = LSTMPredictor()
    criterion = nn.MSELoss() 
    # use LFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(model.parameters(), lr=0.8) # needs closure 

    n_steps = 10
    for i in range(n_steps):
        print("Step", i)

        def closure(): # this goes with LBFGS optimizer
            optimizer.zero_grad()
            out = model(train_input)
            loss = criterion(out, train_target) 
            print("loss", loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)

        with torch.no_grad(): 
            future = 1000
            pred = model(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)

            print("test loss", loss.item())
            y = pred.detach().numpy() # convert to numpy array so that we can plot

        plt.figure(figsize=(12,6))
        plt.title(f"Step {i+1}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        n = train_input.shape[1] # 999

        def draw(y_i, color):
            plt.plot(np.arange(n), y_i[ :n], color, linewidth=2.0) 
            plt.plot(np.arange(n, n+future), y_i[n:], color + ":", linewidth=2.0) 
        
        draw(y[0], 'r')
        draw(y[1], 'b')
        draw(y[2], 'g')

        plt.savefig("predict%d.pdf"%i)
        plt.close()
