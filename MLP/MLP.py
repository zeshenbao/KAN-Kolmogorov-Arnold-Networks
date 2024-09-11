# Madeleine Lindström, madeli@kth.se

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from polynomials import polynomial_d1, polynomial_d2, polynomial_d3

class MLP(nn.Module):
    """TODO"""

    def __init__(self, input_size=1, hidden_size=1, output_size=1):  # TODO add selection of activator function?
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, input):
        x = self.fc1(input)
        x = self.relu(x)
        output = self.fc2(x)

        return output

# Constants of the network
input_size = 1
hidden_size = 1
output_size = 1

# Constants of the data
n_datapoints = 5  # Number of datapoints

# Generate test data
x = torch.rand(n_datapoints, input_size)
y = polynomial_d3(x, a=2, b=60, c=1, d=15)
print(x)
print(y)

# Define network
model = MLP(input_size, hidden_size, output_size)  # Create MLP

# Define loss, optimizer
loss_function = nn.MSELoss()  # Mean Squared Error Loss. Alteratively: L1Loss, CrossEntropyLoss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimzier. Alternatively: optim.STD (Stochastic Gradient Descent)

# Training
n_epochs = 1000
epoch_list = list()
loss_list = list()
for epoch in tqdm(range(1, n_epochs)):
    optimizer.zero_grad()

    # Forward pass
    output = model(x)

    # Compute loss
    loss = loss_function(output, y)

    # Backward pass
    loss.backward()

    # Optimize model parameters
    optimizer.step()

    # Save data
    epoch_list.append(epoch)
    loss_list.append(loss.item())


def plot_loss():
    """TODO"""
    plt.plot(epoch_list, loss_list)
    plt.show()


def print_model_parameters():
    """TODO"""
    print(f"Layer name.parameter: parameter value")
    for name, param in model.state_dict().items():
        print(f"{name}: {param}")


print_model_parameters()
plot_loss()


