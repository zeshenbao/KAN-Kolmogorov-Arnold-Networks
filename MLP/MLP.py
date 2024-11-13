# Madeleine Lindström, madeli@kth.se

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt

from polynomials import polynomial_d1, polynomial_d2, polynomial_d3

class MLP(nn.Module):
    """Creates a Multilayer Perceptron."""

    def __init__(self, input_size=1, hidden_sizes=[1], output_size=1):
        super(MLP, self).__init__()
        self.layers = list()

        i_size = input_size

        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(i_size, h_size))
            self.layers.append(nn.ReLU())  # ReLU on all layers
            i_size = h_size
        
        self.layers.append(nn.Linear(i_size, output_size))
        self.layers = nn.Sequential(*self.layers)
        self.optimizer = optim.Adam(self.parameters(), lr=0.01)  # Adam optimzier. Alternatively: optim.STD (Stochastic Gradient Descent)
        self.loss_function = nn.MSELoss()  # Mean Squared Error Loss. Alteratively: L1Loss, CrossEntropyLoss
        self.epoch_list = list()
        self.loss_list = list()
        self.val_loss_list = list()
        self.validation_x = list()
        self.validation_y = list()

    
    def forward(self, x):
        return self.layers(x)

    def predict(model, new_data):
        model.eval()  # Set the model to evaluation mode
    
        with torch.no_grad():  # Disable gradient calculation for inference
            outputs = model(new_data)  # Directly return the output for regression
        
        return outputs
    
    def fit(self, x, y, n_epochs):
        # Training

        for epoch in tqdm(range(1, n_epochs)):
            # Set model to training mode
            self.train()

            self.optimizer.zero_grad()

            # Forward pass
            output = self.forward(x)

            # Compute loss
            loss = self.loss_function(output, y)

            # Backward pass
            loss.backward()

            # Optimize model parameters
            self.optimizer.step()

            # Save data
            self.epoch_list.append(epoch)
            self.loss_list.append(loss.item())
        
            self.eval()  # Set the model to evaluation mode

            with torch.no_grad():  # Disable gradient calculation for inference
                validation_output = self.forward(self.validation_x)
                validation_loss = self.loss_function(validation_output, self.validation_y)
                self.val_loss_list.append(validation_loss.item())

    def set_val_data(self, x, y):
        self.validation_x = x
        self.validation_y = y

def constants():
    # Constants of the network
    input_size = 1
    hidden_sizes = [8,8]
    output_size = 1

    # Constants of the data
    n_datapoints = 100  # Number of datapoints

    n_epochs = 5000

    # Generate test data
    # x = torch.rand(n_datapoints, input_size)  # Random floats [0, 1)
    x = torch.randint(-100, 100, (n_datapoints, input_size)).float()  # Random numbers [low, high]
    #x = torch.sort(x, dim=0)[0]
    y = polynomial_d2(x, a=1, b=0, c=5)  # To generate y = f(x)

    model = MLP(input_size, hidden_sizes, output_size)
    model.fit(x,y,n_epochs)

    #print(model.forward(x[50]).detach())
    #print(y[50])

    y_approx = model.predict(x)
    plt.title("y as a function of x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, 'r*', label="datapoints")  # Plots datapoints
    plt.plot(x, y_approx, 'b*', label="approximation")  # Plots approximated datapoints
    plt.legend()
    plt.show()


def plot_loss():
    """Plots the loss during training."""
    plt.title("Loss as a Function of Number of Epochs")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(epoch_list, loss_list)
    plt.show()

def plot_result():
    # Sort data
    print(y)
    y_approx = model(x).detach()
    print(y_approx)

    # Plot results
    plt.title("y as a function of x")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y, 'r*', label="datapoints")  # Plots datapoints
    plt.plot(x, y_approx, 'b*', label="approximation")  # Plots approximated datapoints
    plt.legend()
    plt.show()

def print_model_parameters():
    """TODO"""
    print(f"Layer name.parameter: parameter value")
    for name, param in model.state_dict().items():
        print(f"{name}: {param}")



