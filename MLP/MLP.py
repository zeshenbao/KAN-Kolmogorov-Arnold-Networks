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
            loss = torch.sqrt(self.loss_function(output, y)) # sqrt to get RMSE instead of MSE

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
                validation_loss = torch.sqrt(self.loss_function(validation_output, self.validation_y))  # sqrt to get RMSE instead of MSE
                self.val_loss_list.append(validation_loss.item())

    def set_val_data(self, x, y):
        self.validation_x = x
        self.validation_y = y



