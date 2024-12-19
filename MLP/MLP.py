# Madeleine Lindström, madeli@kth.se

import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os


class MLP(nn.Module):
    """Creates a Multilayer Perceptron."""

    def __init__(self, result_path=None, input_size=1, hidden_sizes=[1], output_size=1, lr=0.001):
        super(MLP, self).__init__()
        self.layers = list()
        self.RESULTSPATH = result_path
        i_size = input_size

        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.lr = lr

        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(i_size, h_size))
            self.layers.append(nn.ReLU())  # ReLU on all layers
            i_size = h_size
        
        self.layers.append(nn.Linear(i_size, output_size))
        self.layers = nn.Sequential(*self.layers)
        self.optimizer = optim.LBFGS(self.parameters(), lr=self.lr)  # Adam optimzier. Alternatively: optim.STD (Stochastic Gradient Descent)
        self.loss_function = nn.MSELoss()  # Mean Squared Error Loss. Alteratively: L1Loss, CrossEntropyLoss
        self.epoch_list = list()
        self.loss_list = list()
        self.val_loss_list = list()
        
    def load_data(self, data, deepmimo=False):
        if deepmimo:
            self.X_train =  data['train'][0]           
            self.y_train = data['train'][1]
            self.X_validation = data['validation'][0]
            self.y_validation = data['validation'][1]
            self.X_test =  data['test'][0]
            self.y_test = data['test'][1]
            self.dataset = {"train_input": self.X_train, "train_label":self.y_train, "test_input":self.X_validation, "test_label":self.y_validation}

        else:
            self.X_train =  data['train'][0][:,1].unsqueeze(1)              # get the y_noise
            self.y_train = data['train'][1].unsqueeze(1)
            self.X_validation =  data['validation'][0][:,1].unsqueeze(1)    # get the y_noise
            self.y_validation = data['validation'][1].unsqueeze(1)
            self.X_test =  data['test'][0][:,1].unsqueeze(1)                # get the y_noise
            self.y_test = data['test'][1].unsqueeze(1)
            self.dataset = {"train_input": self.X_train, "train_label":self.y_train, "test_input":self.X_test, "test_label":self.y_test}

    
    def forward(self, x):
        return self.layers(x)

    def predict(self, x, eval=None):
        """Make predictions using the trained model."""
        X_tensor = torch.Tensor(x)
        if eval:
            res = {"preds":0, "test_loss":0}
            loss = torch.nn.MSELoss()
            X_tensor = torch.Tensor(x)
            self.eval()  
            with torch.no_grad():  
                y_pred = self(X_tensor)
                res["preds"] = y_pred
                res["test_loss"] = torch.sqrt(loss(y_pred, self.y_test))
            return res
        else:
            self.eval()
            with torch.no_grad():
                return self(X_tensor)

    
    def fit(self, X, y, n_epochs, cross_validation=False, deepmimo=False):
        # Training
        start = time.time()
        
        for epoch in tqdm(range(1, n_epochs+1)):
            # Set model to training mode
            self.train()

            def closure():

                self.optimizer.zero_grad()

            # Forward pass
                output = self.forward(X)

            # Compute loss
                loss = torch.sqrt(self.loss_function(output, y)) # sqrt to get RMSE instead of MSE

            # Backward pass
                loss.backward()

                return loss

            # Optimize model parameters
            loss = self.optimizer.step(closure)

            # Save data
            self.epoch_list.append(epoch)
            self.loss_list.append(loss.item())
            
            if not cross_validation:
                self.eval()  # Set the model to evaluation mode

                with torch.no_grad():  # Disable gradient calculation for inference
                    y_pred = self(self.X_validation)
                    validation_loss = torch.sqrt(self.loss_function(y_pred, self.y_validation))  # sqrt to get RMSE instead of MSE
                    self.val_loss_list.append(validation_loss.item())
            
            end = time.time()
            elapsed_time = end - start

        return {'train_loss': self.loss_list, 'test_loss': self.val_loss_list}, elapsed_time

    def plot_prediction(self, data, y_preds, type_='test', save=False):
        """
        Plot predictions made by the model.
        """
        # Define colors
        viridis = plt.cm.viridis
        data_point_color = viridis(0.5)
        true_function_color = viridis(0.8)
        predicted_function_color = viridis(0.2)

        plt.figure(figsize=(8,6))

        # Set font size, grid, etc.
        plt.rcParams.update({
            'font.size': 15,
            'axes.labelsize': 15,
            'axes.titlesize': 15,
            'legend.fontsize': 15,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
        })

        #sns.set_theme(style="whitegrid")

        # samples
        x_values_samples = data[type_][0][:,0]
        y_noise = data[type_][0][:,1]

        # plot noisy datapoints
        plt.scatter(x_values_samples, y_noise, label=f"{type_} data".capitalize(), color=data_point_color, alpha=0.8, s=70, zorder=3, marker='.', linestyle='None')
    
        # all data points
        x_all = data['true'][0][:,0]
        y_true = data['true'][1]
        
        # plot true function
        plt.plot(x_all, y_true, "-", label='True function', color=true_function_color, linewidth=3, zorder=3)
        
        sorted_x, indices = torch.sort(x_values_samples, dim = 0)
        sorted_preds = y_preds[indices]

        # plot the predictions
        plt.plot(sorted_x, sorted_preds, label='MLP predictions', color=predicted_function_color, linestyle="--", linewidth=3, zorder=3)

        plt.grid(True, zorder=0, alpha=0.5)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend(loc='upper right')
        #plt.title("Prediction using MLP", fontsize=14, weight='bold')
        plt.tight_layout()

        if save:
            os.makedirs(self.RESULTSPATH, exist_ok=True)
            plt.savefig(f'{self.RESULTSPATH}/train_plot.png', dpi=300)

        plt.show()

    def plot_deepmimo(self, pred_sample=None, true_sample=None, save=False):

        # Set font size, grid, etc.
        plt.rcParams.update({
            'font.size': 15,
            'axes.labelsize': 15,
            'axes.titlesize': 15,
            'legend.fontsize': 15,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
        })


        reshape_dim = int(np.sqrt(pred_sample.shape[0]))
        y_pred = pred_sample
        prediction_reshaped = torch.rot90(y_pred.reshape(reshape_dim, reshape_dim), k=1, dims=(0,1))
        true_reshaped = torch.rot90(true_sample.reshape(reshape_dim, reshape_dim), k=1, dims=(0,1))

        # Create a figure for the plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot the prediction heatmap
        sns.heatmap(prediction_reshaped, ax=ax[0], cmap="viridis", cbar=True)
        ax[0].set_title("Prediction Heatmap")
        ax[0].set_xlabel("RX-antenna")      # Set x-axis label
        ax[0].set_ylabel("TX antenna")       # Set y-axis label

        # Plot the true values heatmap
        sns.heatmap(true_reshaped, ax=ax[1], cmap="viridis", cbar=True)
        ax[1].set_title("True Values Heatmap")
        ax[1].set_xlabel("RX-antenna")      # Set x-axis label
        ax[1].set_ylabel("TX antenna")       # Set y-axis label


        # Create a figure for the plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the prediction heatmap
        sns.heatmap(prediction_reshaped, ax=ax[0], cmap="viridis", cbar=True)
        ax[0].set_title("Prediction Heatmap MLP")

        # Plot the true values heatmap
        sns.heatmap(true_reshaped, ax=ax[1], cmap="viridis", cbar=True)
        ax[1].set_title("True Values Heatmap")

        # Adjust layout and display
        plt.tight_layout()

        if save:
            os.makedirs(self.RESULTSPATH, exist_ok=True)
            plt.savefig(f'{self.RESULTSPATH}/pred_heatmap_plot.png', dpi=300)

        plt.show()

    def plot_deepmimo_with_noise(self, true_sample=None, pred_sample=None,noisy_sample=None, save=False):

        # Set global font size and improve readability
        plt.rcParams.update({
            'font.size': 15,
            'axes.labelsize': 15,
            'axes.titlesize': 15,
            'legend.fontsize': 15,
            'axes.grid': True,  # Enable grid globally
            'grid.alpha': 0.3,   # Make grid lines subtle
            'axes.linewidth': 1.5,  # Thicker axis lines
            'xtick.major.width': 1.5, # Thicker x-tick lines
            'ytick.major.width': 1.5, # Thicker y-tick lines
        })


        reshape_dim = int(np.sqrt(pred_sample.shape[0]))
        y_pred = pred_sample
        noisy = noisy_sample
        noisy_reshaped = torch.rot90(noisy.reshape(reshape_dim, reshape_dim), k=1, dims=(0,1))
        prediction_reshaped = torch.rot90(y_pred.reshape(reshape_dim, reshape_dim), k=1, dims=(0,1))
        true_reshaped = torch.rot90(true_sample.reshape(reshape_dim, reshape_dim), k=1, dims=(0,1))

        # Create a figure for the plots
        fig, ax = plt.subplots(1, 3, figsize=(16, 5))        
        
        # Plot the prediction heatmap
        sns.heatmap(noisy_reshaped, ax=ax[0], cmap="viridis", cbar=True)

        
        ax[0].set_title("Noisy Heatmap")
        ax[0].set_xlabel("RX antenna")      # Set x-axis label
        ax[0].set_ylabel("TX antenna")       # Set y-axis label
      
        # Plot the prediction heatmap
        sns.heatmap(prediction_reshaped, ax=ax[1], cmap="viridis", cbar=True)
        ax[1].set_title("Prediction Heatmap")
        ax[1].set_xlabel("RX antenna")      # Set x-axis label
        ax[1].set_ylabel("TX antenna")       # Set y-axis label


        # Plot the true values heatmap
        sns.heatmap(true_reshaped, ax=ax[2], cmap="viridis", cbar=True)
        ax[2].set_title("True Values Heatmap")
        ax[2].set_xlabel("RX antenna")      # Set x-axis label
        ax[2].set_ylabel("TX antenna")       # Set y-axis label

        # Adjust layout and display
        plt.tight_layout()

        if save:
            os.makedirs(self.RESULTSPATH, exist_ok=True)
            plt.savefig(f'{self.RESULTSPATH}/pred__noisy_heatmap_plot.png', dpi=300)

        plt.show()


    def plot_loss(self, loss_data, save=False,deepmimo=False):
        """
        Plot the training and validation loss over epochs.
        """
        # Set font size, grid, etc.
        plt.rcParams.update({
            'font.size': 15,
            'axes.labelsize': 15,
            'axes.titlesize': 15,
            'legend.fontsize': 15,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.linewidth': 1.5,
            'xtick.major.width': 1.5,
            'ytick.major.width': 1.5,
        })

        # Define colors using viridis
        viridis = plt.cm.viridis
        loss_color_1 = viridis(0.5)
        loss_color_2 = viridis(0.2)

        # Convert loss data to a DataFrame for Seaborn
        loss_df = pd.DataFrame({
            'Epoch': range(1, len(loss_data['train_loss']) + 1),
            'Train Loss': loss_data['train_loss'],
            'Validation Loss': loss_data['test_loss']
        })

        # Ensure correct data types
        loss_df['Epoch'] = loss_df['Epoch'].astype(int)
        loss_df['Train Loss'] = loss_df['Train Loss'].astype(float)
        loss_df['Validation Loss'] = loss_df['Validation Loss'].astype(float)


        self.val_loss = loss_df['Validation Loss']
        self.train_loss = loss_df['Train Loss']
        
        # Melt the DataFrame for easier plotting with Seaborn
        #loss_melted = loss_df.melt(id_vars='Epoch', var_name='Loss Type', value_name='Loss')

        plt.grid(True, zorder=0, alpha=0.5)
        # Line plot for training and validation loss
        #sns.lineplot(data=loss_melted, x='Epoch', y='Loss', hue='Loss Type')
        plt.plot(loss_df['Epoch'], loss_df['Train Loss'], label='Train loss', color=loss_color_1, linewidth=3, zorder=3, linestyle='--')
        plt.plot(loss_df['Epoch'], loss_df['Validation Loss'], label='Validation loss', color=loss_color_2, linewidth=3, zorder=3)

        # Set labels and title
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        #plt.title("Training and Validation Loss Over Epochs", fontsize=14, weight='bold')

        # Customize legend
        plt.legend(loc='upper right')

        # Adjust layout for better spacing
        plt.tight_layout()

        if save:
            os.makedirs(self.RESULTSPATH, exist_ok=True)
            plt.savefig(f'{self.RESULTSPATH}/loss.png', dpi=300)
            print("saved loss to ", f'{self.RESULTSPATH}/loss.png')

        plt.show()

    def write_params_to_file(self, extra_params=None):
        file_path = f'{self.RESULTSPATH}/model_params.txt'
        import os
        os.makedirs(self.RESULTSPATH, exist_ok=True)
        with open(file_path, "w") as file:
            # write cfg
            file.write(f"input size: {self.input_size}\n")
            file.write(f"hidden-layers: {self.hidden_sizes}\n")
            file.write(f"output size: {self.output_size}\n")
            file.write(f"final validation loss: {self.val_loss_list[-1]}\n")
            file.write(f"final training loss: {self.loss_list[-1]}\n")

        if extra_params:
            with open(file_path, "a") as file:
                for key, value in extra_params.items():
                    file.write(f"{key}: {value}\n")
                
        print(f"Model parameters saved to {file_path}")
        #np.savez(f'{RESULTSPATH}/plots.npz', epoch=range(1, len(results['train_loss']) + 1), train_loss = results['train_loss'], val_loss = results['test_loss'] , X_val=X_val, y_noise_val=y_noise_val , X_tot = X_tot, y_true_tot = y_true_tot, sorted_X = sorted_X, sorted_KAN_preds = sorted_KAN_preds)
