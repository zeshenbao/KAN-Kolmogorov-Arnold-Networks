import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torch
import pandas as pd
import seaborn as sns
from kan import *
import time
from sklearn.preprocessing import MinMaxScaler


class KANModel():
    def __init__(self, results_path=None, width=[1, 3, 3, 1], grid=3, k=5, seed=42, lr=0.001, lamb=0.01, deepmimo=False,steps=10):
        self.RESULTSPATH = results_path
        self.deepmimo = deepmimo
        self.width = width
        self.grid = grid
        self.k = k
        self.seed = seed
        self.lr = lr 
        self.lamb = lamb
        self.steps = steps

        self.model = KAN(width=self.width, grid=self.grid, k=self.k, seed=self.seed)

    def load_data(self, data):
        if self.deepmimo:
            self.X_train =  data['train'][0]           
            self.y_train = data['train'][1]
            self.X_validation =  data['test'][0]
            self.y_validation = data['test'][1]
            self.X_test =  data['test'][0]
            self.y_test = data['test'][1]
            self.dataset = {"train_input": self.X_train, "train_label":self.y_train, "test_input":self.X_test, "test_label":self.y_test}
        else:
            self.X_train =  data['train'][0][:,1].unsqueeze(1)          # get the y_noise
            self.y_train = data['train'][1].unsqueeze(1)
            self.X_validation =  data['validation'][0][:,1].unsqueeze(1) # get the y_noise
            self.y_validation = data['validation'][1].unsqueeze(1)
            self.X_test =  data['test'][0][:,1].unsqueeze(1)            # get the y_noise
            self.y_test = data['test'][1].unsqueeze(1)
            self.dataset = {"train_input": self.X_train, "train_label":self.y_train, "test_input":self.X_validation, "test_label":self.y_validation}

    def fit(self):
        start = time.time()
        results = self.model.fit(self.dataset, opt="LBFGS", steps=self.steps, lr=self.lr , lamb=self.lamb)
        end = time.time()

        elapsed_time = end - start
        return results, elapsed_time

    def predict(self):
        """
        Returns predictions made by the model for self.X_test, and the loss: RMSE
        """
        loss = torch.nn.MSELoss()
        y_pred = self.model(self.X_test).detach()
        results = {"preds": y_pred, "test_loss": torch.sqrt(loss(y_pred, self.y_test))}
        return results

    def plot_prediction(self, data, y_preds, type_='test', save=False):
        """
        Plot predictions made by the model.
        """
        sns.set_theme(style="whitegrid")

        # samples
        x_values_samples = data[type_][0][:,0]
        y_noise = data[type_][0][:,1]

        # plot noisy datapoints
        plt.plot(x_values_samples, y_noise, "o", markersize=1, linestyle='None', label=f"{type_} data")
    
        # all data points
        x_all = data['true'][0][:,0]
        y_true = data['true'][1]
        
        # plot true function
        plt.plot(x_all, y_true, "-",label='True function')
        
        sorted_x, indices = torch.sort(x_values_samples, dim = 0)
        sorted_preds = y_preds[indices]

        # plot the predictions
        plt.plot(sorted_x, sorted_preds, "--", label='KAN predictions')

        plt.xlabel("Random X 1D samples")
        plt.ylabel("Function")
        plt.legend()
        plt.title("Prediction using KAN", fontsize=14, weight='bold')
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.RESULTSPATH}/train_plot.png', dpi=300)

        plt.show()

    def plot_deepmimo(self, data, y_preds, type_='test', save=False):

        reshape_dim = int(np.sqrt(y_preds.shape[1]))
        print(reshape_dim)
        # mean all preds as the y_true is same for all
        y_pred = torch.mean(y_preds, dim=0, keepdim=True)
        print(y_pred.shape)
        # Reshape to 64x64, discard the extra element
        prediction_reshaped = y_pred[0,:].reshape(reshape_dim, reshape_dim)
        true_reshaped = data['test'][1][0,:].reshape(reshape_dim, reshape_dim)

        scaler = MinMaxScaler()
        prediction_reshaped = scaler.fit_transform(prediction_reshaped)
        true_reshaped = scaler.fit_transform(true_reshaped)


        # Create a figure for the plots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Plot the prediction heatmap
        sns.heatmap(prediction_reshaped, ax=ax[0], cmap="viridis", cbar=True)
        ax[0].set_title("Prediction Heatmap")

        # Plot the true values heatmap
        sns.heatmap(true_reshaped, ax=ax[1], cmap="viridis", cbar=True)
        ax[1].set_title("True Values Heatmap")

        # Adjust layout and display
        plt.tight_layout()

        if save:
            plt.savefig(f'{self.RESULTSPATH}/pred_heatmap_plot.png', dpi=300)

        plt.show()



    def plot_loss(self, loss_data, save=False):
        """
        Plot the training and validation loss over epochs.
        """
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
        
        # Melt the DataFrame for easier plotting with Seaborn
        loss_melted = loss_df.melt(id_vars='Epoch', var_name='Loss Type', value_name='Loss')

        # Line plot for training and validation loss
        sns.lineplot(data=loss_melted, x='Epoch', y='Loss', hue='Loss Type')

        # Set labels and title
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training and Validation Loss Over Epochs", fontsize=14, weight='bold')

        # Customize legend
        plt.legend(title='Loss Type', fontsize=10, title_fontsize=12)

        # Adjust layout for better spacing
        plt.tight_layout()

        if save:
            plt.savefig('../results/loss.png', dpi=300)
            print("saved loss to ", f'{ self.RESULTSPATH}/loss.png')

        plt.show()

    def write_params_to_file(self, extra_params=None):
        """
        Writes current cfg to a file, saves training parameters and results to a .npz file. 
        Argument extra_params is a dictionary of extra parameters to save.
        """
        file_path = f'{self.RESULTSPATH}/model_params.txt'
        import os
        os.makedirs(self.RESULTSPATH, exist_ok=True)
        with open(file_path, "w") as file:
            # write cfg
            file.write(f"width: {self.width}\n")
            file.write(f"grid: {self.grid}\n")
            file.write(f"k: {self.k}\n")
            file.write(f"seed: {self.seed}\n")
            file.write(f"lr: {self.lr}\n")
            file.write(f"lamb: {self.lamb}\n")
                 # Write extra parameters if provided
            if extra_params:
                for key, value in extra_params.items():
                    file.write(f"{key}: {value}\n")

        print(f"Model parameters saved to {file_path}")
        #np.savez(f'{self.RESULTSPATH}/plots.npz', epoch=range(1, len(results['train_loss']) + 1), train_loss = results['train_loss'], val_loss = results['test_loss'] , X_val=X_val, y_noise_val=y_noise_val , X_tot = X_tot, y_true_tot = y_true_tot, sorted_X = sorted_X, sorted_KAN_preds = sorted_KAN_preds)
            