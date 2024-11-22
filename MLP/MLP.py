# Madeleine Lindström, madeli@kth.se

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

<<<<<<< HEAD

=======
>>>>>>> 439397e9586a8c17d0da39a554d148c98638a238
class MLP(nn.Module):
    """Creates a Multilayer Perceptron."""

    def __init__(self, input_size=1, hidden_sizes=[1], output_size=1, lr=0.01):
        super(MLP, self).__init__()
        self.layers = list()

        i_size = input_size

        for h_size in hidden_sizes:
            self.layers.append(nn.Linear(i_size, h_size))
            self.layers.append(nn.ReLU())  # ReLU on all layers
            i_size = h_size
        
        self.lr = lr
        self.layers.append(nn.Linear(i_size, output_size))
        self.layers = nn.Sequential(*self.layers)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Adam optimzier. Alternatively: optim.STD (Stochastic Gradient Descent)
        self.loss_function = nn.MSELoss()  # Mean Squared Error Loss. Alteratively: L1Loss, CrossEntropyLoss
        self.epoch_list = list()
        self.loss_list = list()
        self.val_loss_list = list()
        #self.validation_x = list()
        #self.validation_y = list()

    
    def forward(self, x):
        #print("forward\nforward\nforward\nforward\nforward\nforward\n", type(x), x.shape)
        if type(x) == torch.Tensor:
            x.clone().float()
        else:    
            x = torch.tensor(x).float()
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
            #print("fit\nfit\nfit\nfit\nfit\nfit\nfit\n", type(x), x.shape, y.shape)
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
                #print("härval\nhärval\nhärval\nhärval\nhärval\nhärval\nhärval\nhärval\nhärval\n",type(self.validation_x), type(self.validation_y))
                #
                # print("shapes:  ", len(self.validation_y), len(self.validation_x))
                validation_output = self.forward(self.validation_x)
                validation_loss = torch.sqrt(self.loss_function(validation_output, self.validation_y))  # sqrt to get RMSE instead of MSE
                self.val_loss_list.append(validation_loss.item())

    def set_val_data(self, x, y):
        self.validation_x = x
        self.validation_y = y




######### CROSS VAL

class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, datasetPath, input_size=1, hidden_sizes=[1], output_size=1, lr=0.01, steps = 10000):
        """
        Initialize the MLP model with the desired hyperparameters.

        Parameters:
        - input_size (int): Architecture input param.
        - hidden_sizes (list): Architecture hidden layer parameters.
        - output_size (int): Architecture output param.
        """
        self.datasetPath = datasetPath
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.steps = steps
        self.lr = lr 

        # Set test dataset #Should be same for MLP
        val_data = pd.read_csv(self.datasetPath)
        X_val = torch.tensor(val_data['x']).float().unsqueeze(1)
        y_noise_val = torch.tensor(val_data['y_noise']).float().unsqueeze(1)
        y_true_val = torch.tensor(val_data['y_true']).float().unsqueeze(1)
        self.y_noise_val =  y_noise_val
        self.y_true_val = y_true_val
        self.X_val = X_val


        # Initialize the actual MLP model with the parameters
        self.model = MLP(input_size=1, hidden_sizes=[1], output_size=1, lr=self.lr)
        self.model.set_val_data(self.X_val, self.y_noise_val)

    def fit(self, X, y):
        """
        Fit the MLP model to the training data.

        Parameters:
        - X: Training features (torch.Tensor).
        - y: Training labels (torch.Tensor).
        """
        if type(X) == torch.Tensor:
            _X = X.clone().float()
        else:
            _X = torch.tensor(X).float()
            
        if type(y) == torch.Tensor:
            _y = y.clone().float()
        else:
            #print("type\ntype\ntype\ntype\n",type(y))
            _y = torch.tensor(y).float()
            

        

        #dataset = {"train_input": X, "train_label": y, "test_input": self.y_noise_val, "test_label": self.y_true_val}
        self.model.fit(_X, _y, n_epochs=self.steps)
        return self

    def predict(self, X):
        """
        Predict using the trained MLP model.

        Parameters:
        - X: Input features for prediction (torch.Tensor).

        Returns:
        - Predictions (torch.Tensor).
        """
        #_X = torch.tensor(X).float()
        return self.model(X).detach()

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns:
        - Dictionary of parameter names mapped to their values.
        """
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'lr': self.lr,
            'datasetPath': self.datasetPath
        }

    def set_params(self, **parameters):
        """
        Set the parameters of this estimator.

        Parameters:
        - **parameters: Estimator parameters.

        Returns:
        - self
        """
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
        # Re-initialize the model with updated parameters
        self.model = MLP(input_size=self.input_size, hidden_sizes=self.hidden_sizes, output_size=self.output_size, lr=self.lr)
        self.model.set_val_data(self.X_val, self.y_noise_val)
        return self



def find_best_params(datasetPath : str, param_grid : dict):
    # Read data
    train_data = pd.read_csv(datasetPath)
    #val_data = read_data(f"./datasets/uniform_sin(x)_241114/validation_data.csv")
    #test_data = read_data(f"./datasets/uniform_sin(x)_241114/test_data.csv")

    X, y = torch.tensor(train_data['y_noise']).float().unsqueeze(1), torch.tensor(train_data['y_true']).float().unsqueeze(1)


    # Initialize MLP model
    mlp_wrapper = MLPWrapper(datasetPath=datasetPath)

    # Define a parameter grid
    param_grid = param_grid if param_grid is not None else {}

    param_grid = {
        'mlp__input_size': [1],
        'mlp__input_size': [[20,10], [20]],
        'mlp__input_size': [1],
        'mlp__lr': [0.01, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006],
        'mlp__datasetPath': [datasetPath]
    }
    

    # (Optional) Create a pipeline if preprocessing is needed
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Example preprocessor
        ('mlp', mlp_wrapper)
    ])

    """
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  # 5-Fold Cross-Validation
        scoring='neg_mean_squared_error',  # Use appropriate regression metric
        n_jobs=-1,  # Utilize all available CPU cores
        verbose=2,
    )

    # Fit GridSearchCV
    grid_search.fit(X, y)
    """
    # Initialize RandomizedSearchCV
    grid_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=10,  # Number of parameter settings sampled
        cv=5,        # 5-Fold Cross-Validation
        scoring='neg_mean_squared_error',  # Appropriate for regression
        random_state=42,                    # For reproducibility
        n_jobs=-1,                          # Utilize all CPU cores
        verbose=2
    )

    # Fit RandomizedSearchCV
    grid_search.fit(X, y)

    # Retrieve the best parameters and best score
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Score:", grid_search.best_score_)


if __name__ == "__main__":
    find_best_params(datasetPath="./datasets/uniform_sin(x)_241114/train_data.csv", param_grid=None)

