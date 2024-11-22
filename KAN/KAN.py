from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import neural_network,pipeline,preprocessing,linear_model
import torch
import pandas as pd
import seaborn as sns
from kan import *
import time
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.manual_seed(0)

class KANWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, datasetPath, width=[1, 3, 3, 1], grid=3, k=5, seed=42, lr=None, lamb=None):
        """
        Initialize the KAN model with the desired hyperparameters.

        Parameters:
        - width (list): Architecture width parameters.
        - grid (int): Grid size parameter.
        - k (int): Parameter k.
        - seed (int): Random seed.
        """
        self.datasetPath = datasetPath
        self.width = width
        self.grid = grid
        self.k = k
        self.seed = seed
        self.lr = lr 
        self.lamb = lamb
        # Set test dataset 
        val_data = pd.read_csv(f"{self.datasetPath}/validation_data.csv")
        X_val = torch.tensor(val_data['x']).float().unsqueeze(1)
        y_noise_val = torch.tensor(val_data['y_noise']).float().unsqueeze(1)
        y_true_val = torch.tensor(val_data['y_true']).float().unsqueeze(1)
        self.y_noise_val =  y_noise_val
        self.y_true_val = y_true_val


        # Initialize the actual KAN model with the parameters
        self.model = KAN(width=self.width, grid=self.grid, k=self.k, seed=self.seed)

    def fit(self, X, y):
        """
        Fit the KAN model to the training data.

        Parameters:
        - X: Training features (torch.Tensor).
        - y: Training labels (torch.Tensor).
        """
        _X = torch.tensor(X).float()
        #_y = torch.tensor(y).float().unsqueeze(1)
        _y = y

        dataset = {"train_input": _X, "train_label": _y, "test_input": self.y_noise_val, "test_label": self.y_true_val}
        self.model.fit(dataset, opt="LBFGS", steps=20, lr=self.lr, lamb=self.lamb)
        return self

    def predict(self, X):
        """
        Predict using the trained KAN model.

        Parameters:
        - X: Input features for prediction (torch.Tensor).

        Returns:
        - Predictions (torch.Tensor).
        """
        _X = torch.tensor(X).float()
        return self.model(_X).detach()

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns:
        - Dictionary of parameter names mapped to their values.
        """
        return {
            'width': self.width,
            'grid': self.grid,
            'k': self.k,
            'seed': self.seed,
            'lr': self.lr,
            'lamb': self.lamb,
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
        self.model = KAN(width=self.width, grid=self.grid, k=self.k, seed=self.seed)
        return self


def find_best_params(datasetPath : str, param_grid : dict):
    # Read data
    train_data = pd.read_csv(f"{datasetPath}/train_data.csv")
    #val_data = read_data(f"./datasets/uniform_sin(x)_241114/validation_data.csv")
    #test_data = read_data(f"./datasets/uniform_sin(x)_241114/test_data.csv")

    X, y = torch.tensor(train_data['y_noise']).float().unsqueeze(1), torch.tensor(train_data['y_true']).float().unsqueeze(1)


    # Initialize KAN model
    kan_wrapper = KANWrapper(datasetPath=datasetPath)

    # Define a parameter grid
    param_grid = param_grid if param_grid is not None else {}

    if param_grid is None:
        param_grid = {
            'kan__width': [[1, 3, 3, 1], [1, 5, 5, 1]],
            'kan__grid': [5],
            'kan__k': [3],
            'kan__seed': [42],
            'kan__lr': [0.01, 0.001],
            'kan__lamb': [0.0, 0.1, 0.2],
            'kan__datasetPath': [datasetPath]
        }
    

    # (Optional) Create a pipeline if preprocessing is needed
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Example preprocessor
        ('kan', kan_wrapper)
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
        n_iter=1,  # Number of parameter settings sampled
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

    return grid_search.best_params_


if __name__ == "__main__":
    datasetPath="./datasets/uniform_sin(x)_241121"
    param_grid = {
        'kan__width': [[1, 3, 3, 1]],
        'kan__grid': [5],
        'kan__k': [3],
        'kan__seed': [42],
        'kan__lr': [0.01],
        'kan__lamb': [0.0],
        'kan__datasetPath': [datasetPath]
    }
    params = find_best_params(datasetPath=datasetPath, param_grid=param_grid)
    import KAN_run as kan_eval
    kan_eval.train_and_evaluate(params, datasetPath, funcName="uniform_sin(x)_241121")

