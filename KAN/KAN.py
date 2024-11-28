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

    def __init__(self, data=None, width=[1, 3, 3, 1], grid=3, k=5, seed=42, lr=0.001, lamb=0.01):
        """
        Initialize the KAN model with the desired hyperparameters.

        Parameters:
        - width (list): Architecture width parameters.
        - grid (int): Grid size parameter.
        - k (int): Parameter k.
        - seed (int): Random seed.
        """
        self.data = data
        self.width = width
        self.grid = grid
        self.k = k
        self.seed = seed
        self.lr = lr 
        self.lamb = lamb
        self.X_train =  self.data['train'][0][:,1].unsqueeze(1)           # get the y_noise
        self.y_train = self.data['train'][1].unsqueeze(1)
        self.X_validation =  self.data['validation'][0][:,1].unsqueeze(1) # get the y_noise
        self.y_validation = self.data['validation'][1].unsqueeze(1)
        self.dataset = {"train_input": self.X_train, "train_label":self.y_train, "test_input":self.X_validation, "test_label":self.y_validation}

        
        # Initialize the actual KAN model with the parameters
        self.model = KAN(width=self.width, grid=self.grid, k=self.k, seed=self.seed)

        
    def fit(self, X, y):
        """
        Fit the KAN model to the training data.

        Parameters:
        """
        _dataset = self.dataset
        _dataset['train_input'] = torch.tensor(X).float()
        _dataset['train_label'] = torch.tensor(y).float()
        self.model.fit(_dataset, opt="LBFGS", steps=50, lr=self.lr, lamb=self.lamb)
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
            'data': self.data,
            'width': self.width,
            'grid': self.grid,
            'k': self.k,
            'seed': self.seed,
            'lr': self.lr,
            'lamb': self.lamb,
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