from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import neural_network,pipeline,preprocessing,linear_model
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from kan import *
import time
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from MLP import MLP

class MLPWrapper(BaseEstimator, RegressorMixin):
    def __init__(self, data, input_size=1, hidden_sizes=[1], output_size=1, steps=800):
        """
        Initialize the MLP model with the desired hyperparameters.

        Parameters:
        - input_size (int): Architecture input param.
        - hidden_sizes (list): Architecture hidden layer parameters.
        - output_size (int): Architecture output param.
        """
        self.data = data
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        self.steps = steps

        self.X_train =  self.data['train'][0][:,1].unsqueeze(1)           # get the y_noise
        self.y_train = self.data['train'][1].unsqueeze(1)
        self.X_validation =  self.data['validation'][0][:,1].unsqueeze(1) # get the y_noise
        self.y_validation = self.data['validation'][1].unsqueeze(1)
        self.dataset = {"train_input": self.X_train, "train_label":self.y_train, "test_input":self.X_validation, "test_label":self.y_validation}

        # Initialize the actual MLP model with the parameters
        self.model = MLP(input_size=1, hidden_sizes=[1], output_size=1)

    def fit(self, X, y):
        """
        Fit the MLP model to the training data.

        Parameters:
        - X: Training features (torch.Tensor).
        - y: Training labels (torch.Tensor).
        """
        X_ = torch.tensor(X).float()
        y_ = torch.tensor(y).float()
        self.model.fit(X_, y_, n_epochs=self.steps, cross_validation=True)
        return self

    def predict(self, X):
        """
        Predict using the trained MLP model.

        Parameters:
        - X: Input features for prediction (torch.Tensor).

        Returns:
        - Predictions (torch.Tensor).
        """
        return self.model.predict(X).detach()

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Returns:
        - Dictionary of parameter names mapped to their values.
        """
        return {
            'data' : self.data,
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'steps': self.steps
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
        self.model = MLP(input_size=self.input_size, hidden_sizes=self.hidden_sizes, output_size=self.output_size)
        return self
