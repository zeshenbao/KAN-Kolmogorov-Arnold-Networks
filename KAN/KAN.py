import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import neural_network,pipeline,preprocessing,linear_model
import torch
import pandas as pd
from kan import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


torch.manual_seed(0)
import os
print(os.getcwd())

data = pd.read_csv("/Users/zeshenbao/KAN-Kolmogorov-Arnold-Networks/datasets/data2.csv")  #/datasets/

X = torch.tensor(data["x"].values).float().unsqueeze(1)
y = torch.tensor(data["y"].values).float().unsqueeze(1)

plt.plot(X, y, "o", markersize=1, linestyle='None')


dataset = create_dataset_from_data(X, y)

#def KAN input, output

kan_model = KAN(width=[1, 10, 10, 1], grid=3, k=5, seed=0)

kan_model.fit(dataset, opt="LBFGS", steps=20, lamb=0.001, lamb_entropy=10.);

KAN_preds=kan_model(dataset['test_input']).detach().numpy()

#plt.plot(X, y,".-",label='ground_truth')
plt.scatter(dataset['test_input'], KAN_preds,label='KAN predictions')
plt.plot(X, y, "o", markersize=1, linestyle='None')
plt.xlabel("Random X 1D samples")
plt.ylabel("Function")
plt.legend()

#print(f"KAN MSE is {sklearn.metrics.mean_squared_error(y, KAN_preds)}")


kan_model.plot()

plt.show()