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

## Data
data = pd.read_csv("./datasets/data2.csv")  #/datasets/ #/Users/zeshenbao/KAN-Kolmogorov-Arnold-Networks

X = torch.tensor(data["x"].values).float().unsqueeze(1)
y = torch.tensor(data["y"].values).float().unsqueeze(1)


dataset = create_dataset_from_data(X, y)


kan_model = KAN(width=[1, 10, 10, 1], grid=3, k=5, seed=0)

kan_model.fit(dataset, opt="LBFGS", steps=5, lamb=0.001, lamb_entropy=10.);

KAN_preds=kan_model(dataset['test_input']).detach()

##Sort before plot

def plot_results(x_test, y_pred, x_tot, y_tot, show_model=False, threshold = 0.3):
    
    
    sorted_indices_test = np.argsort(x_test, axis=0)
    sorted_indices_all = np.argsort(x_tot, axis=0)

    plot_array_test = torch.gather(x_test, dim=0, index=sorted_indices_test)
    plot_array_pred = torch.gather(y_pred, dim=0, index=sorted_indices_test)

    plot_array_x_tot = torch.gather(x_tot, dim=0, index=sorted_indices_all)
    plot_array_y_tot = torch.gather(y_tot, dim=0, index=sorted_indices_all)


    plt.plot(plot_array_x_tot, plot_array_y_tot, "o", markersize=1, linestyle='None', label="Data")
    plt.plot(plot_array_test, plot_array_pred, "--",label='KAN predictions')
    

    plt.xlabel("Random X 1D samples")
    plt.ylabel("Function")
    plt.legend()

    plt.show()

    if show_model is True:
        kan_model = kan_model.prune_node(threshold=threshold)
        kan_model.plot()
        plt.show()



x_test = dataset['test_input']
y_pred = KAN_preds


plot_results(x_test, y_pred, X, y)


