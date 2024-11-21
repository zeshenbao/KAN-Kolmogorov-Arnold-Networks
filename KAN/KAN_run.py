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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.manual_seed(0)


def read_data(filepath:str) -> pd.DataFrame:
    return pd.read_csv(filepath)


def basic_fit(cfg : dict) -> dict:

    folder_name = cfg["folder_name"]
    datasetPath = cfg["datasetPath"]
    RESULTSPATH = f"./results/{folder_name}/KAN"
    # create dataset
    train_data = read_data(f"{datasetPath}/train_data.csv")
    val_data = read_data(f"{datasetPath}/validation_data.csv")
    test_data = read_data(f"{datasetPath}/test_data.csv")
    total_data = read_data(f"{datasetPath}/true_data.csv")
    ## read params
    layers = cfg["layers"]
    grid = cfg["grid"]  # nr of spline grids
    k = cfg["k"] #order of spline
    seed = cfg.get("seed", 0) #random seed
    opt = cfg.get("opt", "Adam") #optimizer
    steps = cfg.get("steps", 800) #nr of steps
    lr = cfg.get("lr", 1e-3)#learning rate
    lamb = cfg.get("lamb", 0.0) #regularization parameter
    ## ----------------------------------------
    os.makedirs(f'{RESULTSPATH}', exist_ok=True)

    # Prepare data
    X_tot = torch.tensor(total_data['x']).float().unsqueeze(1)
    #y_noise_tot = torch.tensor(total_data['y_noise']).float()#.unsqueeze(1)
    y_true_tot = torch.tensor(total_data['y_true']).float().unsqueeze(1)

    X_train = torch.tensor(train_data['x']).float().unsqueeze(1)
    y_noise_train = torch.tensor(train_data['y_noise']).float().unsqueeze(1)
    y_true_train = torch.tensor(train_data['y_true']).float().unsqueeze(1)

    X_val = torch.tensor(val_data['x']).float().unsqueeze(1)
    y_noise_val = torch.tensor(val_data['y_noise']).float().unsqueeze(1)
    y_true_val = torch.tensor(val_data['y_true']).float().unsqueeze(1)

    X_test = torch.tensor(test_data['x']).float().unsqueeze(1)
    y_noise_test = torch.tensor(test_data['y_noise']).float().unsqueeze(1)
    y_true_test = torch.tensor(test_data['y_true']).float().unsqueeze(1)

    #X_test = torch.tensor(test_data['x']).float().unsqueeze(1)
    #y_noise_test = torch.tensor(test_data['y_noise']).float().unsqueeze(1)
    #y_true_test = torch.tensor(test_data['y_true']).float().unsqueeze(1)

    dataset = {"train_input": y_noise_train, "train_label":y_true_train, "test_input":y_noise_val, "test_label":y_true_val}


    # Train model
    start = time.time()
    kan_model = KAN(width=layers, grid=grid, k=k, seed=seed)
    start = time.time()
    results = kan_model.fit(dataset, opt=opt, steps=steps, lr=lr , lamb=lamb)
    end = time.time()
    elapsed_time = end - start

    # Generate predictions
    KAN_preds = kan_model(dataset['test_input']).detach()
    print(f"validation pred shape: {KAN_preds.shape}")
    """
    Plot the validation data and predictions
    """
    # Set Seaborn theme
    sns.set_theme(style="whitegrid")
    

    # plot noisy validation
    plt.plot(X_val, y_noise_val, "o", markersize=1, linestyle='None', label="Validation data")
    # plot true function
    plt.plot(X_tot, y_true_tot, "-",label='True function')
    
    sorted_X, indices = torch.sort(X_val, dim = 0)
    sorted_KAN_preds = KAN_preds[indices][:,:,0]
    plt.plot(sorted_X, sorted_KAN_preds, "--", label='KAN predictions')
    plt.xlabel("Random X 1D samples")
    plt.ylabel("Function")
    plt.legend()
    plt.title("Prediction using KAN", fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(f'{RESULTSPATH}/train_plot.png', dpi=300)

    plt.clf()

    """
    Plot the training and validation loss over epochs.
    """
    # Convert loss data to a DataFrame for Seaborn
    loss_df = pd.DataFrame({
        'Epoch': range(1, len(results['train_loss']) + 1),
        'Train Loss': results['train_loss'],
        'Validation Loss': results['test_loss']
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

    plt.savefig(f'{RESULTSPATH}/loss.png', dpi=300)
    print("saved loss to ", f'{RESULTSPATH}/loss.png')
    print(f"Elapsed Time: {elapsed_time:.3f} seconds")

    print(loss_df['Validation Loss'].iloc[-1])

    plt.clf()

    """
    Plot the test data and predictions
    """
    KAN_preds = kan_model(y_noise_test).detach()
    # calculate loss
    loss = cfg.get("loss_fn", torch.nn.MSELoss())
    test_loss = torch.sqrt(loss(KAN_preds, y_true_test))

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")
    
    #sorted_X, indices = torch.sort(X_test, dim = 0)
    #sorted_y_true = y_true_test[indices][:, :, 0]
    # plot noisy validation
    plt.plot(X_test, y_noise_test, "o", markersize=1, linestyle='None', label="Test data")
    # plot true function
    plt.plot(X_tot, y_true_tot, "-",label='True function')
    
    
    sorted_X, indices = torch.sort(X_test, dim = 0)
    sorted_KAN_preds = KAN_preds[indices][:, :, 0]

    plt.plot(sorted_X, sorted_KAN_preds, "--", label='KAN predictions')
    plt.xlabel("Random X 1D samples")
    plt.ylabel("Function")
    plt.legend()
    plt.title("Prediction using KAN", fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(f'{RESULTSPATH}/test_plot.png', dpi=300)

    plt.clf()

    
    
    file_path = f'{RESULTSPATH}/model_params.txt'

    with open(file_path, "w") as file:
        # write cfg
        file.write(f"width: {layers}\n")
        file.write(f"grid: {grid}\n")
        file.write(f"k: {k}\n")
        file.write(f"seed: {seed}\n")
        file.write(f"opt: {opt}\n")
        file.write(f"steps: {steps}\n")
        file.write(f"lr: {lr}\n")
        file.write(f"lamb: {lamb}\n")
        file.write(f"final validation loss: {loss_df['Validation Loss'].iloc[-1]}\n")
        file.write(f"final training loss: {loss_df['Train Loss'].iloc[-1]}\n")
        file.write(f"TEST LOSS: {test_loss}\n")
            
    print(f"Model parameters saved to {file_path}")
    np.savez(f'{RESULTSPATH}/plots.npz', epoch=range(1, len(results['train_loss']) + 1), train_loss = results['train_loss'], val_loss = results['test_loss'] , X_val=X_val, y_noise_val=y_noise_val , X_tot = X_tot, y_true_tot = y_true_tot, sorted_X = sorted_X, sorted_KAN_preds = sorted_KAN_preds)
        
    return results


def train_and_evaluate(bestParams : dict, datasetPath : str, funcName : str = None):
    """
    Takes in a dictionary of hyperparameters and a dataset path, trains a model, plots training 
    results, saves parameters to a file, then evaluates the model on the test set. Again saving the results
    """
    if funcName is None:
        funcName = datasetPath.split("/")[-1]

    # can also set loss_fn in cfg

    cfg = {
        "folder_name": funcName,
        "datasetPath" : datasetPath,
        "layers" : bestParams.get("kan__width"),
        "grid" : bestParams.get("kan__grid"),
        "k" : bestParams.get("kan__k"),
        "lr": bestParams.get("kan__lr"),
        "steps": 200,
        "lamb": bestParams.get("kan__lamb")
    }
    print(cfg)
    print(f"Running model for {cfg['folder_name']}")
    basic_fit(cfg)