import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import neural_network,pipeline,preprocessing,linear_model
import torch
import pandas as pd
import seaborn as sns
from kan import *
import time


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.manual_seed(0)


def read_data(filepath:str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def basic_fit(folder_name : str, cfg : dict) -> dict:
    # create dataset
    train_data = read_data(f"./datasets/{folder_name}/train_data.csv")
    val_data = read_data(f"./datasets/{folder_name}/validation_data.csv")
    test_data = read_data(f"./datasets/{folder_name}/test_data.csv")
    total_data = read_data(f"./datasets/{folder_name}/true_data.csv")
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
    results_folder_name = f'{folder_name}_{layers}_241114' #save folder name, one for each model run
    os.makedirs(f'./KAN/results/{results_folder_name}', exist_ok=True)

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

    #X_test = torch.tensor(test_data['x']).float().unsqueeze(1)
    #y_noise_test = torch.tensor(test_data['y_noise']).float().unsqueeze(1)
    #y_true_test = torch.tensor(test_data['y_true']).float().unsqueeze(1)

    dataset = {"train_input": y_noise_train, "train_label":y_true_train, "test_input":y_noise_val, "test_label":y_true_val}


    # Train model
    dataset_input = "y_noise, y_true"
    
    kan_model = KAN(width=layers, grid=grid, k=k, seed=seed)
    start = time.time()
    results = kan_model.fit(dataset, opt=opt, steps=steps, lr=lr , lamb=lamb)


    end = time.time()
    elapsed_time = end - start

    # Generate predictions
    KAN_preds = kan_model(dataset['test_input']).detach()

    print("KAN_preds",KAN_preds.shape)
    # Debugging: Inspect results
    print("Keys in results:", results.keys())

    # Verify that 'train_loss' and 'test_loss' are present and are lists
    required_keys = ['train_loss', 'test_loss']
    for key in required_keys:
        if key not in results:
            raise KeyError(f"The 'results' dictionary must contain the '{key}' key.")
        if not isinstance(results[key], list):
            raise TypeError(f"'{key}' should be a list.")
    
    if len(results['train_loss']) != len(results['test_loss']):
        raise ValueError("'train_loss' and 'test_loss' must be of the same length.")

    """
    Plot the data, true function, and KAN predictions.
    """
    # Set Seaborn theme
    sns.set_theme(style="whitegrid")
    
    # --- Plot 1: Data and Predictions ---
    print("Test Input Shape:", dataset['test_input'].shape)
    print("Test Label Shape:", dataset['test_label'].shape)
    print("KAN Predictions Shape:", KAN_preds.shape)

    plt.plot(X_val, y_noise_val, "o", markersize=1, linestyle='None', label="Validation data")
    plt.plot(X_tot, y_true_tot, "-",label='True function')
    
    sorted_X, indices = torch.sort(X_val, dim = 0)
    sorted_KAN_preds = KAN_preds[indices][:,:,0]
    plt.plot(sorted_X, sorted_KAN_preds, "--", label='KAN predictions')
    plt.xlabel("Random X 1D samples")
    plt.ylabel("Function")
    plt.legend()
    plt.title("Prediction using KAN", fontsize=14, weight='bold')

    plt.tight_layout()
    plt.savefig(f'./KAN/results/{results_folder_name}/plot.png', dpi=300)

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

    # Debugging: Inspect the melted DataFrame
    print("Melted Loss DataFrame:")
    print(loss_melted.head())

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

    plt.savefig(f'./KAN/results/{results_folder_name}/loss.png', dpi=300)

    print(f"Elapsed Time: {elapsed_time:.3f} seconds")

    print(loss_df['Validation Loss'].iloc[-1])

    
    
    file_path = f'./KAN/results/{results_folder_name}/model_params.txt'

    with open(file_path, "w") as file:
        file.write(f"input to model: {dataset_input}\n")
        # write cfg
        file.write(f"layers: {layers}\n")
        file.write(f"grid: {grid}\n")
        file.write(f"k: {k}\n")
        file.write(f"seed: {seed}\n")
        file.write(f"opt: {opt}\n")
        file.write(f"steps: {steps}\n")
        file.write(f"lr: {lr}\n")
        file.write(f"lamb: {lamb}\n")
        file.write(f"final validation loss: {loss_df['Validation Loss'].iloc[-1]}\n")
        file.write(f"final training loss: {loss_df['Train Loss'].iloc[-1]}\n")
            
    print(f"Model parameters saved to {file_path}")
    np.savez(f'./KAN/results/{results_folder_name}/plots.npz', epoch=range(1, len(results['train_loss']) + 1), train_loss = results['train_loss'], val_loss = results['test_loss'] , X_val=X_val, y_noise_val=y_noise_val , X_tot = X_tot, y_true_tot = y_true_tot, sorted_X = sorted_X, sorted_KAN_preds = sorted_KAN_preds)
        
    return results


def main():
    # TODO: Implement cross-validation and grid search for hyperparameter tuning
    # TODO: Implement additional hyperparameter configurations
    default = "uniform_sin(x)_241114"
    configs = [{
        "folder_name": default,
        "layers" : [1, 3, 3, 1],
        "grid" : 3,
        "k" : 3,
        "lr": 0.01, 
        "steps": 800,
        "lamb": 0.0,
    }]
    for cfg in configs:
        print(f"Running model for {cfg['folder_name']}")
        basic_fit(cfg["folder_name"], cfg)

if __name__ == "__main__":
    main()