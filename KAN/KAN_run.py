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

def basic_fit(train_data: pd.DataFrame, val_data, test_data, total_data) -> dict:
    # Initialize model and create dataset

    ## Params ## TODO: Choose architecture + choose save folder name
    width = [1, 3, 3, 3, 1] #width for each layer
    grid = 3  # nr of spline grids
    k = 3 #order of spline
    seed = 0 #random seed
    folder_name = f'uniform_sin(x)_{width}_241114' #save folder name, one for each model run
    os.makedirs(f'./KAN/results/{folder_name}', exist_ok=True)

    kan_model = KAN(width=width, grid=grid, k=k, seed=seed)


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
    ## Params ## TODO: can change params, only steps for friday 14 nov
    dataset_input = "y_noise, y_true"
    opt = "Adam"
    steps = 800
    lr = 0.01
    lamb = 0.0
    

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

    # Set Seaborn theme
    sns.set_theme(style="whitegrid")

    # Create a figure with two subplots side by side
    #fig, ax = plt.subplots(1, 2, figsize=(16, 6))

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
    plt.savefig(f'./KAN/results/{folder_name}/plot.png', dpi=300)

    plt.clf()

    # --- Plot 2: Training and Validation Loss ---
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

    

    plt.savefig(f'./KAN/results/{folder_name}/loss.png', dpi=300)
    # Optional: Save the figure with high resolution 
    



    # Display the plots
    #plt.show()

    print(f"Elapsed Time: {elapsed_time:.3f} seconds")

    print(loss_df['Validation Loss'].iloc[-1])

    
    
    file_path = f'./KAN/results/{folder_name}/model_params.txt'

    with open(file_path, "w") as file:
        file.write(f"input to model: {dataset_input}\n")
        file.write(f"width: {width}\n")
        file.write(f"grid: {grid}\n")
        file.write(f"k: {k}\n")
        file.write(f"seed: {seed}\n")
        file.write(f"opt: {opt}\n")
        file.write(f"steps: {steps}\n")
        file.write(f"lr: {lr}\n")
        file.write(f"lamb: {lamb}\n")
        file.write(f"final validation loss: {loss_df['Validation Loss'].iloc[-1]}\n")
        file.write(f"final training loss: {loss_df['Train Loss'].iloc[-1]}\n")
            
    np.savez(f'./KAN/results/{folder_name}/plots.npz', epoch=range(1, len(results['train_loss']) + 1), train_loss = results['train_loss'], val_loss = results['test_loss'] , X_train = X_train, y_noise_train = y_noise_train, X_tot = X_tot, y_true_tot = y_true_tot, sorted_X = sorted_X, sorted_KAN_preds = sorted_KAN_preds)
        
    return results

#### Main():

### Import dataset
import_data_folder = "uniform_sin(x)_241114"  ## TODO: select dataset

train_data = read_data(f"./datasets/{import_data_folder}/train_data.csv")
val_data = read_data(f"./datasets/{import_data_folder}/validation_data.csv")
test_data = read_data(f"./datasets/{import_data_folder}/test_data.csv")
total_data = read_data(f"./datasets/{import_data_folder}/true_data.csv")

### Train model, prediction and plot
basic_fit(train_data=train_data, val_data=val_data, test_data=test_data, total_data=total_data)


### Plot from data
"""
data = np.load("./KAN/results/result3/plots.npz", allow_pickle=True)
print(data["X_train"].shape)
print(data["y_noise_train"].shape)
print(data["sorted_KAN_preds"].shape)
print(data["epoch"].shape)
print(data["train_loss"].shape)
print(data["val_loss"].shape)



loss_df = pd.DataFrame({
        'Epoch': data["epoch"],
        'Train Loss': data["train_loss"],
        'Validation Loss': data["val_loss"]})

loss_melted = loss_df.melt(id_vars='Epoch', var_name='Loss Type', value_name='Loss')

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(1, 2, figsize=(16, 6))

ax[0].plot(data["X_train"], data["y_noise_train"], marker='o', markersize=1, linestyle='None', label="train data")
ax[0].plot(data["X_tot"], data["y_true_tot"], linestyle='-',label='True function')

sorted_X = data["sorted_X"]
sorted_KAN_preds = data["sorted_KAN_preds"]


#print("sorted_KAN_preds",sorted_KAN_preds.shape)
ax[0].plot(sorted_X, sorted_KAN_preds, "--", label='KAN predictions')

ax[0].set_xlabel("Random X 1D samples")
ax[0].set_ylabel("Function")
ax[0].legend()



sns.lineplot(data=loss_melted, x='Epoch', y='Loss', hue='Loss Type',
                 ax=ax[1], marker='.')

# Set labels and title
ax[1].set_xlabel("Epoch", fontsize=12)
ax[1].set_ylabel("Loss", fontsize=12)
ax[1].set_title("Training and Validation Loss Over Epochs", fontsize=14, weight='bold')

# Customize legend
ax[1].legend(title='Loss Type', fontsize=10, title_fontsize=12)

# Adjust layout for better spacing
plt.tight_layout()

# Optional: Save the figure with high resolution
# plt.savefig('enhanced_plots.png', dpi=300)

# Display the plots
plt.show()
"""