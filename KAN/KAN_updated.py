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


def basic_fit(data: pd.DataFrame, data2) -> dict:
    # Initialize model and create dataset

    kan_model = KAN(width=[1, 5, 5, 5, 1], grid=5, k=3, seed=0)

    #X = torch.tensor(data["x"].values).float().unsqueeze(1)
    #y_noise = torch.tensor(data["y_noise"].values).float().unsqueeze(1)
    #y_true = torch.tensor(data["y_true"].values).float().unsqueeze(1)

    #y_comb = torch.cat((X,y_noise), dim=1)

    #X2 = torch.tensor(data2["x"].values).float().unsqueeze(1)
    #y_noise2 = torch.tensor(data2["y_noise"].values).float().unsqueeze(1)
    #y_true2 = torch.tensor(data2["y_true"].values).float().unsqueeze(1)

    #y_comb2 = torch.cat((X2,y_noise2), dim=1)

    X = torch.tensor(data['xs']).float().unsqueeze(1)
    y_noise = torch.tensor(data['y_noise']).float()
    y_true = torch.tensor(data['y_true']).float().unsqueeze(1)

    X2 = torch.tensor(data2['xs']).float().unsqueeze(1)
    y_noise2 = torch.tensor(data2['y_noise']).float()
    y_true2 = torch.tensor(data2['y_true']).float().unsqueeze(1)


    print("X",X.shape)
    print("y_noise",y_noise.shape)
    #print("y_comb", y_comb.shape)

    dataset = create_dataset_from_data(y_noise.T, y_true, train_ratio=0.9)

    # Train model
    start = time.time()
    results = kan_model.fit(dataset, opt="LBFGS", steps=100, lr = 0.01, lamb=10)
    end = time.time()
    elapsed_time = end - start
    #print("4taetaet", y_noise2.shape)
    # Generate predictions
    KAN_preds = kan_model(dataset["test_input"]).detach()

    print("KAN_preds",KAN_preds.shape)
    # Debugging: Inspect results
    print("Keys in results:", results.keys())
    #print("Train Loss:", results.get('train_loss'))
    #print("Validation/Test Loss:", results.get('test_loss'))  

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
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Data and Predictions ---
    print("Test Input Shape:", dataset['test_input'].shape)
    print("Test Label Shape:", dataset['test_label'].shape)
    print("KAN Predictions Shape:", KAN_preds.shape)


    #sorted_indices_test = np.argsort(dataset['test_input'], axis=0)
    #sorted_indices_all = np.argsort(y_noise, axis=0)

    #plot_array_test = torch.gather(dataset['test_input'], dim=0, index=sorted_indices_test)
    #plot_array_pred = torch.gather(KAN_preds, dim=0, index=sorted_indices_test)

    #plot_array_x_tot = torch.gather(X, dim=0, index=sorted_indices_all)
    #plot_array_y_noise_tot = torch.gather(y_noise, dim=0, index=sorted_indices_all)
    #plot_array_y_true_tot = torch.gather(y_true, dim=0, index=sorted_indices_all)


    print("dataset['test_input']", dataset['test_input'].shape)
    print("y_noise", y_noise.shape)
    print("y_true", y_true.shape)
    print("X2", X2.shape)
    
    print("y_noise2.shape",y_noise2.shape)

    for i in range(10):
        y_noise2temp = y_noise2[i].unsqueeze(0).T
        if i == 0:
            ax[0].plot(X2, y_noise2temp, "o", markersize=1, linestyle='None', label="data")
        else:
            ax[0].plot(X2, y_noise2temp, "o", markersize=1, linestyle='None')


    ax[0].plot(X2, y_true2, "-",label='True function')
    ax[0].plot(X, y_true, "-",label='True trained function')
    ax[0].plot(X2, KAN_preds, "--",label='KAN predictions')

    #noise_x = np.linspace(-10, 10, 1000)
    #noise_combined = 0.0001*noise_x#np.sin(noise_x) + np.sin(0.2 * noise_x)
    

    ax[0].set_xlabel("Random X 1D samples")
    ax[0].set_ylabel("Function")
    ax[0].legend()



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

    print(f"Elapsed Time: {elapsed_time:.3f} seconds")

    return results




#data = read_data("./datasets/pink_sin(x)_stack.csv")
#data2 = read_data("./datasets/pink_sin(0.5x)_stack.csv")


data = np.load('./datasets/data_3sin(x)_1.npz', allow_pickle=True)
data2 = np.load('./datasets/data_sin(0.5x)_1.npz', allow_pickle=True)

basic_fit(data, data2)
