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

    ## Params
    width = [1, 3, 3, 1]
    grid = 5
    k = 3
    seed = 0

    kan_model = KAN(width=width, grid=grid, k=k, seed=seed)

    #X = torch.tensor(data["x"].values).float().unsqueeze(1)
    #y_noise = torch.tensor(data["y_noise"].values).float().unsqueeze(1)
    #y_true = torch.tensor(data["y_true"].values).float().unsqueeze(1)

    #y_comb = torch.cat((X,y_noise), dim=1)

    #X2 = torch.tensor(data2["x"].values).float().unsqueeze(1)
    #y_noise2 = torch.tensor(data2["y_noise"].values).float().unsqueeze(1)
    #y_true2 = torch.tensor(data2["y_true"].values).float().unsqueeze(1)

    #y_comb2 = torch.cat((X2,y_noise2), dim=1)

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

    

    #print("X",X.shape)
    print("y_noise_train",y_noise_train.shape)
    print("y_true_train",y_true_train.shape)


    print("y_noise_test",y_noise_test.shape)
    print("y_true_test",y_true_test.shape)

    #print("X",X.shape)
    #print("y_noise",y_noise.shape)
    #print("y_comb", y_comb.shape)

    dataset = {"train_input": y_noise_train, "train_label":y_true_train, "test_input":y_noise_test, "test_label":y_true_test}

    #dataset = create_dataset_from_data(X_tot, y_true_tot)

    #print("dataset1",dataset["train_input"].shape)
    #print("dataset1",dataset["train_label"].shape)

    #print("testtest",dataset["test_input"])
    #print("testtest2",dataset["train_input"])


    # Train model
    ## Params
    dataset_input = "y_noise, y_true"
    opt = "LBFGS"
    steps = 50
    lr = 0.01
    lamb = 0.0

    start = time.time()
    results = kan_model.fit(dataset, opt=opt, steps=steps, lr = lr , lamb = lamb)
    end = time.time()
    elapsed_time = end - start
    #print("4taetaet", y_noise2.shape)
    # Generate predictions
    KAN_preds = kan_model(dataset['test_input']).detach()

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


    #print("dataset['test_input']", dataset['test_input'].shape)
    #print("y_noise", y_noise.shape)
    #print("y_true", y_true.shape)
    #print("X2", X2.shape)
    
    #print("y_noise2.shape",y_noise2.shape)

    #for i in range(10):
    #    y_noise2temp = y_noise2[i].unsqueeze(0).T
    #    if i == 0:
    #        ax[0].plot(X2, y_noise2temp, "o", markersize=1, linestyle='None', label="data")
    #    else:
    #        ax[0].plot(X2, y_noise2temp, "o", markersize=1, linestyle='None')

    ax[0].plot(X_train, y_noise_train, "o", markersize=1, linestyle='None', label="train data")
    ax[0].plot(X_tot, y_true_tot, "-",label='True function')
    #ax[0].plot(X, y_true, "-",label='True trained function')

    sorted_X, indices = torch.sort(X_test, dim = 0)
    #print("indicies",indices.shape)

    sorted_KAN_preds = KAN_preds[indices][:,:,0]
    #print("sorted_KAN_preds",sorted_KAN_preds.shape)
    ax[0].plot(sorted_X, sorted_KAN_preds, "--", label='KAN predictions')

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


    folder_name = "result1"
    os.makedirs(f'./KAN/results/{folder_name}', exist_ok=True)
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

        

    return results



import_data_folder = "pink_sin_test2"

train_data = read_data(f"./datasets/{import_data_folder}/train_data.csv")
val_data = read_data(f"./datasets/{import_data_folder}/validation_data.csv")
test_data = read_data(f"./datasets/{import_data_folder}/test_data.csv")
total_data = read_data(f"./datasets/{import_data_folder}/true_data.csv")

#total_data = read_data(f"./datasets/pink_sin(0.5x).csv")
#total_data = read_data("./datasets/data_pink_noise_flat.csv")


#data = read_data("./datasets/pink_sin(x)_stack.csv")
#data2 = read_data("./datasets/pink_sin(0.5x)_stack.csv")


#data = np.load('./datasets/data_3sin(x)_1.npz', allow_pickle=True)
#data2 = np.load('./datasets/data_sin(0.5x)_1.npz', allow_pickle=True)

basic_fit(train_data=train_data, val_data=val_data, test_data=test_data, total_data=total_data)
