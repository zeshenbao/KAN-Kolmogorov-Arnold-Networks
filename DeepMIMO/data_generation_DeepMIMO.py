import DeepMIMOv3 as DeepMIMO
import matplotlib.pyplot as plt
import numpy as np

def create_dataset(scenario_name, 
                   n_bs_y, n_bs_x, n_ue_y, n_ue_x, 
                   n_paths, n_subcarriers):
    # Load the default params, set constant params
    params = DeepMIMO.default_params()
    params['dataset_folder'] = r'.\scenarios'
    
    # Set params
    # Which scenario
    params['scenario'] = scenario_name
    # For the antennas
    #params['bs_antenna']['shape'] = np.array([n_bs_y, n_bs_x])
    #params['ue_antenna']['shape'] = np.array([n_ue_y, n_ue_x])
    # For the path
    params['num_paths'] = n_paths
    params['OFDM']['subcarriers'] = n_subcarriers
    
    # Create dataset
    dataset = DeepMIMO.generate_data(params)
    
    return dataset

def generate_H(dataset):
    i = 0  # Only one base station in scenario I2
    j = 0  # Only the first user
    H = dataset[i]['user']['channel'][j]
    H = H[:, :, 0]  # Choose first subcarrier
    return H

def generate_H_abs(dataset):
    H = generate_H(dataset)
    H_abs = np.abs(H)

    return H_abs

def generate_noisy_H(H):
    noise = np.random.normal(0, 10**(-6), H.shape)
    H_noise = H + noise

    return H_noise


def plot_heatmat(H):
    plt.figure(figsize=(8, 6))
    plt.imshow(H.T, cmap='viridis', aspect='auto', origin='lower', interpolation='nearest')
    plt.colorbar(label='Channel Gain Magnitude')
    plt.title('Channel Gain Magnitude of (RX, TX) Antenna Pairs')
    plt.xlabel('RX antenna')
    plt.ylabel('TX antenna')
    plt.xticks(ticks=np.arange(H.shape[0]), labels=np.arange(1, H.shape[0] + 1))
    plt.yticks(ticks=np.arange(H.shape[1]), labels=np.arange(1, H.shape[1] + 1))
    plt.show()

def plot_environment(dataset):
    fig = plt.figure(figsize=(8, 6))
    ax = plt.axes(projection ="3d")

    bs_idx = 0
    loc_x = dataset[bs_idx]['user']['location'][:, 0]
    loc_y = dataset[bs_idx]['user']['location'][:, 1]
    loc_z = dataset[bs_idx]['user']['location'][:, 2]
    ax.scatter(loc_x, loc_y, loc_z, c='r', label='ue antennas')

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')

    plt.show()

def data_generation(scenario_name, 
                    n_bs_y, n_bs_x, n_ue_y, n_ue_x,
                    n_paths, n_subcarriers):
    dataset = create_dataset(scenario_name, n_bs_y, n_bs_x, n_ue_y, n_ue_x, n_paths, n_subcarriers)
    H_abs = generate_H_abs(dataset)
    # plot_environment(dataset)

    return H_abs

H_abs = data_generation('I2_28B',
                        8, 1, 16, 1,
                        5, 2)
print(H_abs.shape)
H_noise = generate_noisy_H(H_abs)
print(H_noise.shape)
plot_heatmat(H_noise)
