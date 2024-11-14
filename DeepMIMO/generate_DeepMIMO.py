import DeepMIMO
import matplotlib.pyplot as plt
import numpy as np

# Load the default parameters
parameters = DeepMIMO.default_params()

# Set scenario name
parameters['scenario'] = 'I2_28B'

# Set the main folder containing extracted scenarios
parameters['dataset_folder'] = r'.\scenarios'

# Set subcarriers
# To generate OFDM channels with 256 subcarriers, set
parameters['OFDM']['subcarriers'] = 2  # Changes the dimension of our H matrix

# Generate data
dataset = DeepMIMO.generate_data(parameters)

print(parameters)

# Channel matrix H
i = 0
j = 1
H = dataset[i]['user']['channel'][j]
print(H.shape)

# Assuming 'H' is your 3D channel matrix (num_tx_antennas, num_rx_antennas, num_subcarriers)

# Extract the magnitude of the channel gain for subcarrier 1 (index 0) across all TX and RX pairs
H_subcarrier_1_magnitude = np.abs(H[:, :, 0])  # Shape: (num_tx_antennas, num_rx_antennas)

# Create the heatmap
plt.figure(figsize=(8, 6))
plt.imshow(H_subcarrier_1_magnitude, cmap='viridis', aspect='auto', origin='lower', interpolation='nearest')
plt.colorbar(label='Channel Gain Magnitude')
plt.title('Channel Gain Magnitude for All TX and RX Pairs (Subcarrier 1)')
plt.xlabel('Receiver Index')
plt.ylabel('Transmitter Index')
plt.show()
