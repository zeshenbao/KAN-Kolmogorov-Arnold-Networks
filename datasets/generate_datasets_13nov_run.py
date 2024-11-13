import numpy as np
import pandas as pd
from noise import *
import random
import os 
from sklearn.model_selection import train_test_split  # Import train_test_split

# function 1
def f_1(x):
    return x**2+x

# add function 2 here
def f_2(x):
    return np.sin(x)

def coloured_noise_f(x):
    return pink_noise(x)

def generate_x_values(start_x, end_x, n):
    # create n x values
    xs = np.linspace(start_x, end_x, n)
    return xs

def generate_y_values(xs, func):
    return func(xs)


def calculate_snr(y_true, y_noise): ### write own code later
    # Calculate signal power
    signal_power = np.mean(y_true ** 2)
    # Calculate noise power
    noise_power = np.mean((y_noise - y_true) ** 2)
    # Calculate SNR
    snr = signal_power / noise_power
    # Convert to dB
    snr_db = 10 * np.log10(snr)
    return snr, snr_db

# set amount of datapoints
n_datapoints = 1000 ## TODO: set datapoints

# set startpoint and endpoint
start_x, end_x = -10, 10

# get x values
xs = generate_x_values(start_x, end_x, n_datapoints)

# get y given a function
y_true = np.sin(xs) ## TODO: choose function
func_name = "sin(x)" ## TODO: write function name for saving params for dataset
# set noise distribution

### TODO: uncomment to generate gaussian noise (Hampus previous code part)
#lower_bound, upper_bound = -2, 2
#noise_gaussian = np.random.uniform(lower_bound, upper_bound, n)
# add noise to the y-values

### Räkna SNR 
y_noise = 0.2*coloured_noise_f(xs) + y_true  ### TODO: change first term to noise_gaussian if you want gaussian noise
noise_name = "pink noise" ## TODO: write noise name for saving params for dataset
snr, snr_db = calculate_snr(y_true, y_noise)

# create dataset
#noise_list is list of 10 elements, each element (1000,) array
data = {'x': xs,'y_noise': y_noise, 'y_true': y_true}

df = pd.DataFrame(data)


### split data

# Split data (70% training, 20% validation, 10% testing)
train_df, test_val_df = train_test_split(df, test_size=0.3, random_state=42) 
validation_df, test_df = train_test_split(test_val_df, test_size=1/3, random_state=42)


# Convert DataFrame to CSV
print(train_df.head())
print("------------")
print(test_df.head())
print("------------")
print(validation_df.head())


# Create folder
folder_name = "pink_sin_test5" #folder name for saving dataset  ## TODO: set folder name
os.makedirs(f'./datasets/{folder_name}', exist_ok=True)

# Save training and test sets to CSV files
df.to_csv(f'./datasets/{folder_name}/true_data.csv', index=False)
train_df.to_csv(f'./datasets/{folder_name}/train_data.csv', index=False)
validation_df.to_csv(f'./datasets/{folder_name}/validation_data.csv', index=False)
test_df.to_csv(f'./datasets/{folder_name}/test_data.csv', index=False)

# Save parameters to a text file
file_path = f'./datasets/{folder_name}/params.txt'

with open(file_path, "w") as file:
    file.write(f"n_datapoints: {n_datapoints}\n")
    file.write(f"start_x, end_x: {start_x, end_x }\n")
    file.write(f"snr, snr_db: {snr, snr_db}\n")
    file.write(f"function name: {func_name}\n")
    file.write(f"noise:{noise_name}")

