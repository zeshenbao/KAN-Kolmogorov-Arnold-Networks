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

# set amount of datapoints
n_datapoints = 1000

# set startpoint and endpoint
start_x, end_x = -10, 10

# get x values
xs = generate_x_values(start_x, end_x, n_datapoints)

# get y given a function
y_true = np.sin(xs)

# add noise to the y-values
y_noise = coloured_noise_f(xs) + y_true

# create dataset
data = {'x': xs,'y_noise': y_noise, 'y_true': y_true}
df = pd.DataFrame(data)

### Split data into training and test sets
from sklearn.model_selection import train_test_split

# Split data (80% training, 20% testing)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

print(train_df.head())
print("------------")
print(test_df.head())

exit()

# Create folder
folder_name = "pink_sin_test"
os.makedirs(f'./datasets/{folder_name}', exist_ok=True)

# Save training and test sets to CSV files
train_df.to_csv(f'./datasets/{folder_name}/train_data.csv', index=False)
test_df.to_csv(f'./datasets/{folder_name}/test_data.csv', index=False)

# Save parameters to a text file
file_path = f'./datasets/{folder_name}/params.txt'
with open(file_path, "w") as file:
    file.write(f"n_datapoints: {n_datapoints}\n")
    file.write(f"start_x, end_x: {start_x, end_x}\n")
