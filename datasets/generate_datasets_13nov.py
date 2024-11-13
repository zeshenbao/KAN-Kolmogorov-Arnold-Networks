import numpy as np
import pandas as pd
from noise import *
import random
import os 

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
# set noise distribution
#lower_bound, upper_bound = -2, 2
#noise = np.random.uniform(lower_bound, upper_bound, n)
# add noise to the y-values
#y_noise = y_true + noise



### Räkna SNR 

#mean(coloured_noise_f(xs))
#mean(y_true)
#SNR = mean/mean

y_noise = coloured_noise_f(xs) + y_true


#y_noise_tot = np.vstack(noise_list)
#print(y_noise.shape)
#print(y_noise_tot.shape)

# create dataset
#noise_list is list of 10 elements, each element (1000,) array
data = {'x': xs,'y_noise': y_noise, 'y_true': y_true}

df = pd.DataFrame(data)

### split data


# Convert DataFrame to CSV

### skapa folder ./datasets/pink_sin(x)
folder_name = "pink_sin_test"

os.makedirs(f'./datasets/{folder_name}', exist_ok=True)

df.to_csv(f'./datasets/{folder_name}/data.csv', index=False) 

file_path = f'./datasets/{folder_name}/params.txt'

with open(file_path, "w") as file:
    file.write(f"n_datapoints: {n_datapoints}\n")
    file.write(f"start_x, end_x: {start_x, end_x }\n")
    file.write("")
    file.write("")


#np.savez('./datasets/data_3sin(0.5x)_1.npz', xs=xs, y_true=y_true, y_noise=noise_list)

