import numpy as np
import pandas as pd
from noise import *
import random

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
n = 1000

# set startpoint and endpoint
start_x, end_x = -10, 10

# get x values
xs = generate_x_values(start_x, end_x, n)

# get y given a function
y_true = 3*np.sin(0.5*xs)
# set noise distribution
#lower_bound, upper_bound = -2, 2
#noise = np.random.uniform(lower_bound, upper_bound, n)
# add noise to the y-values
#y_noise = y_true + noise

noise_list = []

for i in range(10):
    random.seed(i)
    y_noise = coloured_noise_f(xs) + y_true
    noise_list.append(y_noise)

#y_noise_tot = np.vstack(noise_list)


#print(y_noise.shape)
#print(y_noise_tot.shape)

# create dataset
#noise_list is list of 10 elements, each element (1000,) array
#data = {'x': xs,'y_noise': noise_list, 'y_true': y_true}

#df = pd.DataFrame(data)

# Convert DataFrame to CSV
#df.to_csv('./datasets/pink_sin(x)_stack.csv', index=False)

np.savez('./datasets/data_3sin(0.5x)_1.npz', xs=xs, y_true=y_true, y_noise=noise_list)
