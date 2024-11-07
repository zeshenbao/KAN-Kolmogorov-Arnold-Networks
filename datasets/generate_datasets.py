import numpy as np
import pandas as pd
from noise import *

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
y_true = np.sin(xs)
# set noise distribution
#lower_bound, upper_bound = -2, 2
#noise = np.random.uniform(lower_bound, upper_bound, n)
# add noise to the y-values
#y_noise = y_true + noise

y_noise = 0.2*coloured_noise_f(xs) + y_true


# create dataset
data = {
    'x': xs,
    'y_noise': y_noise,
    'y_true': y_true
}

df = pd.DataFrame(data)

# Convert DataFrame to CSV
df.to_csv('./datasets/pink_sin(x).csv', index=False)
