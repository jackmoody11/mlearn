# Load data
import pandas as pd
import os
datapath = os.getcwd() + "/../datasets/"
HOUSING_PATH = datapath + "housing.csv"
def load_housing_data(housing_path=HOUSING_PATH):
    return pd.read_csv(housing_path)

housing = load_housing_data()
print(housing.head())

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()

# Create a test set
import numpy as np

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]