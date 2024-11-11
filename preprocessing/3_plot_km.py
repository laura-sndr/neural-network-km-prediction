import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt


# old scaling:
'''
# Load the dataset
df = pd.read_csv('/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv')

df = df.loc[df['below_threshold'] == 0] # drop datapoints with low alphafold confidence score
df = df.loc[df.km_value < 100]
df = df.loc[df.km_value > 0]

# Separate data by sets
train_df = df[df['set'] == 'train']
val_df = df[df['set'] == 'val']

# log() km values
km_values_capped_train = np.where(train_df['km_value'] > 10, 10, train_df['km_value'])
km_values_capped_val = np.where(val_df['km_value'] > 10, 10, val_df['km_value'])
km_log_capped_train = np.log10(km_values_capped_train).reshape(-1, 1)
km_log_capped_val = np.log10(km_values_capped_val).reshape(-1, 1)

print(km_log_capped_train.max())
print(km_log_capped_val)

# Define and fit scaler
km_scaler = MinMaxScaler(feature_range=(0, 1))
km_scaler = km_scaler.fit(km_log_capped_train)
# Scale the km values
km_scaled_train = km_scaler.transform(km_log_capped_train)
km_scaled_val = km_scaler.transform(km_log_capped_val)

# Plot as histograms

plt.hist(km_scaled_train, bins=100)
plt.hist(km_scaled_val, bins=100)
plt.title('Km values training vs. validation')
plt.xlabel('Km values [mM]')
plt.xlim(-0.4, 0.95)
plt.ylim(0, 400)
plt.ylabel('n')

plt.savefig("/homes/chemie/sandner/Thesis/Final_data/Graphs/PCA/km_values_unscaled_zoom.png")

'''

# new scaling:

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv')

df = df.loc[df['below_threshold'] == 0] # drop datapoints with low alphafold confidence score
df = df.loc[df.km_value > 0]

# Separate data by sets
train_df = df[df['set'] == 'train']
val_df = df[df['set'] == 'val']

# log() km values
km_values_capped_train = np.where(train_df['km_value'] > 100, 100, np.where(train_df['km_value'] < 0.0001, 0.0001, train_df['km_value']))
km_values_capped_val = np.where(val_df['km_value'] > 100, 100, np.where(val_df['km_value'] < 0.0001, 0.0001, val_df['km_value']))
km_log_capped_train = np.log10(km_values_capped_train).reshape(-1, 1)
km_log_capped_val = np.log10(km_values_capped_val).reshape(-1, 1)

print(km_log_capped_train.max())
print(km_log_capped_val)

# Define and fit scaler
km_scaler = MinMaxScaler(feature_range=(0, 1))
km_scaler = km_scaler.fit(km_log_capped_train)
# Scale the km values
km_scaled_train = km_scaler.transform(km_log_capped_train)
km_scaled_val = km_scaler.transform(km_log_capped_val)

# Plot as histograms

plt.hist(km_scaled_train, bins=100)
plt.hist(km_scaled_val, bins=100)
plt.title('Km values training vs. validation')
plt.xlabel('Km values [mM]')
#plt.xlim(-0.4, 0.95)
#plt.ylim(0, 500)
plt.ylabel('n')

plt.savefig("/homes/chemie/sandner/Thesis/Final_data/Graphs/PCA/km_values_scaled.png")
