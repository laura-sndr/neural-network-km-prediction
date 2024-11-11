import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataframe
df = pd.read_csv('/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv')
df = df.drop('Ipc', axis=1)
df = df[df['below_threshold'] == 0] # drop datapoints with low alphafold confidence score

# Scale the amino acid sequences
aa_1LC = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
aa_encoded = np.array([ord(aa) for aa in aa_1LC]).reshape(-1, 1)
amino_scaler = MinMaxScaler(feature_range=(0, 1))
amino_scaler.fit(aa_encoded)

# Apply the scaler to the sequences
sequences_scaled = []
for sequence in df['sequence']:
    sequence_scaled = amino_scaler.transform(np.array([ord(aa) for aa in sequence]).reshape(-1, 1)).flatten()
    sequences_scaled.append(sequence_scaled)

# Scale the descriptors
descriptor_scaler_robust = RobustScaler()
descriptor_scaler_minmax = MinMaxScaler(feature_range=(0, 1))
descriptors = df.iloc[:, 5:].values
descriptor_scaler_robust.fit(descriptors)
descriptor_scaler_minmax.fit(descriptor_scaler_robust.transform(descriptors))

# Apply the scalers to the descriptors
descriptors_scaled = descriptor_scaler_minmax.transform(descriptor_scaler_robust.transform(descriptors))

# Divide the dataframe into train and validation sets
train_df = df[df['set'] == 'train']
val_df = df[df['set'] == 'val']

# Define the maximum length of the sequences
max_length = 1024

# Pad the sequences for the training set
train_padded_sequences = []
for seq in train_df['sequence']:
    padded_seq = np.pad(np.array([ord(aa) for aa in seq]), (0, max_length - len(seq)), mode='constant')
    train_padded_sequences.append(padded_seq)
train_padded_sequences = np.array(train_padded_sequences)

# Pad the sequences for the validation set
val_padded_sequences = []
for seq in val_df['sequence']:
    padded_seq = np.pad(np.array([ord(aa) for aa in seq]), (0, max_length - len(seq)), mode='constant')
    val_padded_sequences.append(padded_seq)
val_padded_sequences = np.array(val_padded_sequences)

# Fit the amino acid scaler on the training data
amino_scaler.fit(train_padded_sequences)

# Apply the scaler to both the training and validation data
train_sequences_scaled = amino_scaler.transform(train_padded_sequences)
val_sequences_scaled = amino_scaler.transform(val_padded_sequences)

# Fit the descriptor scalers on the training data
descriptor_scaler_robust.fit(train_df.iloc[:, 5:].values)
descriptor_scaler_minmax.fit(descriptor_scaler_robust.transform(train_df.iloc[:, 5:].values))

# Apply the descriptor scalers to both the training and validation data
train_descriptors_scaled = descriptor_scaler_minmax.transform(descriptor_scaler_robust.transform(train_df.iloc[:, 5:].values))
val_descriptors_scaled = descriptor_scaler_minmax.transform(descriptor_scaler_robust.transform(val_df.iloc[:, 5:].values))

# concatentate for fitting
all_sequences_scaled = np.concatenate((train_sequences_scaled, val_sequences_scaled))
print(len(train_sequences_scaled))
print(len(val_sequences_scaled))
print(len(all_sequences_scaled))

all_descriptors_scaled = np.concatenate((train_descriptors_scaled, val_descriptors_scaled))

# Increase PCA components to 10 for both sequences and descriptors
pca_sequences = PCA(n_components=10).fit(all_sequences_scaled)
train_sequences_pca = pca_sequences.transform(train_sequences_scaled)
val_sequences_pca = pca_sequences.transform(val_sequences_scaled)

pca_descriptors = PCA(n_components=10).fit(all_descriptors_scaled)
train_descriptors_pca = pca_descriptors.transform(train_descriptors_scaled)
val_descriptors_pca = pca_descriptors.transform(val_descriptors_scaled)

# Explained variance ratio for the first 10 components
print("Explained variance ratio for sequence PCA:", pca_sequences.explained_variance_ratio_)
print("Explained variance ratio for descriptor PCA:", pca_descriptors.explained_variance_ratio_)

# Calculate cumulative explained variance for the first 10 components
cumulative_variance_sequences = np.cumsum(pca_sequences.explained_variance_ratio_)
cumulative_variance_descriptors = np.cumsum(pca_descriptors.explained_variance_ratio_)

# Plot the cumulative explained variance
plt.figure(figsize=(14, 7))

plt.subplot(1, 2, 1)
plt.bar(range(1, 11), pca_sequences.explained_variance_ratio_[:10], alpha=0.7, color='blue')
plt.plot(range(1, 11), cumulative_variance_sequences[:10], marker='o', color='red')
plt.title('Explained Variance by Sequence PCA Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.bar(range(1, 11), pca_descriptors.explained_variance_ratio_[:10], alpha=0.7, color='green')
plt.plot(range(1, 11), cumulative_variance_descriptors[:10], marker='o', color='red')
plt.title('Explained Variance by Descriptor PCA Components')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig("/homes/chemie/sandner/Thesis/Final_data/Graphs/PCA/explained_variance_sequences_descriptors_n10.png")

# Plot the first two principal components
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(train_sequences_pca[:, 0], train_sequences_pca[:, 1], label='Train')
plt.scatter(val_sequences_pca[:, 0], val_sequences_pca[:, 1], label='Validation')
plt.title('Sequence PCA (First 2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(train_descriptors_pca[:, 0], train_descriptors_pca[:, 1], label='Train')
plt.scatter(val_descriptors_pca[:, 0], val_descriptors_pca[:, 1], label='Validation')
plt.title('Descriptor PCA (First 2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.tight_layout()
plt.savefig("/homes/chemie/sandner/Thesis/Final_data/Graphs/PCA/first_two_components_sequences_descriptors_n10.png")


from sklearn.manifold import TSNE

# Apply t-SNE on the scaled training and validation sequences
tsne = TSNE(n_components=2, random_state=42)
all_sequences_tsne = tsne.fit_transform(all_sequences_scaled)

# Plot the t-SNE results for the sequences
plt.figure(figsize=(10, 8))
plt.scatter(all_sequences_scaled[0:19671, 0], all_sequences_scaled[0:19671, 1], label='Train', alpha=0.7, s=15, color='blue')
plt.scatter(all_sequences_scaled[19671:, 0], all_sequences_scaled[19671:, 1], label='Validation', alpha=0.7, s=15, color='orange')
plt.title('t-SNE of Amino Acid Sequences')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend()
plt.savefig("/homes/chemie/sandner/Thesis/Final_data/Graphs/tSNE/tsne_sequences.png")