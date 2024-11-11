import torch
import torch.nn as nn 
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from scipy import stats
import numpy as np
import os
from torcheval.metrics.functional import r2_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import pickle

# version for file naming
version = "custom_BatchNorm1D_reg"

# directories
df = "/fast/sandner/Thesis/Thesis/Final_data/final_df_fingerprints_alphafold_cluster.csv"
p2rank_dir = "/fast/sandner/output_p2rank/predict_list/"
final_df = pd.read_csv(df)

# device 
if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")
print("device used:", device)

# Plot selected descriptors over km value
descriptors = ['km_value', 'HallKierAlpha', 'Ipc', 'NumAromaticRings', 'NumRotatableBonds', 'NumHAcceptors']
for descriptor in descriptors:
    plt.hist(final_df[descriptor], bins=100)
    plt.title(f'{descriptor}')
    plt.xlabel(f'{descriptor} value')
    plt.ylabel('n')
    plt.savefig(f"descriptors_{descriptor.lower()}.png")
    plt.close()

# Check for NaN values in the entire dataframe
print(final_df.isna().sum())

def collate_fn(batch):
    # Take care of corrupt data
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Build custom dataset
class KmClass(Dataset):
    def __init__(self, df, split, p2rank_dir, amino_scaler=None, descriptor_scaler_robust=None, descriptor_scaler_minmax=None, km_scaler=None):
        self.dataframe = pd.read_csv(df)
        self.dataframe = self.dataframe[self.dataframe['set'] == split]
        self.dataframe = self.dataframe.drop('Ipc', axis=1)
        self.dataframe = self.dataframe[self.dataframe['below_threshold'] == 0] # drop datapoints with low alphafold confidence score
        self.dataframe = self.dataframe.loc[self.dataframe.km_value > 0]
        self.p2rank = p2rank_dir

        # Prefilter proteins for valid pocket files
        self.valid_data = self.dataframe[self.dataframe['uniprot_key'].apply(self.__has_valid_pocket_file__)]
        print(f"Total valid samples after filtering: {len(self.valid_data)}")

        # Use pre-fitted scalers if provided
        self.amino_scaler = amino_scaler
        self.descriptor_scaler_robust = descriptor_scaler_robust
        self.descriptor_scaler_minmax = descriptor_scaler_minmax
        self.km_scaler = km_scaler

        # amino acid identities
        aa_1LC = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]
        aa_encoded = torch.tensor([ord(aa) for aa in aa_1LC]).unsqueeze(dim=1).numpy()

        # Fit the scaler on amino acids if it's not provided
        if self.amino_scaler is None:
            self.amino_scaler = MinMaxScaler(feature_range=(0, 1))
            self.amino_scaler.fit(aa_encoded)

        # Fit the scalers on the descriptors if not provided
        if self.descriptor_scaler_robust is None:
            self.descriptor_scaler_robust = RobustScaler()
            self.descriptor_scaler_robust.fit(self.dataframe.iloc[:, 5:].values)  # Columns starting from 5 (after km_value, uniprot_key, sequence, set)

        if self.descriptor_scaler_minmax is None:
            self.descriptor_scaler_minmax = MinMaxScaler(feature_range=(0, 1))
            self.descriptor_scaler_minmax.fit(self.dataframe.iloc[:, 5:].values)  # Columns starting from 5

        # Fit the scaler on the km values if it's not provided
        if self.km_scaler is None:
            self.km_scaler = MinMaxScaler(feature_range=(0, 1))
            # new scaling: set the range for km values between 10^-7 and 10^-1 M
            km_values_capped = np.where(self.valid_data['km_value'] > 100, 100, np.where(self.valid_data['km_value'] < 0.0001, 0.0001, self.valid_data['km_value']))
            self.km_log_capped = np.log10(km_values_capped).reshape(-1, 1)
            self.km_scaler.fit(self.km_log_capped)

    def __has_valid_pocket_file__(self, protein_id):
        file_path = os.path.join(self.p2rank, f"{protein_id}.pdb_residues.csv")
        return os.path.exists(file_path)

    def __len__(self):
        return len(self.valid_data)
    
    def __get_pocket_values__(self, protein_id):
        # retrieve files from p2rank directory
        file = os.path.join(self.p2rank, f"{protein_id}.pdb_residues.csv")
        if os.path.exists(file):
            pocket_df = pd.read_csv(file)  # Reading the CSV file
            return pocket_df[' pocket'].tolist()   # Extracting the 'pocket' column and converting it to a list
        else:
            return None
    
    def __structure_data__(self, sequence, pocket_values, row):
        # Create a list for alternating sequence and pocket values
        combined = []
        amino_acid_values = [ord(aa) for aa in sequence]  # Convert amino acids to ASCII values

        # Scale the amino acid ASCII values
        amino_acid_values_scaled = self.amino_scaler.transform(np.array(amino_acid_values).reshape(-1, 1)).flatten()

        # Combine scaled amino acid values with pocket values
        for pocket in pocket_values:
            combined.append(amino_acid_values_scaled[len(combined) // 2])  # Get the scaled value from the list
            combined.append(pocket)

        # Pad the combined list with zeros to reach the desired length of 2048 for the aa sequence
        if len(combined) < 2048:
            combined.extend([0] * (2048 - len(combined)))  # Fill with zeros
        else:
            combined = combined[:2048]  # Truncate if it exceeds

        # Scale the descriptors first with RobustScaler, then with MinMaxScaler
        descriptors = row.iloc[5:].values.reshape(1, -1)  # for selected descriptors
        descriptors_scaled_robust = self.descriptor_scaler_robust.transform(descriptors)  # Apply RobustScaler
        descriptors_scaled = self.descriptor_scaler_minmax.transform(descriptors_scaled_robust)  # Apply MinMaxScaler

        combined.extend(descriptors_scaled.flatten())  # Combine scaled descriptors

        # Add fingerprint information
        fingerprints = row.iloc[-2048:].values.reshape(1, -1)

        combined.extend(fingerprints.flatten())

        return combined
    
    def __getitem__(self, idx):
        row = self.valid_data.iloc[idx]

        # Extract the data
        protein_id = row['uniprot_key']
        sequence = row['sequence']
        km_value = row['km_value']

        # Ensure km_value is not NaN
        if pd.isna(km_value):
            print(f"Skipping {protein_id}: km_value is NaN")
            return None 
        
        # Retrieve pocket values
        pocket_values = self.__get_pocket_values__(protein_id)

        # If pocket values are None, skip this protein (no binding site prediction available)
        if pocket_values is None or len(pocket_values) != len(sequence):
            return None

        # get rid of unwanted pocket values (higher than 1)
        for i, x in enumerate(pocket_values):
            if x != 1: pocket_values[i] = 0

        combined_data = self.__structure_data__(sequence, pocket_values, row)
        
        x = torch.tensor(combined_data, dtype=torch.float32)
        y = torch.tensor([km_value], dtype=torch.float32).log10()
        y = torch.tensor(self.km_scaler.transform(y.unsqueeze(1)),dtype=torch.float32).flatten()

        # Check for NaNs in `x`
        if torch.isnan(x).any() or torch.isnan(y).any():
            return None

        return x, y


# Load and prepare datasets for training, validation, and test sets
# Fit the scalers on the training data
train_data = KmClass('/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv', "train", p2rank_dir)
amino_scaler = train_data.amino_scaler
descriptor_scaler_robust = train_data.descriptor_scaler_robust
descriptor_scaler_minmax = train_data.descriptor_scaler_minmax
km_scaler = train_data.km_scaler

# Now load the validation and test datasets with the pre-fitted scalers
val_data = KmClass('/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv', "val", p2rank_dir, amino_scaler, descriptor_scaler_robust, descriptor_scaler_minmax, km_scaler)
test_data = KmClass('/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv', "test", p2rank_dir, amino_scaler, descriptor_scaler_robust, descriptor_scaler_minmax, km_scaler)

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=True, collate_fn=collate_fn, num_workers=4)


# Neural network class
class Network(nn.Module):
    def __init__(self, input_size=4292):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )  # Final output layer

    def forward(self, x):
        x = self.layers(x)  # Linear output
        return x

net = Network().to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.0001) 

# Loops
loss_values = []
r2_values = []
loss_values_val = []
r2_values_val = []

# Training loop
for epoch in range(20):
    net.train() 
    running_loss = 0.0
    running_r2 = 0.0
    for i, data in enumerate(train_loader):
        if data is None:
            continue

        inputs, labels = data

        # loads tensor into GPU
        inputs = inputs.to(device)                                              
        labels = labels.to(device)

        if torch.isnan(inputs).any() or torch.isnan(labels).any():
            print("Found NaN in inputs or labels!")
            continue

        optimizer.zero_grad()   # Zero the parameter gradients

        # Forward pass
        outputs = net(inputs)   # Get the network outputs

        loss = criterion(outputs, labels.view(-1, 1))
        loss.backward() # Backward pass (compute gradients)
        optimizer.step()  

        # Rename for statistics
        y_pred = outputs
        y_true = labels

        r2 = r2_score(y_pred, y_true)
        
        running_loss += loss.item()
        running_r2 += r2.item()
        if i % 10 == 0 :
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, R2: {r2}")

    # Loss calculation and storing for plotting
    average_loss = running_loss / len(train_loader)
    loss_values.append(average_loss)

    # R2 calculation and storing for plotting
    average_r2 = running_r2 / len(train_loader)
    r2_values.append(average_r2)

    # Validation loop
    net.eval()
    val_loss = 0.0
    val_r2 = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            if val_data is None:
                continue

            val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            val_outputs = net(val_inputs)

            val_loss += criterion(val_outputs, val_labels.view(-1, 1)).item()
            val_r2 += r2_score(val_outputs, val_labels).item()

    # Average validation loss and R2 score
    average_val_loss = val_loss / len(val_loader)
    loss_values_val.append(average_val_loss)

    average_val_r2 = val_r2 / len(val_loader)
    r2_values_val.append(average_val_r2)

    print(f"Epoch {epoch}, Training Loss: {average_loss}, Validation Loss: {average_val_loss}, Training R2: {average_r2}, Validation R2: {average_val_r2}")

# Test set
test_x, test_y = next(iter(test_loader))  

# loads tensor into GPU
test_x = test_x.to(device)
test_y = test_y.to(device)

outputs_test = net(test_x)
print(test_y.shape)

# Reshape the outputs and targets to ensure they match in size
y_test = outputs_test.view(-1)
y_true = test_y.view(-1)

# R2 score
r2_test = r2_score(y_test, y_true)
print(f"Test R2: {r2_test}")

# Mean Absolute Error
mae = nn.L1Loss()
mae_value = mae(y_test, y_true)
print(f"Test MAE: {mae_value}")

# Mean Squared Error
mse = nn.MSELoss()
mse_value = mse(y_test, y_true)
print(f"Test MSE: {mse_value}")

# Root Mean Squared Error
rmse_value = torch.sqrt(mse_value)
print(f"Test RMSE: {rmse_value}")


# save r2 and loss to pickle file
with open(f"/homes/chemie/sandner/Thesis/Final_data/pickle_files/metrics_{version}.pkl", "wb") as f:
    pickle.dump({"loss train": loss_values, "r2 train": r2_values, "loss val": loss_values_val, "r2 val": r2_values_val}, f)


# Plots

# Residual Analysis Plot
y_true = y_true.to("cpu")
y_test = y_test.to("cpu")
residuals = (y_true - y_test).detach().numpy()
plt.scatter(y_true.detach().numpy(), residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Values")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.savefig(f"residual_analysis_plot_{version}.png")
plt.close()

# Predicted vs. True Value Plot
plt.scatter(y_true.detach().numpy(), y_test.detach().numpy())
plt.plot([y_true.max(), y_true.min()], [y_true.max(), y_true.min()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values")
plt.savefig(f"predicted_vs_true_plot_{version}.png")
plt.close()

# Plot the loss over epoch
plt.plot(loss_values, label='Training Loss')
plt.plot(loss_values_val, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f"loss_over_epoch_{version}.png")
plt.close()

# Plot R2 over epoch
plt.plot(r2_values, label='Training R2')
plt.plot(r2_values_val, label='Validation R2')
plt.title('Training and Validation R2')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.legend()
plt.savefig(f"r2_over_epoch_{version}.png")
plt.close()