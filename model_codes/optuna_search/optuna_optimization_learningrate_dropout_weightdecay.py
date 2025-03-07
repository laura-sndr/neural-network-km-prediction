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
from sklearn.model_selection import train_test_split
import pickle

# version for file naming
folder = "xxx"
version = "xxx"

df = "/fast/sandner/Thesis/Thesis/Final_data/final_df_fingerprints_alphafold_cluster.csv"
p2rank_dir = "/fast/sandner/output_p2rank/predict_list/"
final_df = pd.read_csv(df)


if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")
print("device used:", device)

# Check for NaN values in the entire dataframe
print(final_df.isna().sum())

def collate_fn(batch):
    # Take care of corrupt data
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Build custom dataset
class KmClass(Dataset):
    def __init__(self, df, p2rank_dir, amino_scaler=None, descriptor_scaler_robust=None, descriptor_scaler_minmax=None, km_scaler=None):
        self.dataframe = pd.DataFrame(df)
        self.dataframe = self.dataframe.drop('Ipc', axis=1)
        self.dataframe = self.dataframe[self.dataframe['below_threshold'] == 0] # drop datapoints with low alphafold confidence score
        self.dataframe = self.dataframe.loc[self.dataframe.km_value > 0]
        # clip km values to biological range
        self.dataframe['km_value'] = self.dataframe['km_value'].clip(0.00001, 1000)
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
        descriptors = self.dataframe.iloc[:, 5:-2048].values  # Descriptor columns only
        if self.descriptor_scaler_robust is None:
            self.descriptor_scaler_robust = RobustScaler()
            descriptors_scaled_robust = self.descriptor_scaler_robust.fit_transform(descriptors)
        else:
            descriptors_scaled_robust = self.descriptor_scaler_robust.transform(descriptors)
        
        # Fit MinMaxScaler on the robust-scaled descriptors if not provided
        if self.descriptor_scaler_minmax is None:
            self.descriptor_scaler_minmax = MinMaxScaler(feature_range=(0, 1))
            self.descriptor_scaler_minmax.fit(descriptors_scaled_robust)

        # When fitting
        descriptors = self.dataframe.iloc[:, 5:-2048].values  # Descriptor columns only
        print("Number of descriptor features when fitting:", descriptors.shape[1])

        # Fit the scaler on the km values if it's not provided
        if self.km_scaler is None:
            self.km_scaler = MinMaxScaler(feature_range=(0, 1))
            self.km_log_capped = np.log10(self.valid_data['km_value']).values.reshape(-1, 1)
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
        
        # Apply both scalers to the descriptors
        descriptors = row.iloc[5:-2048].values.reshape(1, -1)
        descriptors_scaled_robust = self.descriptor_scaler_robust.transform(descriptors)
        descriptors_scaled = self.descriptor_scaler_minmax.transform(descriptors_scaled_robust)

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


train_set, temp_set = train_test_split(final_df, test_size=0.4, random_state=42)

# Step 2: Split temporary set into validation and test sets
val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)

# all_data = KmClass(final_df, p2rank_dir)
train_data = KmClass(train_set, p2rank_dir)

amino_scaler = train_data.amino_scaler
descriptor_scaler_robust = train_data.descriptor_scaler_robust
descriptor_scaler_minmax = train_data.descriptor_scaler_minmax
km_scaler = train_data.km_scaler

val_data = KmClass(val_set, p2rank_dir, amino_scaler, descriptor_scaler_robust, descriptor_scaler_minmax, km_scaler)
test_data = KmClass(test_set, p2rank_dir, amino_scaler, descriptor_scaler_robust, descriptor_scaler_minmax, km_scaler)

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=4)
test_loader = DataLoader(test_data, batch_size=len(test_set), shuffle=True, collate_fn=collate_fn, num_workers=4)


import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import Trainer


# Define the model class
class Network(nn.Module):
    def __init__(self, input_size, dropout1, dropout2, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Network, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout1),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout2),
            nn.Linear(hidden_dim2, hidden_dim3),
            nn.ReLU(),
            nn.Linear(hidden_dim3, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


# Define the objective function for Optuna
def objective(trial):
    # Hyperparameter sampling
    input_size = 4292
    hidden_dim1 = 2048
    hidden_dim2 = 1024
    hidden_dim3 = 512
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    dropout1 = trial.suggest_float("dropout1", 0.0, 0.6, step=0.1)
    dropout2 = trial.suggest_float("dropout2", 0.0, 0.6, step=0.1)
    weightdecay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)

    # Instantiate the model
    net = Network(input_size, dropout1, dropout2, hidden_dim1, hidden_dim2, hidden_dim3).to(device)

    # Define optimizer and loss function
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weightdecay)

    print(f"Trial {trial.number} with parameters: {trial.params}")

    # Loops
    loss_values = []
    r2_values = []
    loss_values_val = []
    r2_values_val = []
    epochs = 20

    # Training loop
    for epoch in range(epochs):
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

        # Report intermediate results to Optuna
        trial.report(average_val_loss, epoch)
        
         # Optuna pruning

        if epoch < 5 and average_val_r2 < -2:
            raise optuna.TrialPruned()
        elif epoch >= 5 and average_val_r2 < 0:
            raise optuna.TrialPruned()

        if trial.should_prune():
            raise optuna.TrialPruned()

        print(
            f"Epoch {epoch}, Training Loss: {average_loss}, Validation Loss: {average_val_loss}, "
            f"Training R2: {average_r2}, Validation R2: {average_val_r2}"
        )

    # Return the final validation loss for the trial
    return average_val_loss


# Run Optuna optimization
from optuna.pruners import MedianPruner

study = optuna.create_study(direction="minimize", pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3))
# Enqueue prior good values
study.enqueue_trial({"learning_rate": 0.000062,"dropout1": 0.4, "dropout2": 0.6, "weight_decay": 0.0005})
study.optimize(objective, n_trials=100)


print("Best hyperparameters:", study.best_params)
print("Best validation loss:", study.best_value)


import plotly.io as pio

# Plot the optimization history
opt_hist_fig = optuna.visualization.plot_optimization_history(study)
pio.write_image(opt_hist_fig, f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/optimization_history_{version}.png")

# Plot the parameter importances
param_importances_fig = optuna.visualization.plot_param_importances(study)
pio.write_image(param_importances_fig, f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/param_importances_{version}.png")

# Plot the parallel coordinate
parallel_coord_fig = optuna.visualization.plot_parallel_coordinate(study)
pio.write_image(parallel_coord_fig, f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/parallel_coordinate_{version}.png")

# Plot the slice plot
slice_plot_fig = optuna.visualization.plot_slice(study)
pio.write_image(slice_plot_fig, f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/plot_slice_{version}.png")

# Plot the contour plot for specified parameters
contour_plot_fig = optuna.visualization.plot_contour(study, params=["dropout1", "dropout2"])
pio.write_image(contour_plot_fig, f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/plot_contour_{version}.png")

