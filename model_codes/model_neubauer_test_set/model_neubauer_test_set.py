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
from torchmetrics import PearsonCorrCoef
import pickle

import torch
torch.multiprocessing.set_sharing_strategy('file_system')

# version for file naming
folder = "xxx"
version = "xxx"

df_train = "/fast/sandner/Thesis/Thesis/Final_data/final_df_fingerprints_alphafold.csv"
p2rank_dir_train = "/fast/sandner/output_p2rank/predict_list/"
final_df_train = pd.read_csv(df_train)

df_test = "/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes_descriptors_fingerprints.csv"
p2rank_dir_test = "/fast/sandner/Thesis/Thesis/Final_data/Neubauer/p2rank/"
final_df_test = pd.read_csv(df_test)

print(final_df_test)


if torch.cuda.is_available():
    device = torch.device("cuda")
else: 
    device = torch.device("cpu")
print("device used:", device)

# Check for NaN values in the entire dataframe
print(final_df_train.isna().sum())

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
        descriptors = self.dataframe.iloc[:, 4:-2048].values  # Descriptor columns only
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
        descriptors = self.dataframe.iloc[:, 4:-2048].values  # Descriptor columns only
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
        descriptors = row.iloc[4:-2048].values.reshape(1, -1)
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
            # pocket_values[i] = 0

        combined_data = self.__structure_data__(sequence, pocket_values, row)
        
        x = torch.tensor(combined_data, dtype=torch.float32)
        y = torch.tensor([km_value], dtype=torch.float32).log10()
        y = torch.tensor(self.km_scaler.transform(y.unsqueeze(1)),dtype=torch.float32).flatten()

        # Check for NaNs in `x`
        if torch.isnan(x).any() or torch.isnan(y).any():
            return None

        return x, y


# Split  set into train and validation sets
train_set, val_set = train_test_split(final_df_train, test_size=0.05, random_state=42)

# all_data = KmClass(final_df, p2rank_dir)
train_data = KmClass(train_set, p2rank_dir_train)

amino_scaler = train_data.amino_scaler
descriptor_scaler_robust = train_data.descriptor_scaler_robust
descriptor_scaler_minmax = train_data.descriptor_scaler_minmax
km_scaler = train_data.km_scaler

# Extract min and max values from km_scaler
km_min = km_scaler.data_min_[0]
km_max = km_scaler.data_max_[0]

val_data = KmClass(val_set, p2rank_dir_train, amino_scaler, descriptor_scaler_robust, descriptor_scaler_minmax, km_scaler)
test_data = KmClass(final_df_test, p2rank_dir_test, amino_scaler, descriptor_scaler_robust, descriptor_scaler_minmax, km_scaler)

# DataLoader
train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=16)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=16)
test_loader = DataLoader(test_data, batch_size=len(final_df_test), shuffle=False, collate_fn=collate_fn, num_workers=16)


# MODEL 

hyperparameters = {
    "hidden_dim1": 2048,
    "hidden_dim2": 1024,
    "hidden_dim3": 512,
    "learning_rate": 0.000062,
    "weight_decay": 0.00001,
    "dropout1": 0.4,
    "dropout2": 0.6
}

# Neural network class
class Network(nn.Module):
    def __init__(self, hidden_dim1, hidden_dim2, hidden_dim3, dropout1, dropout2, input_size=4292):
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
        )  # Final output layer

    def forward(self, x):
        x = self.layers(x)  # Linear output
        return x

net = Network(hidden_dim1=hyperparameters["hidden_dim1"], hidden_dim2=hyperparameters["hidden_dim2"], hidden_dim3=hyperparameters["hidden_dim3"], dropout1=hyperparameters["dropout1"], dropout2=hyperparameters["dropout2"]).to(device)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=hyperparameters["learning_rate"], weight_decay=hyperparameters["weight_decay"]) 

# Saving best version of the model (if generated r2 for validation is better than previously)
# Define the file path for the saved model
best_model_path = "/fast/sandner/Thesis/Thesis/Final_data/weights_biases/random_split.pth"
best_model_path_full = "/fast/sandner/Thesis/Thesis/Final_data/weights_biases/random_split_full.pth"

# Check if the best model file exists and load its r2 score
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path)
    best_val_r2 = checkpoint["best_r2"]  # The best R² stored in the checkpoint
    print(f"Loaded best R² from previous model: {best_val_r2:.4f}")
else:
    best_val_r2 = float('-inf')  # If no model exists, start with a very low value
    print("No previous model found, starting from scratch.")

# saving the best version of the model for each run separately 
best_r2 = float('-inf')  # Initialize the best R² 
best_weights = None  # To store the best model weights

# Loops
loss_values = []
r2_values = []
pearson_values = []
loss_values_val = []
r2_values_val = []
pearson_values_val = []

y_true_train = []
y_pred_train = []
y_true_val = []
y_pred_val = []

epochs = 20

# Training loop
for epoch in range(epochs):
    net.train() 
    running_loss = 0.0
    running_r2 = 0.0
    running_pearson = 0.0
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

        y_true_train.append(y_true.to("cpu"))
        y_pred_train.append(y_pred.to("cpu"))

        r2 = r2_score(y_pred, y_true)

        pcc = PearsonCorrCoef().to(device)
        pearson = pcc(y_pred, y_true)
        
        running_loss += loss.item()
        running_r2 += r2.item()
        running_pearson += pearson.item()

        if i % 10 == 0 :
            print(f"Epoch {epoch}, Batch {i}, Loss: {loss.item()}, R2: {r2}, Pearson: {pearson}")

    # Loss calculation and storing for plotting
    average_loss = running_loss / len(train_loader)
    loss_values.append(average_loss)

    # R2 calculation and storing for plotting
    average_r2 = running_r2 / len(train_loader)
    r2_values.append(average_r2)

    # Pearson calculation and storing for plotting
    average_pearson = running_pearson / len(train_loader)
    pearson_values.append(average_pearson)

    # Validation loop
    net.eval()
    val_loss = 0.0
    val_r2 = 0.0
    val_pearson = 0.0
    with torch.no_grad():
        for val_data in val_loader:
            if val_data is None:
                continue

            val_inputs, val_labels = val_data
            val_inputs = val_inputs.to(device)
            val_labels = val_labels.to(device)

            val_outputs = net(val_inputs)

            y_pred_val.append(val_outputs.to("cpu"))
            y_true_val.append(val_labels.to("cpu"))

            val_loss += criterion(val_outputs, val_labels.view(-1, 1)).item()
            val_r2 += r2_score(val_outputs, val_labels).item()
            val_pearson += pcc(val_outputs, val_labels).item()

    # Average validation loss and R2 score
    average_val_loss = val_loss / len(val_loader)
    loss_values_val.append(average_val_loss)

    average_val_r2 = val_r2 / len(val_loader)
    r2_values_val.append(average_val_r2)

    average_val_pearson = val_pearson / len(val_loader)
    pearson_values_val.append(average_val_pearson)

    # Save the model if the current validation R² is better than the best one saved
    if average_val_r2 > best_val_r2:
        best_val_r2 = average_val_r2  # Update the best R² value
        torch.save({
            "model_state_dict": net.state_dict(),
            "best_r2": best_val_r2,  # Save the best R² value alongside the model
            "hyperparameters": hyperparameters  # Save the hyperparameters too
        }, best_model_path)  # Save the model's state_dict and the best R²
        print(f"New best validation R²: {best_val_r2:.4f}. Model weights saved!")
        torch.save(net, best_model_path_full)

    if average_val_r2 > best_r2:
        best_r2 = average_val_r2
        best_weights = net.state_dict()  # Save the current model weights
        print(f"New best model - Saving weights for epoch {epoch+1}")

    print(f"Epoch {epoch}, Training Loss: {average_loss}, Validation Loss: {average_val_loss}, Training R2: {average_r2}, Validation R2: {average_val_r2}, Training Pearson: {average_pearson}, Validation Pearson: {average_val_pearson}")


# After training, load the best weights for testing
net.load_state_dict(best_weights)
net.eval()  # Set the model to evaluation mode for testing

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

# Pearson correlation coefficient
from torchmetrics import PearsonCorrCoef

pcc = PearsonCorrCoef().to(device)
pcc_value = pcc(y_test, y_true)
print(f"Test Pearson's R: {pcc_value}")

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

r2_test = r2_test.to("cpu")
pcc_value = pcc_value.to("cpu")
mae_value = mae_value.to("cpu")
mse_value = mse_value.to("cpu")
rmse_value = rmse_value.to("cpu")

y_true = y_true.to("cpu")
y_test = y_test.to("cpu")
test_x = test_x.to("cpu")

# Check test values for outliers
# Convert them to numpy arrays if they are tensors
test_x_np = test_x.detach().numpy() if hasattr(test_x, 'numpy') else test_x
y_test_np = y_test.detach().numpy() if hasattr(y_test, 'numpy') else y_test
y_true_np = y_true.detach().numpy() if hasattr(y_true, 'numpy') else y_true

# save r2 and loss to pickle file
with open(f"/fast/sandner/Thesis/Thesis/Final_data/pickle_files/{folder}/metrics_{version}.pkl", "wb") as f:
    pickle.dump({"loss train": loss_values, "r2 train": r2_values, "loss val": loss_values_val, "r2 val": r2_values_val}, f)

# save test scores to pickle file
with open(f"/fast/sandner/Thesis/Thesis/Final_data/pickle_files/{folder}/metrics_{version}_test.pkl", "wb") as f:
    pickle.dump({"test R2": r2_test, "test Pearson's R": pcc_value, "test MAE": mae_value, "test MSE": mse_value, "Test RMSE": rmse_value}, f)


# Saving predicted Km values scaled and original
y_scaled_pred = y_test.detach().numpy()

# Step 1: Reverse Min-Max scaling
y_log_pred = km_scaler.inverse_transform(y_scaled_pred.reshape(-1, 1))

# Step 2: Reverse log10 transformation
y_original_pred = np.power(10, y_log_pred)

# Save predictions for the fold
test_predictions = pd.DataFrame({
    'Predicted Value (Scaled)': y_scaled_pred.flatten(),
    'Predicted Value (Original)': y_original_pred.flatten(),
    'km_min': km_min,
    'km_max': km_max
})

test_predictions.to_csv(f"/fast/sandner/Thesis/Thesis/Final_data/pickle_files/{folder}/predictions_{version}_test.tsv", sep='\t', index=False)


# Plots

# Residual Analysis Plot
residuals = (y_true - y_test).detach().numpy()
plt.scatter(y_true.detach().numpy(), residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("True Values")
plt.ylabel("Residuals")
plt.title("Residuals Plot")
plt.savefig(f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/residual_analysis_plot_{version}.png")
plt.close()

# Predicted vs. True Value Plot
plt.scatter(y_true.detach().numpy(), y_test.detach().numpy())
plt.plot([y_true.max(), y_true.min()], [y_true.max(), y_true.min()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values")
plt.savefig(f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/predicted_vs_true_plot_{version}.png")
plt.close()

# Predicted vs. True Value Plot Training

y_true_train = torch.cat(y_true_train)
y_pred_train = torch.cat(y_pred_train)

plt.scatter(y_true_train.detach().numpy(), y_pred_train.detach().numpy())
plt.plot([y_true_train.max(), y_true_train.min()], [y_true_train.max(), y_true_train.min()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values - Training")
plt.savefig(f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/predicted_vs_true_plot_train_{version}.png")
plt.close()

# Predicted vs. True Value Plot Validation

y_true_val = torch.cat(y_true_val)
y_pred_val = torch.cat(y_pred_val)

plt.scatter(y_true_val.detach().numpy(), y_pred_val.detach().numpy())
plt.plot([y_true_val.max(), y_true_val.min()], [y_true_val.max(), y_true_val.min()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Predicted vs True Values - Validation")
plt.savefig(f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/predicted_vs_true_plot_val_{version}.png")
plt.close()

# Plot the loss over epoch
plt.plot(loss_values, label='Training Loss')
plt.plot(loss_values_val, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.ylim(bottom=0)
plt.legend()
plt.savefig(f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/loss_over_epoch_{version}.png")
plt.close()

# Plot R2 over epoch
plt.plot(r2_values, label='Training R2')
plt.plot(r2_values_val, label='Validation R2')
plt.title('Training and Validation R2')
plt.xlabel('Epochs')
plt.ylabel('R2')
plt.ylim(top=1)
plt.legend()
plt.savefig(f"/fast/sandner/Thesis/Thesis/Final_data/Graphs/model_output/{folder}/r2_over_epoch_{version}.png")
plt.close()


# save a file to keep track of versions:

import csv

# Define the path to the CSV file for logging
log_file = "/fast/sandner/Thesis/Thesis/Final_data/tracking/experiments_log.tsv"

# Function to log results into a CSV file
def log_experiment(version, hyperparameters, epochs, train_r2, val_r2, test_r2):
    # Ensure the hyperparameters are saved in a readable format
    hidden_dims = [hyperparameters["hidden_dim1"], hyperparameters["hidden_dim2"]]
    dropout_values = [hyperparameters["dropout1"], hyperparameters["dropout2"]]
    weight_decay = hyperparameters["weight_decay"]
    learning_rate = hyperparameters["learning_rate"]

    # Header for the CSV file
    header = [
        "Version Name", 
        "Learning Rate", 
        "Hidden Dimensions", 
        "Dropout Values", 
        "Weight Decay", 
        "Epochs", 
        "Train R2", 
        "Validation R2", 
        "Test R2"
    ]

    # Check if the file exists
    file_exists = os.path.isfile(log_file)

    # Open the CSV in append mode
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Write the header if the file is new
        if not file_exists:
            writer.writerow(header)
        
        # Log the current experiment details
        writer.writerow([
            version,
            learning_rate,
            hidden_dims,
            dropout_values,
            weight_decay,
            epochs,
            train_r2,
            val_r2,
            test_r2
        ])

# usage after training is complete
log_experiment(
    version=version,
    hyperparameters=hyperparameters,
    epochs=epochs,
    train_r2=r2_values[-1],   # Last epoch's training R²
    val_r2=r2_values_val[-1], # Last epoch's validation R²
    test_r2=r2_test.item()           # Test set R²
)
