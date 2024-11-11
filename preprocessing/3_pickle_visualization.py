import pickle
from matplotlib import pyplot as plt
import pandas as pd

# save data as pickle file
metrics_pkl = "/homes/chemie/sandner/Thesis/Final_data/pickle_files/metrics.pkl"
metrics_wo_protein_information_pkl = "/homes/chemie/sandner/Thesis/Final_data/pickle_files/metrics_wo_protein_information.pkl"
metrics_wo_descriptors_pkl = "/homes/chemie/sandner/Thesis/Final_data/pickle_files/metrics_wo_descriptors.pkl"
metrics_wo_binding_information_pkl = "/homes/chemie/sandner/Thesis/Final_data/pickle_files/metrics_wo_binding_information.pkl"
metrics_alphafold_pkl = "/homes/chemie/sandner/Thesis/Final_data/pickle_files/metrics_alphafold.pkl"

# normal model

metrics = pd.read_pickle(metrics_pkl)
metrics = pd.DataFrame.from_dict(metrics)
print(metrics)

# model without protein information
metrics_wo_protein = pd.read_pickle(metrics_wo_protein_information_pkl)
metrics_wo_protein = pd.DataFrame.from_dict(metrics_wo_protein)
print(metrics_wo_protein)

# model without descriptors
metrics_wo_descriptors = pd.read_pickle(metrics_wo_descriptors_pkl)
metrics_wo_descriptors = pd.DataFrame.from_dict(metrics_wo_descriptors)
print(metrics_wo_descriptors)

# model without binding information
metrics_wo_binding = pd.read_pickle(metrics_wo_binding_information_pkl)
metrics_wo_binding = pd.DataFrame.from_dict(metrics_wo_binding)
print(metrics_wo_binding)

# model with alphafold filter
metrics_alphafold = pd.read_pickle(metrics_alphafold_pkl)
metrics_alphafold = pd.DataFrame.from_dict(metrics_alphafold)

# visualization

# Plotting R2 train
plt.figure(figsize=(12, 6))
plt.plot(metrics["r2 train"], label="Normal Model")
plt.plot(metrics_wo_protein["r2 train"], label="Without Protein Info")
plt.plot(metrics_wo_descriptors["r2 train"], label="Without Descriptors")
plt.plot(metrics_wo_binding["r2 train"], label="Without Binding Info")
plt.title("Training R2 with different features")
plt.xlabel("Epoch")
plt.ylabel("R2 Train")
plt.ylim(0,1)
plt.legend()
plt.savefig("r2_train_comparison.png")

    # alphafold
plt.figure(figsize=(12, 6))
plt.plot(metrics["r2 train"], label="Normal Model")
plt.plot(metrics_alphafold["r2 train"], label="AlphaFold Filter")
plt.title("Training R2")
plt.xlabel("Epoch")
plt.ylabel("R2 Train")
plt.legend()
plt.savefig("r2_train_comparison_alphafold.png")

# Plotting Loss train
plt.figure(figsize=(12, 6))
plt.plot(metrics["loss train"], label="Normal Model")
plt.plot(metrics_wo_protein["loss train"], label="Without Protein Info")
plt.plot(metrics_wo_descriptors["loss train"], label="Without Descriptors")
plt.plot(metrics_wo_binding["loss train"], label="Without Binding Info")
plt.title("Training loss with different features")
plt.xlabel("Epoch")
plt.ylabel("Loss Train")
plt.legend()
plt.savefig("loss_train_comparison.png")

plt.figure(figsize=(12, 6))
plt.plot(metrics["loss train"], label="Normal Model")
plt.plot(metrics_alphafold["loss train"], label="AlphaFold Filter")
plt.title("Training R2")
plt.xlabel("Epoch")
plt.ylabel("R2 Train")
plt.legend()
plt.savefig("loss_train_comparison_alphafold.png")

fig, axs = plt.subplots(4,1, figsize=(8,10))

# Plot each dataframe on its subplot
axs[0].plot(metrics['r2 train'], marker='o', linestyle='-', color='b', label='train')
axs[0].plot(metrics['r2 val'], marker='*', linestyle='--', color='r', label='val')
axs[0].set_ylim([0,1])
axs[0].set_title('All features')
axs[0].legend()

axs[1].plot(metrics_wo_protein['r2 train'], marker='o', linestyle='-', color='b', label='train')
axs[1].plot(metrics_wo_protein['r2 val'], marker='*', linestyle='--', color='r', label='val')
axs[1].set_ylim([0,1])
axs[1].set_title('without protein information')
axs[1].legend()

axs[2].plot(metrics_wo_descriptors['r2 train'], marker='o', linestyle='-', color='b', label='train')
axs[2].plot(metrics_wo_descriptors['r2 val'], marker='*', linestyle='--', color='r', label='val')
axs[2].set_ylim([0,1])
axs[2].set_title('without descriptors')
axs[2].legend()

axs[3].plot(metrics_wo_binding['r2 train'], marker='o', linestyle='-', color='b', label='train')
axs[3].plot(metrics_wo_binding['r2 val'], marker='*', linestyle='--', color='r', label='val')
axs[3].set_ylim([0,1])
axs[3].set_title('without binding information')
axs[3].legend()

# Adjust layout and display the plot
plt.tight_layout()
plt.savefig("r2_over_epoch_different_features.png")
