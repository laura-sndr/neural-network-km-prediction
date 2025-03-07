import os
import glob
from Bio import PDB
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

# Set the directories containing the PDB files
pdb_dir = '/fast/sandner/Thesis/Thesis/Final_data/Neubauer/pdbs/pdbs'

# Set the threshold for pLDDT score
threshold = 70

# Initialize a dictionary to store the results
results = {}

# Loop through all PDB files in both directories
for pdb_dir in [pdb_dir]:
    for file in glob.glob(os.path.join(pdb_dir, '*.pdb')):
        # Read the PDB file
        parser = PDB.PDBParser()
        structure = parser.get_structure(os.path.basename(file), file)

        # Extract the pLDDT scores for each atom
        plddt_scores = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_bfactor() is not None:
                            plddt_scores.append(atom.get_bfactor())

        # Calculate the average pLDDT score
        avg_plddt = sum(plddt_scores) / len(plddt_scores)

        # Calculate the percentage of atoms with pLDDT score above the threshold
        above_threshold = sum(1 for score in plddt_scores if score > threshold) / len(plddt_scores) * 100

        # Store the results in the dictionary
        results[os.path.basename(file)] = {'avg_plddt': avg_plddt, 'above_threshold': above_threshold, 'database': os.path.basename(pdb_dir)}

# Write the results to a file
with open('/fast/sandner/Thesis/Thesis/Final_data/Neubauer/results_alphafold_scores_neubauer.txt', 'w') as f:
    for file, result in results.items():
        f.write(f'{file} ({result["database"]}): avg_pLDDT={result["avg_plddt"]:.2f}, above_threshold={result["above_threshold"]:.2f}%\n')

# Create pandas dataframe
data = []
for file, result in results.items():
    data.append({'uniprot_key': file, 'avg_pLDDT': result["avg_plddt"], 'above_threshold': result["above_threshold"]})
df = pd.DataFrame(data)
print(df)

# Write the results to a csv for plotting
df.to_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/results_alphafold_scores_neubauer.csv", index=False)


# plot the above threshold scores
df = pd.read_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/results_alphafold_scores_neubauer.csv")
plt.hist(df['above_threshold'], bins=100)
plt.title('Residues above threshold')
plt.xlabel('Residues above threshold [%]')
plt.xticks(np.arange(0, 100, step=10))
plt.ylabel('Number of proteins')
plt.savefig("results_above_threshold.png")

# filter for enzymes with certain amount of residues under threshold
enzyme_ids = []

with open("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/results_alphafold_scores_neubauer.txt", 'r') as f:
    count = 0
    for line in f:
        enzyme_name = line.split(':')[0]
        uniprot_id = enzyme_name.split('.')[0]  # Extract the Uniprot ID
        line = line.split(':')[1]  # Remove the enzyme name
        above_threshold = line.split(', ')[1].split('=')[1].strip('%\n')  # Remove the percentage sign and newline character
        if float(above_threshold) < 70:
            print(f"{enzyme_name}: above_threshold={above_threshold}%")
            enzyme_ids.append(uniprot_id)
            count += 1
print(count)
print(enzyme_ids)

# check in dataset for frequency of found enzymes
import pandas as pd
final_df = pd.read_csv('/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes.csv')  
frequency_df = final_df['uniprot_key'].value_counts()
wrongly_predicted_df = frequency_df[frequency_df.index.isin(enzyme_ids)]
print(wrongly_predicted_df.sum())

final_df.insert(2, 'below_threshold', final_df['uniprot_key'].isin(enzyme_ids).astype(int))
print(final_df)
print(final_df.loc[final_df["below_threshold"] == 1])
final_df.to_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes.csv", index=False)