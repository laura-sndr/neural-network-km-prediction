
import pandas as pd
from rdkit import Chem
from rdkit.Chem import PandasTools, rdMolDescriptors
import torch
import torch.nn.functional as F
from tqdm import tqdm  # For progress bar

# hide the warnings from CalcMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

df_brenda = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_complete.csv")
df_sabio = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/sabio_complete.tsv", sep='\t')

print(df_brenda.shape)
print(df_sabio.shape)

# check for missing smiles
missing_smiles_brenda = df_brenda['mol_obj'].isna().sum()
missing_smiles_sabio = df_sabio['mol_obj'].isna().sum()
print(f"Number of missing or NaN mob_obj before (brenda): {missing_smiles_brenda}")
print(f"Number of missing or NaN mob_obj before (sabio): {missing_smiles_sabio}")

print(df_brenda['mol_obj'].apply(type))

# Recalculate the mol_obj
PandasTools.AddMoleculeColumnToFrame(df_brenda, smilesCol='canonical_smiles', molCol='mol_obj')
PandasTools.AddMoleculeColumnToFrame(df_sabio, smilesCol='canonical_smiles', molCol='mol_obj')

# # Convert mol_obj column from strings to RDKit Mol objects
# df_brenda['mol_obj'] = df_brenda['mol_obj'].apply(lambda x: convert_to_mol(x) if isinstance(x, str) else x)
# df_sabio['mol_obj'] = df_sabio['mol_obj'].apply(lambda x: convert_to_mol(x) if isinstance(x, str) else x)

print(df_brenda['mol_obj'].apply(type).value_counts())
print(df_sabio['mol_obj'].apply(type).value_counts())

print(df_brenda['mol_obj'].head())
print(df_sabio['mol_obj'].head())

# Define function to calculate 1024-bit Morgan fingerprint
def calculate_morgan_fingerprint(mol, radius=2, n_bits=2024):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

# Calculate Morgan fingerprints
fingerprints_brenda = [calculate_morgan_fingerprint(mol) for mol in df_brenda['mol_obj'] if mol is not None]
fingerprints_sabio = [calculate_morgan_fingerprint(mol) for mol in df_sabio['mol_obj'] if mol is not None]

print(fingerprints_brenda)
print(fingerprints_sabio)

# Convert the bit vectors to lists and then to DataFrames
fingerprints_df_brenda = pd.DataFrame([list(fp) for fp in fingerprints_brenda])
fingerprints_df_sabio = pd.DataFrame([list(fp) for fp in fingerprints_sabio])

# Optionally, save the fingerprint DataFrames or proceed with further analysis
print(fingerprints_df_brenda)
print(fingerprints_df_sabio)

# Reset index before concatenating to ensure indices match
df_brenda = df_brenda.reset_index(drop=True)
df_sabio = df_sabio.reset_index(drop=True)

fingerprints_df_brenda = fingerprints_df_brenda.reset_index(drop=True)
fingerprints_df_sabio = fingerprints_df_sabio.reset_index(drop=True)

# combine the descriptor dataframe with the BRENDA dataframe
combined_brenda = [df_brenda, fingerprints_df_brenda]
combined_sabio = [df_sabio, fingerprints_df_sabio]

final_df_brenda = pd.concat(combined_brenda, axis=1)
final_df_sabio = pd.concat(combined_sabio, axis=1)
print(final_df_brenda)
print(final_df_sabio)

s_brenda = final_df_brenda.eq(0).any()
s_sabio = final_df_sabio.eq(0).any()
s_brenda[s_brenda].tolist()
s_sabio[s_sabio].tolist()
print(s_brenda)
print(s_sabio)

final_df_brenda.to_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_complete_fingerprints.csv", index=False)
final_df_sabio.to_csv("/homes/chemie/sandner/Thesis/Final_data/sabio_complete_fingerprints.csv", sep=',', index=False)
