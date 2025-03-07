
import pubchempy as pcp
import pandas as pd
from tqdm import tqdm

# Function to get SMILES from molecule name
def get_smiles(molecule_name):
    compound = pcp.get_compounds(molecule_name, 'name')
    if compound:
        return compound[0].canonical_smiles
    else:
        return None

# Read molecule names from a CSV file
df = pd.read_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes.csv")  # Replace with your file path

# Initialize tqdm for progress bar
tqdm.pandas()

# Apply the SMILES retrieval function
df['smiles'] = df['substrate'].progress_apply(get_smiles)

print(df)

# Save the results to a new CSV file
df.to_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes.csv", index=False)

print("SMILES retrieval completed and saved.")


from rdkit.Chem import PandasTools, Descriptors, rdMolDescriptors
import torch
import torch.nn.functional as F

# hide the warnings from CalcMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

print(df.columns.values)

# check for missing smiles
missing_smiles_sum = df['smiles'].isna().sum()
print(f"Number of missing or NaN smiles before: {missing_smiles_sum}")

# remove entries containing empty cells
df = df.dropna()

missing_smiles_sum = df['smiles'].isna().sum()
print(f"Number of missing or NaN smiles after: {missing_smiles_sum}")


# add the rdkit mol objects to the dataframe
PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles', molCol='mol_obj')

# check for invalid mol objects
invalid_mols = df[df['mol_obj'].isnull()]
print(invalid_mols)

# continue only with valid mol objects
df = df.dropna(subset=['mol_obj'])

print(df)

# calculate the descriptors
ligands_desc_list = [Descriptors.CalcMolDescriptors(mol) for mol in df['mol_obj']]
# convert the resulting list of dictionaries to a dataframe
ligands_desc_df = pd.DataFrame(ligands_desc_list)
# remove the columns containing empty cells
ligands_desc_df = ligands_desc_df.dropna(axis=1)

print(ligands_desc_df)

# Reset index before concatenating to ensure indices match
df = df.reset_index(drop=True)
ligands_desc_df = ligands_desc_df.reset_index(drop=True)

# combine the descriptor dataframe with the BRENDA dataframe
combined = [df, ligands_desc_df]
descriptors = pd.concat(combined, axis=1)
print(descriptors)

s = descriptors.eq(0).any()
s[s].tolist()
print(s)

descriptors.to_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes_descriptors.csv")

df = pd.read_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes_descriptors.csv")

# Recalculate the mol_obj
PandasTools.AddMoleculeColumnToFrame(df, smilesCol='smiles', molCol='mol_obj')

# Define function to calculate 1024-bit Morgan fingerprint
def calculate_morgan_fingerprint(mol, radius=2, n_bits=2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)

# Calculate Morgan fingerprints
fingerprints = [calculate_morgan_fingerprint(mol) for mol in df['mol_obj'] if mol is not None]

print(fingerprints)

# Convert the bit vectors to lists and then to DataFrames
fingerprints_df = pd.DataFrame([list(fp) for fp in fingerprints])

# save the fingerprint DataFrames 
print(fingerprints_df)

# Reset index before concatenating to ensure indices match
df = df.reset_index(drop=True)

fingerprints_df = fingerprints_df.reset_index(drop=True)

# combine the descriptor dataframe with the BRENDA dataframe
combined = [df, fingerprints_df]

final_df = pd.concat(combined, axis=1)

print(final_df)

# keep only relevant columns
final_df = final_df[["uniprot_key", "below_threshold", "sequence"]].join(final_df.iloc[:, 7:])
print(final_df)

final_df.to_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes_descriptors_fingerprints.csv", index=False)