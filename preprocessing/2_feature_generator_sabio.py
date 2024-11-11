
import pandas as pd
from rdkit.Chem import PandasTools, Crippen, rdMolDescriptors, Descriptors
import torch
import torch.nn.functional as F
from tqdm import tqdm  # For progress bar

# hide the warnings from CalcMolDescriptors
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

df = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/KM_sabio_clean_unisubstrate_final.tsv", sep='\t')
print(df.columns.values)

# retreive subtrate name and smiles
ligands = df.loc[:,['Substrate','smiles']].drop_duplicates()
print(ligands)

# check for missing smiles
missing_smiles_sum = ligands['smiles'].isna().sum()
print(f"Number of missing or NaN Smiles before: {missing_smiles_sum}")

# remove entries containing empty cells
ligands = ligands.dropna()

missing_smiles_sum = ligands['smiles'].isna().sum()
print(f"Number of missing or NaN smiles after: {missing_smiles_sum}")


# add the rdkit mol objects to the dataframe
PandasTools.AddMoleculeColumnToFrame(ligands, smilesCol='smiles', molCol='mol_obj')

# check for invalid mol objects
invalid_mols = ligands[ligands['mol_obj'].isnull()]
print(invalid_mols)

# invalid_mols.to_csv("C:\\Users\\lausa\\Dropbox\\Thesis\\Project\\Brenda_2024_1_invalid_smiles.csv")

# continue only with valid mol objects
ligands = ligands.dropna(subset=['mol_obj'])

print(ligands)

# ligands.to_csv("C:\\Users\\lausa\\Dropbox\\Thesis\\Project\\Brenda_2024_1_valid_smiles.csv")

# calculate the descriptors
ligands_desc_list = [Descriptors.CalcMolDescriptors(mol) for mol in ligands['mol_obj']]
# convert the resulting list of dictionaries to a dataframe
ligands_desc_df = pd.DataFrame(ligands_desc_list)
# remove the columns containing empty cells
ligands_desc_df = ligands_desc_df.dropna(axis=1)

print(ligands_desc_df)

# Reset index before concatenating to ensure indices match
ligands = ligands.reset_index(drop=True)
ligands_desc_df = ligands_desc_df.reset_index(drop=True)

# combine the descriptor dataframe with the BRENDA dataframe
combined = [ligands, ligands_desc_df]
descriptors = pd.concat(combined, axis=1)
print(descriptors)

s = descriptors.eq(0).any()
s[s].tolist()
print(s)


descriptors.to_csv("/homes/chemie/sandner/Thesis/Final_data/KM_sabio_clean_unisubstrate_final_descriptors.tsv", sep='\t')
