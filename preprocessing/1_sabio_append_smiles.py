import pandas as pd
from rdkit.Chem.rdmolfiles import MolToSmiles, MolFromSmiles

# filter sabio dataframe

df_sequences = pd.read_csv("/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_WT_sequence.tsv", sep='\t')
df_smiles = pd.read_csv("/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_WT_smiles.tsv" ,sep='\t')

# sabio data with only smiles and index number
df_only_smiles = df_smiles.iloc[:,[0,10]]

# append smile column based on index number
combined_sabio = pd.merge(df_sequences, df_only_smiles, on='Unnamed: 0')
print("combined sabio count:", combined_sabio.shape[0]) # --> 11962 rows
print("columns:", combined_sabio.columns)
print("#################################")

# filter combined data for empty entries
sabio_filtered = combined_sabio.dropna()
print("sabio DB count after droping na:", sabio_filtered.shape[0]) # --> 10352 rows
print("sabio DB unique smiles: ", sabio_filtered.smiles.unique().shape[0])

# rename uniprot ID column to match brenda data
sabio_filtered.rename(columns={"UniprotID":"uniprot_key", "Value": "km_value"}, inplace=True)

# create canonical smiles for better comparison
for smiles in sabio_filtered['smiles']:
    mol = MolFromSmiles(smiles)  # Convert SMILES string to RDKit Mol object
    if mol:  # Check if Mol object creation was successful
        canonical_smiles = MolToSmiles(mol)  # Convert back to canonical SMILES

sabio_filtered['canonical_smiles'] = sabio_filtered['smiles'].apply(lambda x: MolToSmiles(MolFromSmiles(x)) if MolFromSmiles(x) else None)

# save combined data 
sabio_filtered.to_csv("/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_COMBINED.tsv" ,sep='\t')


# filter brenda dataframe

df_brenda_sequences = pd.read_csv("/homes/chemie/sandner/Thesis/Project/brenda_2024_1_WTs_length.csv")

# delete all rows with invalid smiles
brenda_filtered = df_brenda_sequences[df_brenda_sequences.smiles != "unknown"]
print("Brenda db with known smiles:", brenda_filtered.shape[0]) # --> 18131 rows
print("Unique smiles in brenda:", brenda_filtered.smiles.unique().shape[0])

# create canonical smiles for better comparison
for smiles in brenda_filtered['smiles']:
    if isinstance(smiles, str):  # Ensure the SMILES is a string
        mol_brenda = MolFromSmiles(smiles)  # Convert SMILES string to RDKit Mol object
        if mol_brenda:  # Check if Mol object creation was successful
            canonical_smiles = MolToSmiles(mol_brenda)  # Convert back to canonical SMILES

# Remove rows where 'smiles' is NaN or not a string
brenda_filtered = brenda_filtered[brenda_filtered['smiles'].apply(lambda x: isinstance(x, str))]

# Now safely convert SMILES to canonical SMILES
brenda_filtered['canonical_smiles'] = brenda_filtered['smiles'].apply(lambda x: MolToSmiles(MolFromSmiles(x)) if MolFromSmiles(x) else None)

# save combined data
brenda_filtered.to_csv("/fast/sandner/sabio/Processing/brenda_2024_1_COMBINED.csv")

# compare sabio data to brenda data

# Merge the DataFrames based on uniprot_key, canonical smiles and km_value to avoid loosing different km_value
# for the same canonical smiles and uniprot_key.
merged_df = sabio_filtered.merge(brenda_filtered, on=['uniprot_key', 'canonical_smiles', "km_value"], how='inner')

# Get the number of matching rows
matching_rows_count = merged_df.shape[0]
print(f"Number of matching rows between sabio and brenda: {matching_rows_count}")

# Remove the matching rows from s_df based on uniprot_key, smiles and km_value
new_rows_from_sabio = sabio_filtered.loc[(~(
    sabio_filtered.uniprot_key.isin(merged_df.uniprot_key.tolist()) &
    sabio_filtered.canonical_smiles.isin(merged_df.canonical_smiles.tolist()) &
    sabio_filtered.km_value.isin(merged_df.km_value.tolist()))) &
    (sabio_filtered.sequence.str.len() <= 1024)
]

print("New rows from sabio:", new_rows_from_sabio.shape[0])
# Unique uniprot_ids to fetch from alphafold database:
unique_uniprot_ids = sabio_filtered.loc[
    (~sabio_filtered.uniprot_key.isin(brenda_filtered.uniprot_key.tolist())) &
    (sabio_filtered.sequence.str.len() <= 1024)
].uniprot_key.unique()
print("unique sabio uniprot_ids:", unique_uniprot_ids.shape[0])

# 2073 unique proteins (when adding .unique()). remove to save.

# save the filtered sabio data:
# new_rows_from_sabio.to_csv("/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_final.tsv" ,sep='\t')

# # save the  sabio data for PDB fetching:
# unique_uniprot_ids.to_csv("/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_for_PDB.tsv" ,sep='\t')

# --> use this dataframe to create PDB files from alphafold database and predict the binding sites