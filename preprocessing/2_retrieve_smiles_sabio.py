
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

# Read your molecule names from a CSV file (or any other source)
# Assuming you have a CSV file with a column named 'molecule_name'
df = pd.read_csv("/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_WT.tsv", sep='\t')  # Replace with your file path

# Initialize tqdm for progress bar
tqdm.pandas()

# Apply the SMILES retrieval function
df['smiles'] = df['Substrate'].progress_apply(get_smiles)

print(df)

# Save the results to a new CSV file
df.to_csv("/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_WT_smiles.tsv", sep='\t', index=False)

print("SMILES retrieval completed and saved to 'molecules_with_smiles.csv'")
