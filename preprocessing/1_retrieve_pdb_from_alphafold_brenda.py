import os
import pandas as pd
import requests
from tqdm import tqdm

# Function to download PDB files
def download_pdb(url, output_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            f.write(response.content)
    else:
        print(f"Failed to download {url}")

# Load your dataframe
df = pd.read_csv('/homes/chemie/sandner/Thesis/Project/brenda_2024_1_WTs_length.csv')

# Process only unique uniprot_keys
unique_keys = df.drop_duplicates(subset=['uniprot_key'])

# Loop through each unique uniprot_key
for index, row in tqdm(unique_keys.iterrows(), total=unique_keys.shape[0]):
    protein_name = row['uniprot_key']
    predicted_pdb = row['predicted_pdb']
    pdb_id = row['pdb_id']
    
    # # Create directory for the protein
    # protein_dir = os.path.join('/fast/sandner/output_pdbs', protein_name)  # Adjust the output directory
    # os.makedirs(protein_dir, exist_ok=True)
    
    # Determine the URL to download from
    if pd.notna(predicted_pdb):
        # Use the AlphaFold link
        url = predicted_pdb
    elif pd.notna(pdb_id):
        # Construct the PDB URL from the pdb_id
        url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_name}-F1-model_v4.pdb"
    else:
        continue  # Skip if neither is available
    
    # Define the output file path
    output_file = os.path.join('/fast/sandner/output_pdbs', f"{protein_name}.pdb")
    
    # Download the PDB file
    download_pdb(url, output_file)
