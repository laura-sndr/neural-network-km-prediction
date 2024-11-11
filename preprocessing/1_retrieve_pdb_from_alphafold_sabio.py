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
df = pd.read_csv('/fast/sandner/sabio/Processing/KM_sabio_clean_unisubstrate_for_PDB.tsv', sep='\t')

# Process only unique uniprot_keys
unique_keys = df.drop_duplicates(subset=['uniprot_key'])

# Loop through each unique uniprot_key
for index, row in tqdm(unique_keys.iterrows(), total=unique_keys.shape[0]):
    protein_name = row['uniprot_key']
    
    # # Create directory for the protein
    # protein_dir = os.path.join('/fast/sandner/output_pdbs', protein_name)  # Adjust the output directory
    # os.makedirs(protein_dir, exist_ok=True)
    
    # Determine the URL to download from
    url = f"https://alphafold.ebi.ac.uk/files/AF-{protein_name}-F1-model_v4.pdb"

    
    # Define the output file path
    output_file = os.path.join('/fast/sandner/output_pdbs_sabio', f"{protein_name}.pdb")
    
    # Download the PDB file
    download_pdb(url, output_file)
