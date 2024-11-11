import pandas as pd

df = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/final_df.csv")

# Drop duplicates based on 'uniprot_key' to ensure each key is unique
unique_proteins = df.drop_duplicates(subset='uniprot_key')

# Define the FASTA file path
fasta_file_path = "/homes/chemie/sandner/Thesis/Project/Blast/fasta_for_blast.fasta"

# Write the FASTA file
with open(fasta_file_path, 'w') as fasta_file:
    for index, row in unique_proteins.iterrows():
        uniprot_key = row['uniprot_key']
        sequence = row['sequence']
        fasta_file.write(f">{uniprot_key}\n")
        fasta_file.write(f"{sequence}\n")

print(f"FASTA file created for {len(unique_proteins)} unique UniProt IDs.")