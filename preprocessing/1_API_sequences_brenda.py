# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 11:13:37 2024

@author: lausa
"""

import requests
import pandas as pd

def fetch_sequence(uniprot_id, df):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    
    if response.status_code == 200:
        # Extract sequence from FASTA format
        lines = response.text.splitlines()
        sequence = ''.join(lines[1:])  # Skip the first line which is the header
        print(sequence)
        df.loc[df.uniprot_key == uniprot_id, ['sequence']] = sequence
    else:
        print(f"Failed to retrieve data for {uniprot_id}")
        

def read_csv_and_fetch_sequences(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    df.columns
    print(df)
        
    if "sequence" not in df.columns:
         df["sequence"] = None
         
    # amount of sequence columns which have entries (unique entries --> no replicates included) 
    print(len(df.loc[~df.sequence.isna()].sequence.unique()))
    
    # # Assuming the UniProt IDs are in a column named "uniprot_key"
    uniprot_ids = df.loc[df.sequence.isna()].uniprot_key.unique().tolist()
    # amount of unique uniprot keys given in the csv 
    print(len(uniprot_ids))
    
    # # Fetch the sequence for each unique UniProt ID
    for i, uniprot_id in enumerate(set(uniprot_ids)):
         print(f"Fetching sequence for {uniprot_id}")
         fetch_sequence(uniprot_id, df)
         
         if i % 100 == 0:
             df.to_csv("C:\\Users\\lausa\\OneDrive\\Desktop\\Thesis\\Project\\Brenda_2024_1.csv", index = False)
             print("Saving successful")
             
    df.to_csv("C:\\Users\\lausa\\OneDrive\\Desktop\\Thesis\\Project\\Brenda_2024_1.csv", index = False)
             

# Example usage
if __name__ == "__main__":
    input_csv = "C:\\Users\\lausa\\OneDrive\\Desktop\\Thesis\\Project\\Brenda_2024_1.csv"
    
    # Fetch sequences and update the DataFrame
    read_csv_and_fetch_sequences(input_csv)
    

