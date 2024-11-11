import pandas as pd

sabio_enzyme = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/KM_sabio_clean_unisubstrate_final.tsv", sep='\t')
print(sabio_enzyme.shape)
sabio_substrate = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/KM_sabio_clean_unisubstrate_final_descriptors.tsv", sep='\t')
print(sabio_substrate.shape)

brenda_enzyme = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_2024_1_COMBINED.csv")
print(brenda_enzyme.shape)
brenda_substrate = pd.read_csv("/homes/chemie/sandner/Thesis/Project/Brenda_2024_1_descriptors_test.csv")
print(brenda_substrate.shape)

# combine enzyme and substrate information for sabio
sabio = pd.merge(sabio_enzyme, sabio_substrate, on="Substrate")
sabio.to_csv("/homes/chemie/sandner/Thesis/Final_data/sabio_complete.tsv", sep='\t')

# keep only relevant columns (km value, sequence, descriptors)
sabio = sabio[["km_value", "uniprot_key", "sequence"]].join(sabio.iloc[:, 19:217])
print(sabio)

# combine enzyme and substrate information for brenda
brenda = pd.merge(brenda_enzyme, brenda_substrate, on="substrate")
brenda.to_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_complete.csv")

# keep only relevant columns (km value, sequence, descriptors)
brenda = brenda[["km_value", "uniprot_key", "sequence"]].join(brenda.iloc[:, 18:215])
print(brenda)

final_df = pd.concat([brenda, sabio])
final_df = final_df.drop(columns="SPS") # column only exists in sabio data # 27259x197 
print(final_df) 
final_df = final_df.dropna() # 27259x197
final_df = final_df.drop_duplicates() # 26545x197
print(final_df) 

final_df.to_csv("/homes/chemie/sandner/Thesis/Final_data/final_df.csv", index=False)
