import pandas as pd

sabio_enzyme = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/KM_sabio_clean_unisubstrate_final.tsv", sep='\t')
print(sabio_enzyme.shape) # 9132X14
sabio_descriptors_fingerprints = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/sabio_complete_fingerprints.csv")
print(sabio_descriptors_fingerprints.shape) # 9132x2240

brenda_enzyme = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_2024_1_COMBINED.csv")
print(brenda_enzyme.shape) #(18127, 13)
brenda_descriptors_fingerprints = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_complete_fingerprints.csv")
print(brenda_descriptors_fingerprints.shape) #(18127, 2238)

# create subsets with relevant columns
df_sabio_subset_1 = sabio_enzyme[["km_value", "uniprot_key", "sequence", "Substrate"]]
df_sabio_subset_2 = sabio_descriptors_fingerprints.iloc[:, [5] + list(range(18, sabio_descriptors_fingerprints.shape[1]))]

# combine enzyme and substrate information for sabio
sabio = pd.merge(df_sabio_subset_1, df_sabio_subset_2, on="Substrate", how="inner")
# sabio = sabio.drop_duplicates()
sabio.to_csv("/homes/chemie/sandner/Thesis/Final_data/sabio_complete_new.tsv", sep='\t')

print(sabio) # 8997x2226

# # combine enzyme and substrate information for brenda
# brenda = pd.merge(brenda_enzyme, brenda_substrate, on="substrate")
# brenda.to_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_complete.csv")

# # keep only relevant columns (km value, sequence, descriptors)
# brenda = brenda[["km_value", "uniprot_key", "sequence"]].join(brenda.iloc[:, 18:215])
# print(brenda)

# final_df = pd.concat([brenda, sabio])
# final_df = final_df.drop(columns="SPS") # column only exists in sabio data # 27259x197 
# print(final_df) 
# final_df = final_df.dropna() # 27259x197
# final_df = final_df.drop_duplicates() # 26545x197
# print(final_df) 

# final_df.to_csv("/homes/chemie/sandner/Thesis/Final_data/final_df.csv", index=False)
