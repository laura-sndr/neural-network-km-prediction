import pandas as pd

final_df = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv")
fingerprints_brenda_df = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/brenda_complete_fingerprints.csv")
fingerprints_sabio_df = pd.read_csv("/homes/chemie/sandner/Thesis/Final_data/sabio_complete_fingerprints.csv")

print(fingerprints_brenda_df.columns.get_loc("0")) # 214
print(fingerprints_sabio_df.columns.get_loc("0")) # 216

fingerprints_brenda = fingerprints_brenda_df.iloc[:, 214:]
fingerprints_sabio = fingerprints_sabio_df.iloc[:, 216:]

print(fingerprints_brenda)
print(fingerprints_sabio)

