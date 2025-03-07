import pandas as pd

neubauer = pd.read_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes_descriptors_fingerprints.csv")
train = pd.read_csv("/fast/sandner/Thesis/Thesis/Final_data/final_df_fingerprints_alphafold_cluster.csv")

print(neubauer.columns.difference(train.columns))

# Get columns to drop (those not in 'train')
dropcolumns = neubauer.columns.difference(train.columns).to_list()

# Drop the columns from 'neubauer' that are not in 'train'
neubauer.drop(columns=dropcolumns, inplace=True)

# Print columns that are still in 'neubauer' but not in 'train' (should be empty now)
print(neubauer.columns.difference(train.columns))
print(train.columns.difference(neubauer.columns))
print(neubauer)

neubauer.to_csv("/fast/sandner/Thesis/Thesis/Final_data/Neubauer/Neubauer_enzymes_descriptors_fingerprints.csv", index=False)