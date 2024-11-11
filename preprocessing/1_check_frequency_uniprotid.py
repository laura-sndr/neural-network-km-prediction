# check in dataset for frequency of found enzymes
import pandas as pd

final_df = pd.read_csv('/homes/chemie/sandner/Thesis/Final_data/final_df.csv') 
other_df = pd.read_csv("/fast/sandner/train_val_test/test_set_ids.csv")

enzyme_ids = other_df.iloc[: ,0].unique()

frequency_df = final_df['uniprot_key'].value_counts()
count_df = frequency_df[frequency_df.index.isin(enzyme_ids)]
print(count_df)
print(count_df.sum())

