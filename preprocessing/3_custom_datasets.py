import pandas as pd

# Load the original DataFrame
original_df = pd.read_csv('/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold.csv')  # Update the path accordingly

# Load the UniProt IDs for each set and ensure they are unique
train_ids = pd.read_csv('/fast/sandner/train_val_test/train_set_ids.csv')['Sequence_ID'].astype(str).str.strip().tolist()
val_ids = pd.read_csv('/fast/sandner/train_val_test/val_set_ids.csv')['Sequence_ID'].astype(str).str.strip().tolist()
test_ids = pd.read_csv('/fast/sandner/train_val_test/test_set_ids.csv')['UniProt ID'].astype(str).str.strip().tolist()

print(train_ids[:10])  # print the first 10 IDs in the train set
print(val_ids[:10])    # print the first 10 IDs in the val set
print(test_ids[:10])   # print the first 10 IDs in the test set

print(original_df['uniprot_key'].head())  # print the first few IDs in the uniprot_key column

# Print unique IDs and their lengths for verification
print("Unique Train IDs:", train_ids)
print("Number of Unique Train IDs:", len(train_ids))

print("Unique Validation IDs:", val_ids)
print("Number of Unique Validation IDs:", len(val_ids))

print("Unique Test IDs:", test_ids)
print("Number of Unique Test IDs:", len(test_ids))

# Check how many times each UniProt ID appears in the original DataFrame
original_id_counts = original_df['uniprot_key'].value_counts()
print("\nUniProt ID counts in the original DataFrame:")
print(original_id_counts)

# Create a new column in the original DataFrame to indicate which set each row belongs to
original_df.insert(3, 'set', None)
original_df.loc[original_df.uniprot_key.isin(train_ids), ['set']] = 'train'
original_df.loc[original_df.uniprot_key.isin(val_ids), ['set']] = 'val'
original_df.loc[original_df.uniprot_key.isin(test_ids), ['set']] = 'test'

# Print the counts after filtering
print(f"\Original DataFrame length: {len(original_df)}")
print(f"\nUnique uniprot_keys in original: {len(original_df.uniprot_key.unique())}")
print(f"\nTrain DataFrame length: {len(original_df.loc[original_df['set'] == 'train'])}")
print(f"\n Train IDs length: {len(train_ids)}")
print(f"\nVal DataFrame length: {len(original_df.loc[original_df['set'] == 'val'])}")
print(f"\n Train IDs length: {len(val_ids)}")
print(f"\nTest DataFrame length: {len(original_df.loc[original_df['set'] == 'test'])}")
print(f"\n Train IDs length: {len(test_ids)}")


original_df.to_csv("/homes/chemie/sandner/Thesis/Final_data/final_df_alphafold_cluster.csv", index=False)