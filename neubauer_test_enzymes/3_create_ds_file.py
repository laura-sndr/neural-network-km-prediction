import os

def create_dataset_file(directory, output_file):
    """
    Creates a `.ds` file containing paths to all files in the given directory.

    Args:
        directory (str): Path to the directory containing files.
        output_file (str): Path to the output `.ds` file.
    """
    # Open the output file for writing
    with open(output_file, 'w') as ds_file:
        # Iterate over all files in the directory
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                # Write the file path to the .ds file
                ds_file.write(file_path + '\n')


# Paths
create_dataset_file('/fast/sandner/Thesis/Thesis/Final_data/Neubauer/pdbs/pdbs', '/fast/sandner/Thesis/Thesis/Final_data/Neubauer/p2rank_list_neubauer.ds')
