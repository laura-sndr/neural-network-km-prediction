import os
import subprocess

# paths
input_dir = "/fast/sandner/input_files"
output_dir = "/fast/sandner/output_prank"
p2rank_executable = "/fast/sandner/p2rank/prank.sh"

# loop over all PDB files in the input directory
for pdb_file in os.listdir(input_dir):
    pdb_path = os.path.join(input_dir, pdb_file)
    output_path = os.path.join(output_dir, pdb_file.replace('.pdb', '_output'))

    # run p2rank
    cmd = [p2rank_executable, 'predict', '-f', pdb_path, '-o', output_path]
    subprocess.run(cmd)

    print(f"Processed {pdb_file}")

