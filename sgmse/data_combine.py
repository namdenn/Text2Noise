import os
import shutil

# 1. Define your specific folders and splits
folders = [
    'metadata_encoded', 
    'metadata_white_encoded', 
    'metadata_street_encoded', 
    'metadata_car_encoded', 
    'metadata_lr_encoded', 
    'metadata_cafe_encoded'
]
splits = ['train', 'test', 'val']

# 2. Create the destination directory
output_dir = "generated/metadata/raw"
os.makedirs(output_dir, exist_ok=True)

print(f"Starting merge into {output_dir}...")

# 3. Perform the merge
for split in splits:
    # We save directly into the new folder
    output_path = os.path.join(output_dir, f"{split}.jsonl")
    
    count = 0
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for fld in folders:
            file_path = os.path.join(fld, f"{split}.jsonl")
            
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as infile:
                    for line in infile:
                        line = line.strip()
                        if line:
                            outfile.write(line + '\n')
                            count += 1
                print(f"  [+] Merged {file_path}")
            else:
                print(f"  [!] Warning: {file_path} not found. Skipping.")

    print(f"Successfully created {output_path} with {count} total entries.\n")

print("--- Merging Complete ---")
print(f"Your combined files are located in: {os.path.abspath(output_dir)}")
