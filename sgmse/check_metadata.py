import os

def count_lines(file_path):
    """Helper to count lines in a file efficiently."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def check_mismatch():
    base_path = os.environ.get(
        "METADATA_ROOT",
        os.path.dirname(os.path.abspath(__file__)),
    )
    
    raw_dir = os.path.join(base_path, "metadata")
    encoded_dir = os.path.join(base_path, "metadata_encoded")
    
    files_to_check = ["train.jsonl", "val.jsonl", "test.jsonl"]
    
    print(f"{'File Type':<15} | {'Raw Lines':<12} | {'Encoded Lines':<15} | {'Status'}")
    print("-" * 65)

    all_match = True

    for filename in files_to_check:
        raw_path = os.path.join(raw_dir, filename)
        enc_path = os.path.join(encoded_dir, filename)
        
        raw_count = count_lines(raw_path)
        enc_count = count_lines(enc_path)
        
        if raw_count is None or enc_count is None:
            status = "missing file"
            all_match = False
        elif raw_count == enc_count:
            status = "match"
        else:
            status = f"mismatch ({raw_count - enc_count} diff)"
            all_match = False
            
        r_str = str(raw_count) if raw_count is not None else "N/A"
        e_str = str(enc_count) if enc_count is not None else "N/A"
        print(f"{filename:<15} | {r_str:<12} | {e_str:<15} | {status}")

    if all_match:
        print("Success: All datasets are perfectly aligned.")
    else:
        print("Error: Detected discrepancies. You may need to re-run your encoding script.")

if __name__ == "__main__":
    check_mismatch()
