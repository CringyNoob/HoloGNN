import pandas as pd

def peek_file(filepath, name):
    print(f"\n--- INSPECTING {name} ---")
    try:
        if filepath.endswith('.csv'):
            df = pd.read_csv(filepath, nrows=2)
            print("Columns found:", list(df.columns))
            print("First row sample:")
            print(df.iloc[0])
        elif filepath.endswith('.sptxt'):
            with open(filepath, 'r') as f:
                print("First 10 lines of text:")
                for i in range(10):
                    print(repr(f.readline().strip()))
        elif filepath.endswith('.vcf'):
            with open(filepath, 'r') as f:
                print("First 10 lines (skipping comments):")
                count = 0
                for line in f:
                    if not line.startswith('##'):
                        print(line.strip())
                        count += 1
                    if count >= 5: break
    except Exception as e:
        print(f"Error reading {name}: {e}")

# 1. Inspect MassIVE-KB
massive_path = "data/massive_kb/LIBRARY_TO_SPTXT-3440aba4-download_sptxt_library-main.sptxt"
peek_file(massive_path, "MassIVE-KB")

# 2. Inspect Mega-scale cDNA (The processed one)
mega_path = "data/mega_scale_cdna/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"
peek_file(mega_path, "Mega-scale cDNA")

# 3. Inspect FireProtDB
fire_path = "data/fireprotdb/fireprotdb_20251015-164116.csv"
peek_file(fire_path, "FireProtDB")