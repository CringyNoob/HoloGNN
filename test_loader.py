from src.dataset import MassIVEKBDataset, MegaScaleDataset
import torch

# 1. Test MassIVE-KB
print("--- TESTING MASSIVE-KB ---")
# This path points to the folder containing the extracted .sptxt
kb_path = "data/massive_kb" 
kb_dataset = MassIVEKBDataset(kb_path)

if len(kb_dataset) > 0:
    sample = kb_dataset[0]
    print(f"Success! Sample Input Shape: {sample['input_ids'].shape}")
else:
    print("Warning: MassIVE-KB still found 0 files. Check if .sptxt is in the folder.")

print("\n--- TESTING MEGA-SCALE cDNA ---")
# This points to the SPECIFIC CSV file we found
mega_path = "data/mega_scale_cdna/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"

try:
    mega_dataset = MegaScaleDataset(mega_path)
    sample = mega_dataset[0]
    print(f"Success! Transformed DNA to Protein.")
    print(f"Input Shape: {sample['input_ids'].shape}")
    print(f"Stability Label (DeltaG): {sample['label']}")
except FileNotFoundError:
    print("Error: Could not find the Mega-scale CSV file.")
except Exception as e:
    print(f"An error occurred: {e}")