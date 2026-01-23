import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset
from transformers import EsmTokenizer

# Custom Modules
from src.dataset import MegaScaleDataset
from src.full_model import HoloGNN

# --- CONFIGURATION ---
MODEL_PATH = "holognn_stability_final.pth"
DATA_PATH = "data/mega_scale_cdna/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"
TEST_SAMPLES = 1000  # We test on 1000 random proteins for the graph
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_model():
    print("--- 1. LOADING MODEL FOR EVALUATION ---")
    model = HoloGNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("--- 2. PREPARING TEST DATA ---")
    # We load the dataset but only take a small "Test Set"
    full_dataset = MegaScaleDataset(DATA_PATH)
    
    # We grab the LAST 1000 samples (assuming the model trained on the first ones)
    total_len = len(full_dataset)
    test_indices = range(total_len - TEST_SAMPLES, total_len)
    test_dataset = Subset(full_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    print(f"Testing on {len(test_dataset)} unseen proteins...")

    predictions = []
    actuals = []

    print("--- 3. RUNNING INFERENCE ---")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            # Data Object
            class DataBatch: pass
            data = DataBatch()
            data.input_ids = input_ids
            data.mask = mask
            data.edge_index = None # Use the Graph Builder

            preds = model(data, task="idr")
            
            predictions.extend(preds.squeeze().tolist())
            actuals.extend(labels.tolist())

    # --- 4. CALCULATE METRICS ---
    r_value, _ = pearsonr(actuals, predictions)
    print(f"\n[RESULT] Pearson Correlation (R): {r_value:.4f}")

    # --- 5. GENERATE FIGURE 1 ---
    plt.figure(figsize=(8, 6))
    plt.scatter(actuals, predictions, alpha=0.5, s=10, c='blue')
    plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--') # Perfect line
    plt.xlabel("Actual Stability (Experimental)")
    plt.ylabel("Predicted Stability (Holo-GNN)")
    plt.title(f"Holo-GNN Validation\nPearson R = {r_value:.4f}")
    
    # Save the plot
    plt.savefig("Figure_1_Correlation.png")
    print("Graph saved to: Figure_1_Correlation.png")

if __name__ == "__main__":
    evaluate_model()