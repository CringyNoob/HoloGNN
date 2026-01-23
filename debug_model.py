import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from src.dataset import MegaScaleDataset
from src.full_model import HoloGNN

# --- DEBUG CONFIG ---
BATCH_SIZE = 8
LR = 1e-3  # High learning rate to force learning
EPOCHS = 50 # Force memorization
SAMPLES = 100 # Tiny dataset

DATA_PATH = "data/mega_scale_cdna/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset1_20230416.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def debug():
    print(f"--- STARTING OVERFIT TEST on {device} ---")
    
    # 1. Load tiny data
    full_dataset = MegaScaleDataset(DATA_PATH)
    subset = Subset(full_dataset, range(SAMPLES))
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 2. Model
    model = HoloGNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    
    loss_history = []
    
    # 3. Aggressive Training Loop
    print("Trying to memorize 100 proteins...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            data = type('Data', (), {})() # Empty object
            data.input_ids = input_ids
            data.mask = mask
            data.edge_index = None
            
            optimizer.zero_grad()
            preds = model(data, task="idr")
            loss = criterion(preds.squeeze(), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        loss_history.append(avg_loss)
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")

    # 4. Check Results
    print("\n--- CHECKING MEMORIZATION ---")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader: # Test on the SAME data we trained on
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            data = type('Data', (), {})()
            data.input_ids = input_ids
            data.mask = mask
            data.edge_index = None
            
            preds = model(data, task="idr")
            all_preds.extend(preds.squeeze().tolist())
            all_labels.extend(labels.tolist())
            
    # 5. Print Metrics
    r, _ = pearsonr(all_labels, all_preds)
    print(f"\n[DIAGNOSTIC RESULT] R-Score: {r:.4f}")
    
    if r > 0.8:
        print("✅ SUCCESS: Model CAN learn. The previous failure was just under-training.")
    else:
        print("❌ FAILURE: Model cannot even memorize. Code bug exists.")

    # 6. Show Scatter Plot
    plt.scatter(all_labels, all_preds)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Overfit Test: R={r:.4f}")
    plt.savefig("debug_plot.png")
    print("Saved debug_plot.png")

if __name__ == "__main__":
    debug()