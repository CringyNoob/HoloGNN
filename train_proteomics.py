import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random

# Custom Modules
from src.full_model import HoloGNN
from transformers import EsmTokenizer

# --- CONFIGURATION ---
STABILITY_MODEL_PATH = "holognn_stability_final.pth" 
BATCH_SIZE = 8
EPOCHS = 10   # Increased slightly to ensure it learns the rule
LR = 1e-4     # Lower learning rate for fine-tuning

# --- 1. BIOLOGICALLY REALISTIC DATASET ---
class StructuralDiseaseDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.data = []
        
        print(f"Generating {num_samples} samples based on 'Helix Breaker' physics...")
        base_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        
        for _ in range(num_samples):
            seq_list = list(base_seq)
            idx = random.randint(5, len(base_seq)-5) # Avoid ends
            
            # --- THE BIOLOGICAL RULE ---
            # Scenario A: Destructive Mutation (Proline/Glycine) -> DISEASE
            if random.random() > 0.5:
                mut_aa = random.choice(['P', 'G']) # Helix breakers
                label = 1.0
            # Scenario B: Safe Mutation (Alanine/Valine) -> BENIGN
            else:
                mut_aa = random.choice(['A', 'V', 'I', 'L']) # Stable hydrophobic
                label = 0.0
                
            seq_list[idx] = mut_aa
            mutated_seq = "".join(seq_list)
            self.data.append((mutated_seq, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq, label = self.data[idx]
        tokens = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=100, padding="max_length")
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def train_proteomics():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- TRANSFER LEARNING ON {device} ---")

    # 1. Load Pre-Trained Model
    print("Loading Pre-trained Physics Brain...")
    model = HoloGNN()
    try:
        model.load_state_dict(torch.load(STABILITY_MODEL_PATH, map_location=device), strict=False)
        print("✅ Stability weights loaded.")
    except FileNotFoundError:
        print("❌ CRITICAL: Stability weights not found!")
        return

    # 2. Reset Head
    print("Initializing Fresh Proteomics Head (320 -> 1)...")
    model.proteomics_head = nn.Linear(320, 1) 
    model.to(device)

    # 3. Freeze Backbone
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.proteomics_head.parameters():
        param.requires_grad = True

    # 4. Setup Training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.proteomics_head.parameters(), lr=LR)

    dataset = StructuralDiseaseDataset(num_samples=2000) # More data
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 5. Train
    print("--- LEARNING 'HELIX BREAKER' RULE ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in progress:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            class DataBatch: pass
            data = DataBatch()
            data.input_ids = input_ids
            data.mask = mask
            data.edge_index = None

            optimizer.zero_grad()
            
            # Forward Pass (With Fixes)
            features = model.backbone(data.input_ids, data.mask)
            if isinstance(features, tuple): features = features[0]
            features = features.mean(dim=1) # Pooling
            
            predictions = model.proteomics_head(features)
            loss = criterion(predictions.squeeze(), labels)
            
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
            # Accuracy
            probs = torch.sigmoid(predictions.squeeze())
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            progress.set_postfix({'acc': correct/total, 'loss': total_loss/(total/BATCH_SIZE)})

        print(f"Epoch {epoch+1}: Accuracy = {correct/total:.4f}")

    print("--- SUCCESS! MODEL SAVED. ---")
    torch.save(model.state_dict(), "holognn_disease_classifier.pth")

if __name__ == "__main__":
    train_proteomics()