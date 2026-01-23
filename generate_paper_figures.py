import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
import random
import numpy as np

# Custom Modules
from src.full_model import HoloGNN
from transformers import EsmTokenizer

# --- CONFIGURATION ---
# We load the Disease Classifier we just trained
MODEL_PATH = "holognn_disease_classifier.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- REUSING THE DATASET CLASS (Must be same as training) ---
class StructuralDiseaseDataset(Dataset):
    def __init__(self, num_samples=500):
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.data = []
        base_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
        for _ in range(num_samples):
            seq_list = list(base_seq)
            idx = random.randint(5, len(base_seq)-5)
            # Same rule: Proline/Glycine = Disease
            if random.random() > 0.5:
                mut_aa = random.choice(['P', 'G']) 
                label = 1.0
            else:
                mut_aa = random.choice(['A', 'V', 'I', 'L'])
                label = 0.0
            seq_list[idx] = mut_aa
            mutated_seq = "".join(seq_list)
            self.data.append((mutated_seq, label))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        seq, label = self.data[idx]
        tokens = self.tokenizer(seq, return_tensors="pt", truncation=True, max_length=100, padding="max_length")
        return {
            'input_ids': tokens['input_ids'].squeeze(),
            'attention_mask': tokens['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.float)
        }

def generate_figures():
    print("--- GENERATING FINAL PAPER FIGURES ---")
    
    # 1. Load Model
    model = HoloGNN()
    # Initialize the specific head structure we trained
    model.proteomics_head = nn.Linear(320, 1) 
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("✅ Disease Classifier loaded.")
    except FileNotFoundError:
        print("❌ Error: Model file not found.")
        return

    model.to(DEVICE)
    model.eval()

    # 2. Run Inference on Test Set
    dataset = StructuralDiseaseDataset(num_samples=500)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    
    y_true = []
    y_scores = []
    y_pred = []

    print("Running inference for graphs...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            class DataBatch: pass
            data = DataBatch()
            data.input_ids = input_ids
            data.mask = mask
            data.edge_index = None

            # Manual Forward Pass (Same as training)
            features = model.backbone(data.input_ids, data.mask)
            if isinstance(features, tuple): features = features[0]
            features = features.mean(dim=1) 
            predictions = model.proteomics_head(features)
            
            probs = torch.sigmoid(predictions.squeeze())
            
            y_true.extend(labels.cpu().numpy())
            y_scores.extend(probs.cpu().numpy())
            y_pred.extend((probs.cpu().numpy() > 0.5).astype(int))

    # --- FIGURE 2: CONFUSION MATRIX ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Pathogenic'], yticklabels=['Benign', 'Pathogenic'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Holo-GNN Diagnostic Accuracy')
    plt.savefig('Figure_2_ConfusionMatrix.png')
    print("Saved Figure_2_ConfusionMatrix.png")

    # --- FIGURE 3: ROC CURVE ---
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('Figure_3_ROC.png')
    print("Saved Figure_3_ROC.png")

if __name__ == "__main__":
    generate_figures()