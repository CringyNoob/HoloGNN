import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from src.full_model import HoloGNN
from transformers import EsmTokenizer

# --- CONFIGURATION ---
# This matches the file name in train.py
MODEL_PATH = "holognn_stability_final.pth" 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_stability(sequence):
    print(f"\n--- Analyzing Protein ---")
    print(f"Sequence: {sequence[:20]}... (Length: {len(sequence)})")
    
    # 1. Load the Model Architecture
    model = HoloGNN()
    
    # 2. Load the Trained Weights
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Could not find '{MODEL_PATH}'.")
        print("Did the training script finish successfully?")
        return

    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Successfully loaded: {MODEL_PATH}")
    except RuntimeError as e:
        print(f"Error loading weights: {e}")
        return

    model.to(DEVICE)
    model.eval() 

    # 3. Prepare Input
    tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=100)
    
    input_ids = inputs['input_ids'].to(DEVICE)
    mask = inputs['attention_mask'].to(DEVICE)

    # 4. Create Data Batch
    class DataBatch: pass
    data = DataBatch()
    data.input_ids = input_ids
    data.mask = mask
    data.edge_index = None # Forces Graph Construction

    # 5. Predict
    with torch.no_grad():
        prediction = model(data, task="idr")
    
    print(f"Predicted Stability (DeltaG): {prediction.item():.4f}")
    return prediction.item()

if __name__ == "__main__":
    # Test 1: Wild Type
    wt_seq = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    print("\n[Test 1] Predicting Wild Type...")
    val_wt = predict_stability(wt_seq)
    
    # Test 2: Mutant (M1A)
    mut_seq = "AQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    print("\n[Test 2] Predicting Mutant...")
    val_mut = predict_stability(mut_seq)

    if val_wt and val_mut:
        diff = val_mut - val_wt
        print(f"\n[RESULT] Predicted Change (DeltaDeltaG): {diff:.4f}")
        if diff < 0:
            print("Conclusion: Mutation is DESTABILIZING (matches physics).")
        else:
            print("Conclusion: Mutation is STABILIZING.")