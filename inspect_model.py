import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

# Load and inspect the checkpoint
checkpoint_path = "holognn_stability_final.pth"

print(f"=== Inspecting: {checkpoint_path} ===\n")

# Check if file exists
if not os.path.exists(checkpoint_path):
    print(f"ERROR: File not found!")
    exit(1)

file_size = os.path.getsize(checkpoint_path) / (1024**2)
print(f"File size: {file_size:.2f} MB")
print(f"Last modified: {os.path.getmtime(checkpoint_path)}")

# Load the checkpoint
print("\nLoading checkpoint...")
checkpoint = torch.load(checkpoint_path, map_location='cpu')

# Check what type of data it is
if isinstance(checkpoint, dict):
    print(f"\nCheckpoint type: Dictionary with {len(checkpoint)} keys")
    print(f"Keys: {list(checkpoint.keys())}")
    
    # Check for common training checkpoint fields
    if 'epoch' in checkpoint:
        print(f"\n✓ Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"✓ Best validation loss: {checkpoint['best_val_loss']:.4f}")
    if 'model_state_dict' in checkpoint:
        print(f"✓ Model state dict contains {len(checkpoint['model_state_dict'])} parameters")
    if 'optimizer_state_dict' in checkpoint:
        print(f"✓ Optimizer state saved")
    if 'training_history' in checkpoint:
        history = checkpoint['training_history']
        print(f"\n=== Training History ===")
        if 'train_loss' in history:
            print(f"Training losses recorded: {len(history['train_loss'])} epochs")
            print(f"Final training loss: {history['train_loss'][-1]:.4f}")
        if 'val_loss' in history:
            print(f"Validation losses recorded: {len(history['val_loss'])} epochs")
            print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
else:
    # It's a state_dict directly (only model weights)
    print(f"\nCheckpoint type: Direct state_dict with {len(checkpoint)} parameters")
    print("\nSample parameter names:")
    for i, key in enumerate(list(checkpoint.keys())[:10]):
        param = checkpoint[key]
        print(f"  {key}: shape {list(param.shape)}")
    
    if len(checkpoint) > 10:
        print(f"  ... and {len(checkpoint) - 10} more parameters")

print("\n=== Analysis Complete ===")
