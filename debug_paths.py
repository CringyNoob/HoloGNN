import os

print("--- DIAGNOSTIC START ---")
print(f"Current Working Directory: {os.getcwd()}")

target_dir = "data"
if not os.path.exists(target_dir):
    print(f"ERROR: The folder '{target_dir}' does not exist!")
else:
    print(f"\nScanning '{target_dir}' recursively:")
    found_any = False
    for root, dirs, files in os.walk(target_dir):
        print(f"\n[Folder] {root}")
        for f in files:
            print(f"  - {f}")
            found_any = True
    
    if not found_any:
        print("\nWARNING: No files found at all! Did you extract the zips?")

print("\n--- DIAGNOSTIC END ---")