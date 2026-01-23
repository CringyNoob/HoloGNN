import os
from collections import defaultdict

def scan_directory():
    root_dir = "."
    ext_counts = defaultdict(int)
    ext_examples = {}

    print(f"--- SCANNING PROJECT STRUCTURE ---")
    
    # Walk through all folders
    for root, dirs, files in os.walk(root_dir):
        # Skip the .git folder so it doesn't clutter the results
        if ".git" in dirs:
            dirs.remove(".git")
            
        for filename in files:
            # Get the file extension (e.g., .csv, .pth)
            name, ext = os.path.splitext(filename)
            ext = ext.lower()
            
            if not ext:
                ext = "[No Extension]"
            
            # Count it
            ext_counts[ext] += 1
            
            # Save the first one we find as an example
            if ext not in ext_examples:
                # Make path readable
                full_path = os.path.join(root, filename)
                ext_examples[ext] = full_path

    # Print the Report
    print(f"{'EXTENSION':<15} | {'COUNT':<8} | {'EXAMPLE PATH'}")
    print("-" * 70)
    
    # Sort by most common files first
    for ext, count in sorted(ext_counts.items(), key=lambda item: item[1], reverse=True):
        example = ext_examples[ext]
        print(f"{ext:<15} | {count:<8} | {example}")

if __name__ == "__main__":
    scan_directory()