import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from transformers import EsmTokenizer
from Bio.Seq import Seq # Biopython for DNA->Protein translation

class MassIVEKBDataset(Dataset):
    """
    Loader for MassIVE-KB. 
    NOTE: The current file seems to lack explicit 'RetentionTime' tags.
    This loader will extract SEQUENCES for pre-training.
    """
    def __init__(self, data_dir, max_length=100):
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.data = []
        self.max_length = max_length
        
        print(f"Scanning {data_dir} for .sptxt files...")
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith(".sptxt"):
                    self._parse_sptxt(os.path.join(root, file))
        
        print(f"Loaded {len(self.data)} sequences from MassIVE-KB.")

    def _parse_sptxt(self, filepath):
        with open(filepath, 'r') as f:
            for line in f:
                # Target Line: Name: AAPSPSGGGGS.../3
                if line.startswith('Name:'):
                    try:
                        # Split by space to get the ID part, then split by '/' to remove charge
                        raw_content = line.strip().split(' ')[1]
                        sequence = raw_content.split('/')[0]
                        
                        # Clean sequence (remove non-amino acid chars if any)
                        sequence = ''.join([char for char in sequence if char.isalpha()])
                        
                        # For now, we assign a dummy label (0.0) since RT is missing
                        if len(sequence) > 0:
                            self.data.append({'seq': sequence, 'label': 0.0})
                    except IndexError:
                        continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(item['seq'], padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(item['label'], dtype=torch.float)
        }

class MegaScaleDataset(Dataset):
    """
    Loader for Mega-scale cDNA.
    Translates 'dna_seq' -> Protein Sequence.
    Target: 'deltaG' (Stability).
    """
    def __init__(self, csv_path, max_length=100):
        self.tokenizer = EsmTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
        self.max_length = max_length
        
        print(f"Loading Mega-scale data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Filter for valid rows
        self.df = self.df.dropna(subset=['dna_seq', 'deltaG'])
        print(f"Loaded {len(self.df)} samples. (Translating DNA to Protein on-the-fly...)")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # 1. Get DNA and Translate to Protein
        dna = row['dna_seq']
        # Biopython translation
        protein_seq = str(Seq(dna).translate(to_stop=True))
        
        # 2. Get Label (Stability / deltaG)
        label = float(row['deltaG'])
        
        # 3. Tokenize
        encoding = self.tokenizer(
            protein_seq,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.float)
        }