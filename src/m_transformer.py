#!/usr/bin/env python
from ast import List
import os
import pickle
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, get_dataset_config_names
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
      
    def __init__(self, n_embd=256):
        self.n_embd = n_embd
        self.vocab = { "\x02": 0, "\x03": 1, "\ufffd": 2 } # PAD, EOS, UNK
        self.itos = {0: "\x02", 1: "\x03", 2: "\ufffd"}
        self.model = None
    
    @classmethod
    def load_training_data(cls):
        # your code here
        configs = get_dataset_config_names('Davlan/sib200')
        all_data = []

        # Sample languages
        for lang in random.sample(configs, 40):
            try:
                # Use split='train' to get the data directly
                data = load_dataset('Davlan/sib200', lang, split='train')
                # Append only the text strings
                all_data.extend(data['text'][200:])
            except Exception:
                continue
        return all_data

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname, 'rt', encoding='utf-8') as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt', encoding='utf-8') as f: # added utf-8 for multilingual
            for p in preds:
                f.write('{}\n'.format(p))

    def build_vocab(self, data):
        """Extracts every unique character from the multilingual data."""
        unique_chars = set()
        for text in data:
            if isinstance(text, str):
                unique_chars.update(text)
        
        chars = sorted(list(unique_chars))
        for char in chars:
            if char not in self.vocab:
                idx = len(self.vocab)
                self.vocab[char] = idx
                self.itos[idx] = char
        
    def encode(self, s):
        return [self.vocab.get(c, self.vocab["\ufffd"]) for c in s]

    def decode(self, l):
        return "".join([self.itos.get(i, "\ufffd") for i in l])

    def run_train(self, data, work_dir):
        self.build_vocab(data)
        v_size = len(self.vocab)
        self.model = MiniTransformer(v_size, self.n_embd).to(DEVICE)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        for epoch in range(10): # Train for a few epochs
            print(f"Epoch {epoch+1}/10")
            for text in data:
                if len(text) < 2: continue
                idx = torch.tensor([self.encode(text)], device=DEVICE)
                # Predict next character (standard language modeling)
                logits = self.model(idx[:, :-1]) 
                targets = idx[:, 1:]
                loss = F.cross_entropy(logits.view(-1, v_size), targets.view(-1))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    def run_pred(self, data):
        """The core requirement: returns the 3 most likely next characters."""
        self.model.eval()
        predictions = []

        for text in data:
            # Handle empty lines/prompts
            input_text = text if text else "\x02"
            
            # Encode and move to device
            idx = torch.tensor([self.encode(input_text)], device=DEVICE)
            
            with torch.no_grad():
                logits = self.model(idx)
                # Get the predictions for the last character position
                last_logits = logits[0, -1, :]
                # Find the top 3 most likely character indices
                _, top_indices = torch.topk(last_logits, 3)
                
            # Convert indices to characters and join into a single string (e.g., "yWA")
            chars = [self.itos[i.item()] for i in top_indices]
            predictions.append("".join(chars))

        return predictions

    def save(self, work_dir):
        data = {
            'vocab': self.vocab, 
            'itos': self.itos, 
            'n_embd': self.n_embd
        }
        # Using the 'with' statement ensures the file is closed and saved properly
        meta_path = os.path.join(work_dir, 'data.pkl')
        with open(meta_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        model_path = os.path.join(work_dir, 'model.pt')
        torch.save(self.model.state_dict(), model_path)
        print(f"Model and meta saved successfully to {work_dir}")
            

    @classmethod
    def load(cls, work_dir):
        # your code here
        meta_path = os.path.join(work_dir, 'data.pkl')
        model_weights_path = os.path.join(work_dir, 'model.pt')

        # 1. Safety check: Does the file exist and have content?
        if not os.path.exists(meta_path) or os.path.getsize(meta_path) == 0:
            raise FileNotFoundError(f"Missing or empty metadata at {meta_path}. Please run 'train' mode first.")

        # 2. Load the metadata dictionary
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        
        # 3. Initialize the MyModel instance with saved hyperparameters
        instance = cls(n_embd=data['n_embd'])
        instance.vocab = data['vocab']
        instance.itos = data['itos']

        # 4. Reconstruct the PyTorch model architecture
        v_size = len(instance.vocab)
        instance.model = MiniTransformer(v_size, instance.n_embd).to(DEVICE)
        
        # 5. Load the actual weights into the PyTorch model
        if os.path.exists(model_weights_path):
            instance.model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
            instance.model.eval() # Set to evaluation mode for testing
        else:
            print("Warning: model.pt weights file not found. Model will be untrained.")
            
        return instance

class MiniTransformer(nn.Module):
    def __init__(self, v_size, n_embd):
        super().__init__()
        self.tok_emb = nn.Embedding(v_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 1024, n_embd))
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, v_size)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        return self.head(self.ln(x))

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
