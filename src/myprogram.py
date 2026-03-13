
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
      
    def __init__(self, n_embd=256, n_heads=8):
        self.n_embd = n_embd
        self.n_heads = n_heads
        self.vocab = { "\x02": 0, "\x03": 1, "\ufffd": 2 } # PAD, EOS, UNK
        self.itos = {0: "\x02", 1: "\x03", 2: "\ufffd"}
        self.model = None
    
    @classmethod
    def load_training_data(cls):
        # your code here
        configs = get_dataset_config_names('Davlan/sib200')
        all_data = []
        # anchor_langs = ['eng_Latn', 'spa_Latn', 'zho_Hans']
        # for lang in anchor_langs:
        #     try:
        #         data = load_dataset('Davlan/sib200', lang, split='train')
        #         # Add more for these to help the model learn real patterns
        #         all_data.extend(data['text']) 
        #     except: continue
        
        # Sample other languages
        for lang in configs:
            try:
                # use split='train' to get the data directly
                data = load_dataset('Davlan/sib200', lang, split='train')
                # append only the text strings
                all_data.extend(data['text'][:500])
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
        self.model = MiniTransformer(v_size, self.n_embd, self.n_heads).to(DEVICE)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        
        max_seq_len = 128 
        encoded_data = []
        for text in data:
            if len(text) < 2: continue
            tokens = self.encode(text[:max_seq_len])
            encoded_data.append(torch.tensor(tokens, dtype=torch.long))
        batch_size = 64

        for epoch in range(10):
            random.shuffle(encoded_data)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, len(encoded_data), batch_size):
                batch = encoded_data[i:i+batch_size]
                x_padded = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0).to(DEVICE)
                
                if x_padded.shape[1] < 2: continue
                
                logits = self.model(x_padded[:, :-1])
                targets = x_padded[:, 1:]
                
                loss = F.cross_entropy(logits.reshape(-1, v_size), targets.reshape(-1), ignore_index=0)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # safe division
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")


    def run_pred(self, data):
        """The core requirement: returns the 3 most likely next characters."""
        self.model.eval()
        predictions = []
        for text in data:
            # empty lines/prompts
            input_text = text if text else "\x02"
            
            # encode and move to device
            idx = torch.tensor([self.encode(input_text)], device=DEVICE)
            
            with torch.no_grad():
                logits = self.model(idx)
                # get predictions for the last character position
                last_logits = logits[0, -1, :]
                # find the top 3 most likely character indices
                _, top_indices = torch.topk(last_logits, 3)
                
            # convert indices to chars and join into a single string (e.g., "yWA")
            chars = [self.itos[i.item()] for i in top_indices]
            predictions.append("".join(chars))
        return predictions
    def save(self, work_dir):
        data = {
            'vocab': self.vocab, 
            'itos': self.itos, 
            'n_embd': self.n_embd, 
            'n_heads': self.n_heads
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
        # check whether file exists and has content
        if not os.path.exists(meta_path) or os.path.getsize(meta_path) == 0:
            raise FileNotFoundError(f"Missing or empty metadata at {meta_path}. Please run 'train' mode first.")
        # load metadata dictionary
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
        
        # initialize with saved hyperparameters
        instance = cls(n_embd=data['n_embd'], n_heads=data['n_heads'])
        instance.vocab = data['vocab']
        instance.itos = data['itos']
        # reconstruct pytorch model architecture
        v_size = len(instance.vocab)
        instance.model = MiniTransformer(v_size, instance.n_embd, instance.n_heads).to(DEVICE)
        
        # load actual weights into pytorch model
        if os.path.exists(model_weights_path):
            instance.model.load_state_dict(torch.load(model_weights_path, map_location=DEVICE))
            instance.model.eval() # Set to evaluation mode for testing
        else:
            print("Warning: model.pt weights file not found. Model will be untrained.")
            
        return instance
class MiniTransformer(nn.Module):
    def __init__(self, v_size, n_embd, n_heads):
        super().__init__()
        self.tok_emb = nn.Embedding(v_size, n_embd)
        self.pos_emb = nn.Parameter(torch.zeros(1, 256, n_embd))
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, v_size)
        self.n_heads = n_heads
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:, :T, :]
        
        for _ in range(3):
            x = x + self_attention(self.ln(x), self.ln(x), self.ln(x), n_heads=self.n_heads)
            x = x + self.ffn(self.ln(x))
        return self.head(self.ln(x))
def self_attention(Q, K, V, n_heads=1, causal=True):
    """
    Optimized version for the MiniTransformer.
    Works for 3D (B, T, D) or 4D (B, H, T, D) inputs.
    """
    # if 3D, add a dummy head dimension to make it 4D: (B, 1, T, D)
    if Q.dim() == 3:
        Q = Q.unsqueeze(1)
        K = K.unsqueeze(1)
        V = V.unsqueeze(1)
    B, H, T, D = Q.shape
    scaling = D**0.5
    # compute attention Scores: (B, H, T, T)
    attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / scaling
    if causal:
        mask = torch.tril(torch.ones(T, T, device=Q.device))
        attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
    # softmax and weighted sum
    attn_probs = torch.softmax(attn_weights, dim=-1)
    y = torch.matmul(attn_probs, V) # (B, H, T, D)
    # reshape back to 3D: (B, T, H*D)
    y = y.transpose(1, 2).contiguous().view(B, T, -1)
    
    return y

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
