#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import pickle
import numpy as np
from typing import List
from datasets import load_dataset


class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self, N: int = 3, k: int = 1e-7):
        self.N = N
        self.k = k
        self.vocab = set()
        self.ngram_counts = {}
        self.context_counts = {}
        self.sorted_vocab = []

        self.UNK = "\ufffd" # actual Unicode 'unknown' char
        self.PAD = "\x02"   # 'Start of Text' char
        self.EOS = "\x03"   # 'End of Text' char
    
    def process_text_for_Ngram(self, data, N: int, is_test: bool = False) -> List[str]:
        processed = []
        for item in data:
            # 1. Handle Hugging Face objects vs. raw strings
            if isinstance(item, dict):
                # If it's from load_training_data (HF Dataset)
                text = item.get('text', '')
            else:
                # If it's from load_test_data (tab-separated file string)
                parts = item.split('\t')
                text = parts[2] if len(parts) > 2 else item
            
            
            # start padding
            padding = self.PAD * (N - 1)
            if is_test:
                processed.append(padding + text)
            else:
                # Add a single-character EOS marker
                processed.append(padding + text + self.EOS)
        return processed
        
    @classmethod
    def load_training_data(cls):
        # your code here
        from datasets import load_dataset
        data = load_dataset('Davlan/sib200', 'eng_Latn') # FIX THIS BECAUSE WE DON'T KNOW THE LANGUAGE
        return data["train"]

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        # your code here
        processed = self.process_text_for_Ngram(data, self.N)
        
        # all characters seen in training
        raw_vocab = set("".join(processed))
        raw_vocab.add(self.UNK)
        self.sorted_vocab = sorted(list(raw_vocab))
        self.vocab = set(self.sorted_vocab)

        # track the "empty" context (global frequencies)
        self.context_counts = {}
        self.ngram_counts = {}

        self.unigram_counts = {}
        self.total_unigrams = 0

        for sent in processed:
            # adding unigram counts
            for char in sent:
                self.unigram_counts[char] = self.unigram_counts.get(char, 0) + 1
                self.total_unigrams += 1
            
            
            for i in range(len(sent) - self.N + 1):
                context = sent[i : i + self.N - 1]
                target = sent[i + self.N - 1]
                
                if context not in self.ngram_counts:
                    self.ngram_counts[context] = {}
                    self.context_counts[context] = 0
                
                self.ngram_counts[context][target] = self.ngram_counts[context].get(target, 0) + 1
                self.context_counts[context] += 1
        
        # store the model
        self.save(work_dir)

    def run_pred(self, data):
        # your code here
        processed_inputs = self.process_text_for_Ngram(data, self.N, is_test=True)
        preds = []
        V_list = list(self.vocab)
        V_size = len(self.vocab)

        # for sent in processed_inputs:
        #     # replace unseen chars with UNK marker
        #     safe_sent = "".join([c if c in self.vocab else self.UNK for c in sent])
            
        #     context = safe_sent[-(self.N - 1):] #if self.N > 1 else ""
        #     # If we haven't seen this character sequence, shorten it until we have
        #     while len(context) > 0 and context not in self.context_counts:
        #         context = context[1:]
            
        #     counts = self.ngram_counts.get(context, {})
        #     total = self.context_counts.get(context, 0)

        #     # add k smoothing
        #     probs = []
        #     for char in V_list:
        #         p = (counts.get(char, 0) + self.k) / (total + self.k * V_size)
        #         probs.append(p)
        
            # interpolation weights
        lambda3 = 0.6
        lambda2 = 0.3
        lambda1 = 0.1

        for sent in processed_inputs:

            safe_sent = "".join(
                [c if c in self.vocab else self.UNK for c in sent]
            )

            context_full = safe_sent[-(self.N - 1):]

            probs = []

            for char in V_list:

                # --- TRIGRAM ---
                context3 = context_full
                count3 = self.ngram_counts.get(context3, {}).get(char, 0)
                total3 = self.context_counts.get(context3, 0)

                P3 = (count3 + self.k) / (total3 + self.k * V_size) \
                    if total3 > 0 else 0


                # --- BIGRAM ---
                context2 = context_full[-1:]  # last character only
                count2 = self.ngram_counts.get(context2, {}).get(char, 0)
                total2 = self.context_counts.get(context2, 0)

                P2 = (count2 + self.k) / (total2 + self.k * V_size) \
                    if total2 > 0 else 0


                # --- UNIGRAM ---
                count1 = self.unigram_counts.get(char, 0)
                total1 = self.total_unigrams

                P1 = (count1 + self.k) / (total1 + self.k * V_size)


                # --- INTERPOLATION ---
                P = lambda3 * P3 + lambda2 * P2 + lambda1 * P1

                probs.append(P)

            # get top 3
            top_indices = np.argsort(probs)[::-1] #[-3:][::-1]
            top_guesses = [] # [V_list[i] for i in top_indices]
            for idx in top_indices:
                char = V_list[idx]
                if char not in [self.PAD, self.EOS, self.UNK]: # filter out eos and unk
                    top_guesses.append(char)
                if len(top_guesses) == 3:
                    break
            
            preds.append(''.join(top_guesses))
            
        return preds

    def save(self, work_dir):
        # your code here
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wb') as f:
            # f.write('dummy save')
            pickle.dump(self, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        with open(os.path.join(work_dir, 'model.checkpoint'), 'rb') as f:
            return pickle.load(f)
        # return MyModel()


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