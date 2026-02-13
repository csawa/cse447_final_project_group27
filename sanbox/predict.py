from datasets import load_dataset
import multiprocess

class MyModel:
    def __init__(self, n_gram=3):
        self.n = n_gram
        self.lookup_table = {}

    def train(self, corpus):
        """builds dfa of prefix to transition"""
        temp_counts = {} # {prefix: {char: count}}

        for text in corpus:
            if len(text) <= self.n:
                continue
            for i in range(len(text) - self.n):
                prefix = text[i : i + self.n]
                next_char = text[i + self.n]
                
                if prefix not in temp_counts:
                    temp_counts[prefix] = {}
                temp_counts[prefix][next_char] = temp_counts[prefix].get(next_char, 0) + 1
        
        self.lookup_table = {
            prefix: max(chars, key=chars.get) 
            for prefix, chars in temp_counts.items()
        }
        # DONE :)

    def predict(self, current_prefix):
        return self.lookup_table.get(current_prefix[-self.n:], None)

# dataset
def get_training_corpus():
    dataset = load_dataset('Davlan/sib200', 'eng_Latn', split='train')
    return dataset['text']

# test
# corpus = get_training_corpus()
# model = MyModel(n_gram=4) # Using 4-gram for slightly better context
# model.train(corpus)

# infernce:
# context = "The pre"
# prediction = model.predict(context)
# print(f"Prefix: '{context}' -> Prediction: '{prediction}'")