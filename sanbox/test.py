import numpy as np
from datasets import load_dataset

# import multiprocess
# try:
#     multiprocess.set_start_method('spawn', force=True)
# except RuntimeError:
#     pass

# # To specifically suppress the ResourceTracker noise
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module='multiprocess')

class UltimateDFAPredictor:
    def __init__(self, n_gram=3):
        self.n = n_gram
        self.vocab = {} # char -> int
        self.states = {} # prefix_tuple -> int
        self.transitions = None # 2D Array: [state_id][char_id] -> next_state_id
        self.predictions = None # 1D Array: [state_id] -> best_char_id

    def train(self, corpus):
        # 1. Build Vocab and State Map
        unique_chars = sorted(list(set("".join(corpus))))
        self.vocab = {c: i for i, c in enumerate(unique_chars)}
        self.inv_vocab = {i: c for c, i in self.vocab.items()}
        
        raw_counts = {} # state_id -> {char_id: count}
        state_counter = 0
        
        for text in corpus:
            for i in range(len(text) - self.n):
                prefix = text[i : i + self.n]
                next_char = text[i + self.n]
                
                if prefix not in self.states:
                    self.states[prefix] = state_counter
                    state_counter += 1
                
                s_id = self.states[prefix]
                c_id = self.vocab[next_char]
                
                if s_id not in raw_counts: raw_counts[s_id] = {}
                raw_counts[s_id][c_id] = raw_counts[s_id].get(c_id, 0) + 1

        # 2. Build the NumPy DFA Table
        num_states = len(self.states)
        num_chars = len(self.vocab)
        self.predictions = np.zeros(num_states, dtype=np.uint32)
        self.transitions = np.zeros((num_states, num_chars), dtype=np.int32) - 1

        for prefix, s_id in self.states.items():
            # Set the prediction (output of the DFA)
            if s_id in raw_counts:
                best_c_id = max(raw_counts[s_id], key=raw_counts[s_id].get)
                self.predictions[s_id] = best_c_id
            
            # Pre-calculate the "Jump" to the next state for every possible character
            for char, c_id in self.vocab.items():
                next_prefix = (prefix + char)[1:]
                if next_prefix in self.states:
                    self.transitions[s_id, c_id] = self.states[next_prefix]

        print(f"DFA optimized. Memory: {self.transitions.nbytes / 1024:.2f} KB")

    def predict_sequence(self, start_prefix, length=10):
        """Pure DFA Inference: No strings, just array indexing."""
        res = []
        curr_state = self.states.get(start_prefix)
        if curr_state is None: return "<unk>"

        for _ in range(length):
            # 1. Get prediction from current state
            char_id = self.predictions[curr_state]
            res.append(self.inv_vocab[char_id])
            
            # 2. Transition to next state
            curr_state = self.transitions[curr_state, char_id]
            if curr_state == -1: break
            
        return "".join(res)

# --- Run ---
dataset = load_dataset('Davlan/sib200', 'eng_Latn', split='train')
model = UltimateDFAPredictor(n_gram=4)
# model.train(dataset['text'])
model.train("Hello, World")

# Start the machine
context = "Hello, Worl"
print(f"Context: '{context}'")
print(f"Generated: {model.predict_sequence(context, length=1)}")


# import time

# start = time.perf_counter()
# for _ in range(10000):
#     model.predict_sequence("The presid", length = 3)
# end = time.perf_counter()

# print(f"10,000 predictions took: {end - start:.6f} seconds")
# print(f"Average time per char: {(end - start)/10000:.9f} seconds")
# print(f"Th{model.predict_sequence("The ", length = 3)}")

# import graphviz

# def visualize_dfa(model, seed_prefix, depth=10):
#     dot = graphviz.Digraph(comment='DFA Subgraph')
#     dot.attr(rankdir='LR')  # Left to Right orientation
    
#     current_prefix = seed_prefix
    
#     for i in range(depth):
#         prediction = model.predict_sequence(current_prefix, length = 3)
#         if not prediction:
#             break
            
#         next_prefix = (current_prefix + prediction)[-model.n:]
        
#         # Create nodes and edges
#         # Node labels show the prefix; edges show the predicted character
#         dot.node(current_prefix, current_prefix)
#         dot.node(next_prefix, next_prefix)
#         dot.edge(current_prefix, next_prefix, label=f" '{prediction}'")
        
#         current_prefix = next_prefix

#     # Save and render
#     dot.render('dfa_path', format='png', cleanup=True)
#     print("DFA visualization saved as dfa_path.png")

# # --- Usage ---
# # After training your model:
# visualize_dfa(model, "The ", depth=15)