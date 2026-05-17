import sys
sys.path.insert(0, ".")
import numpy as np
from gpt2_triton.model import GPT2, Config

config = Config(vocab_size=1000, n_embd=128, n_layer=2)
model = GPT2(config)

idx = np.array([[1, 5, 23, 99, 42]], dtype=np.int32)
logits = model(idx)

print(f"Input shape : {idx.shape}")
print(f"Logits shape: {logits.shape}")
print("Minimal GPT-2 forward pass successful!")