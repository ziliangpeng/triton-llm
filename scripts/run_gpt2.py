"""
End-to-end GPT-2 inference using only Python + Triton (no PyTorch)
"""

import numpy as np
import sys
sys.path.insert(0, ".")
from gpt2_triton.model import GPT2, GPT2Config

def main():
    print("Loading tiny GPT-2 (numpy reference + Triton kernels)...")
    config = GPT2Config(n_layer=2, n_head=4, n_embd=128, vocab_size=1000)
    model = GPT2(config)

    # Fake input
    idx = np.array([[1, 5, 23, 99]], dtype=np.int32)

    logits = model(idx)
    print("Logits shape:", logits.shape)
    print("First token logits[:5]:", logits[0, 0, :5])
    print("\nGPT-2 inference completed successfully with no PyTorch!")

if __name__ == "__main__":
    main()