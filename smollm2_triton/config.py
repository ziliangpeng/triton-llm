"""SmolLM2 configuration — fully compatible with Llama architecture.

SmolLM2 uses `LlamaForCausalLM` from HuggingFace Transformers.
See https://huggingface.co/HuggingFaceTB/SmolLM2-135M
"""
from dataclasses import dataclass


@dataclass
class SmolLM2Config:
    """Configuration for a SmolLM2 / Llama model.

    Attributes
    ----------
    vocab_size : int
        Size of the vocabulary.
    hidden_size : int
        Dimensionality of embeddings and hidden states (n_embd).
    num_hidden_layers : int
        Number of transformer layers (n_layer).
    num_attention_heads : int
        Number of query attention heads (n_head).
    num_key_value_heads : int
        Number of key/value attention heads for GQA. If equal to
        num_attention_heads, this is standard MHA.
    intermediate_size : int
        Size of the SwiGLU feed-forward hidden layer.
    max_position_embeddings : int
        Maximum sequence length (context window).
    rms_norm_eps : float
        Epsilon for RMSNorm numerical stability.
    rope_theta : float
        Base frequency for Rotary Position Embedding.
    tie_word_embeddings : bool
        Whether LM head shares weights with token embeddings.
    hidden_act : str
        Activation function ('silu' for SwiGLU).
    torch_dtype : str
        Default tensor dtype ('bfloat16' for training, float32 for inference).
    """
    vocab_size: int = 49152
    hidden_size: int = 576
    num_hidden_layers: int = 30
    num_attention_heads: int = 9
    num_key_value_heads: int = 3
    intermediate_size: int = 1536
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 100000.0
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"
    torch_dtype: str = "float32"

    @classmethod
    def from_pretrained(cls, name: str) -> "SmolLM2Config":
        """Return config for a known SmolLM2 variant."""
        variants = {
            "SmolLM2-135M": SmolLM2Config(),
            "SmolLM2-360M": SmolLM2Config(
                hidden_size=960,
                num_hidden_layers=32,
                num_attention_heads=15,
                num_key_value_heads=5,
                intermediate_size=2560,
            ),
            "SmolLM2-1.7B": SmolLM2Config(
                hidden_size=2048,
                num_hidden_layers=24,
                num_attention_heads=32,
                num_key_value_heads=32,
                intermediate_size=8192,
            ),
            # Instruct variants — same architecture, different weights
            "SmolLM2-135M-Instruct": SmolLM2Config(),
            "SmolLM2-360M-Instruct": SmolLM2Config(
                hidden_size=960,
                num_hidden_layers=32,
                num_attention_heads=15,
                num_key_value_heads=5,
                intermediate_size=2560,
            ),
            "SmolLM2-1.7B-Instruct": SmolLM2Config(
                hidden_size=2048,
                num_hidden_layers=24,
                num_attention_heads=32,
                num_key_value_heads=32,
                intermediate_size=8192,
            ),
        }
        if name not in variants:
            raise ValueError(f"Unknown variant: {name!r}. Available: {list(variants.keys())}")
        return variants[name]

    # Aliases for GPT-2 naming compatibility in shared kernel code
    @property
    def n_embd(self): return self.hidden_size
    @property
    def n_layer(self): return self.num_hidden_layers
    @property
    def n_head(self): return self.num_attention_heads
    @property
    def n_kv_head(self): return self.num_key_value_heads
    @property
    def n_positions(self): return self.max_position_embeddings
    @property
    def n_ffn(self): return self.intermediate_size
