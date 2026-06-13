"""
GPT-2 Configuration dataclass.

Provides a standard configuration container and factory methods to
instantiate known GPT-2 variants by name.
"""

from dataclasses import dataclass


@dataclass
class GPT2Config:
    """Configuration for a GPT-2 model.

    Attributes
    ----------
    vocab_size : int
        Size of the vocabulary (number of tokens).
    n_embd : int
        Dimensionality of token/position embeddings and hidden states.
    n_layer : int
        Number of transformer layers.
    n_head : int
        Number of attention heads per layer.
    n_positions : int
        Maximum sequence length supported by positional embeddings.
    layer_norm_epsilon : float
        Epsilon for LayerNorm numerical stability.
    """

    vocab_size: int = 50257
    n_embd: int = 768
    n_layer: int = 12
    n_head: int = 12
    n_positions: int = 1024
    layer_norm_epsilon: float = 1e-5

    @classmethod
    def from_pretrained(cls, name: str) -> "GPT2Config":
        """Return the default configuration for a known GPT-2 variant.

        Parameters
        ----------
        name : str
            One of ``"gpt2"``, ``"gpt2-medium"``, ``"gpt2-large"``,
            or ``"gpt2-xl"``.

        Returns
        -------
        GPT2Config
            Configuration matching the requested variant.
        """
        variants = {
            "gpt2": GPT2Config(),
            "gpt2-medium": GPT2Config(
                vocab_size=50257,
                n_embd=1024,
                n_layer=24,
                n_head=16,
                n_positions=1024,
            ),
            "gpt2-large": GPT2Config(
                vocab_size=50257,
                n_embd=1280,
                n_layer=36,
                n_head=20,
                n_positions=1024,
            ),
            "gpt2-xl": GPT2Config(
                vocab_size=50257,
                n_embd=1600,
                n_layer=48,
                n_head=25,
                n_positions=1024,
            ),
        }
        if name not in variants:
            raise ValueError(
                f"Unknown GPT-2 variant: {name!r}. "
                f"Available: {list(variants.keys())}"
            )
        return variants[name]
