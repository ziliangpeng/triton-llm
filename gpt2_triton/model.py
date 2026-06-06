"""
GPT-2 Model with forward and generate using Triton kernels.

No KV cache (full recompute each decode step).  Batch=1 only.
"""

import numpy as np

from .config import GPT2Config
from .kernels.add import add
from .kernels.attention import attention
from .kernels.embedding import embedding
from .kernels.gelu import gelu
from .kernels.gemm import gemm
from .kernels.layernorm import layer_norm
from .kernels.softmax import softmax


class GPT2Model:
    """GPT-2 inference model backed by Triton kernels.

    Parameters
    ----------
    config : GPT2Config
        Model configuration.
    weights : dict of str -> np.ndarray
        Dictionary mapping HF weight names to NumPy arrays.
        Weight matrices are transposed from HF's (out, in) layout
        to (in, out) at load time.
    """

    def __init__(self, config: GPT2Config, weights: dict):
        self.config = config
        n_layer = config.n_layer

        # --- Embedding tables (stored as-is, (vocab, n_embd) and (pos, n_embd)) ---
        # Wte also serves as lm_head (tied embeddings).
        # Pre-transpose for gemm(hidden, wte.T): store (n_embd, vocab_size) C-contiguous.
        self.wte = np.require(
            weights["wte.weight"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )
        self.lm_head_w = np.require(
            self.wte.T, dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )
        self.wpe = np.require(
            weights["wpe.weight"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )

        # --- Final layer norm ---
        self.ln_f_g = np.require(
            weights["ln_f.weight"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )
        self.ln_f_b = np.require(
            weights["ln_f.bias"], dtype=np.float32, requirements=["C_CONTIGUOUS"]
        )

        # --- Per-layer weight lists ---
        self.ln_1_g: list[np.ndarray] = []
        self.ln_1_b: list[np.ndarray] = []
        self.ln_2_g: list[np.ndarray] = []
        self.ln_2_b: list[np.ndarray] = []
        self.c_attn_w: list[np.ndarray] = []
        self.c_attn_b: list[np.ndarray] = []
        self.c_proj_w: list[np.ndarray] = []
        self.c_proj_b: list[np.ndarray] = []
        self.c_fc_w: list[np.ndarray] = []
        self.c_fc_b: list[np.ndarray] = []
        self.c_proj_mlp_w: list[np.ndarray] = []
        self.c_proj_mlp_b: list[np.ndarray] = []

        for i in range(n_layer):
            # --- Attention block ---
            # c_attn: HF (3*n_embd, n_embd) -> store .T as (n_embd, 3*n_embd)
            w = weights[f"h.{i}.attn.c_attn.weight"]
            self.c_attn_w.append(
                np.require(w.T, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_attn_b.append(
                np.require(
                    weights[f"h.{i}.attn.c_attn.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # c_proj: HF (n_embd, n_embd) -> store .T as (n_embd, n_embd)
            w = weights[f"h.{i}.attn.c_proj.weight"]
            self.c_proj_w.append(
                np.require(w.T, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_proj_b.append(
                np.require(
                    weights[f"h.{i}.attn.c_proj.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # --- MLP block ---
            # c_fc: HF (4*n_embd, n_embd) -> store .T as (n_embd, 4*n_embd)
            w = weights[f"h.{i}.mlp.c_fc.weight"]
            self.c_fc_w.append(
                np.require(w.T, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_fc_b.append(
                np.require(
                    weights[f"h.{i}.mlp.c_fc.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # c_proj (MLP output): HF (n_embd, 4*n_embd) -> store .T as (4*n_embd, n_embd)
            w = weights[f"h.{i}.mlp.c_proj.weight"]
            self.c_proj_mlp_w.append(
                np.require(w.T, dtype=np.float32, requirements=["C_CONTIGUOUS"])
            )
            self.c_proj_mlp_b.append(
                np.require(
                    weights[f"h.{i}.mlp.c_proj.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

            # --- Layer norms ---
            self.ln_1_g.append(
                np.require(
                    weights[f"h.{i}.ln_1.weight"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )
            self.ln_1_b.append(
                np.require(
                    weights[f"h.{i}.ln_1.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )
            self.ln_2_g.append(
                np.require(
                    weights[f"h.{i}.ln_2.weight"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )
            self.ln_2_b.append(
                np.require(
                    weights[f"h.{i}.ln_2.bias"],
                    dtype=np.float32,
                    requirements=["C_CONTIGUOUS"],
                )
            )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Run a full forward pass over *all* input tokens.

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Input token IDs.

        Returns
        -------
        logits : np.ndarray, shape ``(1, seq, vocab_size)``, float32
            Unnormalised logits for each position.
        """
        config = self.config
        n_layer = config.n_layer
        n_embd = config.n_embd

        # --- Embedding ---
        hidden = embedding(token_ids, self.wte, self.wpe)  # (1, seq, n_embd)

        # --- Transformer layers ---
        for i in range(n_layer):
            # --- Attention sub-block ---
            residual = hidden
            h = hidden.reshape(-1, n_embd)                     # (seq, n_embd)
            h = layer_norm(h, self.ln_1_g[i], self.ln_1_b[i],
                           config.layer_norm_epsilon)
            h = self._apply_attention(h, i)                     # (seq, n_embd)
            hidden = add(h.reshape(1, -1, n_embd), residual)

            # --- MLP sub-block ---
            residual = hidden
            h = hidden.reshape(-1, n_embd)                     # (seq, n_embd)
            h = layer_norm(h, self.ln_2_g[i], self.ln_2_b[i],
                           config.layer_norm_epsilon)
            h = self._apply_mlp(h, i)                           # (seq, n_embd)
            hidden = add(h.reshape(1, -1, n_embd), residual)

        # --- Final layer norm ---
        h = hidden.reshape(-1, n_embd)                          # (seq, n_embd)
        h = layer_norm(h, self.ln_f_g, self.ln_f_b,
                       config.layer_norm_epsilon)

        # --- LM head ---
        logits = gemm(h, self.lm_head_w)  # (seq, n_embd) @ (n_embd, vocab) -> (seq, vocab)
        return logits.reshape(1, -1, config.vocab_size)

    # ------------------------------------------------------------------
    # Attention sub-block
    # ------------------------------------------------------------------

    def _apply_attention(self, h: np.ndarray, layer_idx: int) -> np.ndarray:
        """QKV projection -> split -> per-head causal attention -> output proj."""
        n_embd = self.config.n_embd
        n_head = self.config.n_head
        d_k = n_embd // n_head

        # QKV projection: (seq, n_embd) -> (seq, 3*n_embd)
        qkv = gemm(h, self.c_attn_w[layer_idx])  # (seq, 3*n_embd)
        qkv = add(qkv, np.broadcast_to(self.c_attn_b[layer_idx], qkv.shape))

        # Split into Q, K, V (each (seq, n_embd))
        q_all = qkv[:, :n_embd]
        k_all = qkv[:, n_embd:2 * n_embd]
        v_all = qkv[:, 2 * n_embd:]

        seq = h.shape[0]
        attn_out = np.empty((seq, n_embd), dtype=np.float32)

        # Per-head causal attention
        for head in range(n_head):
            s = head * d_k
            e = s + d_k
            q_h = q_all[:, s:e]   # (seq, d_k)
            k_h = k_all[:, s:e]   # (seq, d_k)
            v_h = v_all[:, s:e]   # (seq, d_k)
            o_h = attention(q_h, k_h, v_h)  # (seq, d_k)
            attn_out[:, s:e] = o_h

        # Output projection: (seq, n_embd) -> (seq, n_embd)
        out = gemm(attn_out, self.c_proj_w[layer_idx])  # (seq, n_embd)
        return add(out, np.broadcast_to(self.c_proj_b[layer_idx], out.shape))

    # ------------------------------------------------------------------
    # MLP sub-block
    # ------------------------------------------------------------------

    def _apply_mlp(self, h: np.ndarray, layer_idx: int) -> np.ndarray:
        """FC up -> GELU -> FC down."""
        # (seq, n_embd) -> (seq, 4*n_embd)
        h = gemm(h, self.c_fc_w[layer_idx])  # (seq, 4*n_embd)
        h = add(h, np.broadcast_to(self.c_fc_b[layer_idx], h.shape))
        h = gelu(h)  # (seq, 4*n_embd)
        # (seq, 4*n_embd) -> (seq, n_embd)
        out = gemm(h, self.c_proj_mlp_w[layer_idx])  # (seq, n_embd)
        return add(out, np.broadcast_to(self.c_proj_mlp_b[layer_idx], out.shape))

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        token_ids: np.ndarray,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> np.ndarray:
        """Autoregressive generation.

        No KV cache — full forward pass at each decode step.

        Parameters
        ----------
        token_ids : np.ndarray, shape ``(1, seq)``, int32
            Prompt token IDs.
        max_new_tokens : int
            Number of tokens to generate.
        temperature : float
            Sampling temperature.  Values close to 0 produce greedy
            decoding; values > 0 enable stochastic sampling.
        top_k : int
            If > 0, restrict sampling to the top-k most probable tokens.

        Returns
        -------
        out : np.ndarray, shape ``(1, seq + max_new_tokens)``, int32
            Prompt extended with generated tokens.
        """
        if token_ids.shape[0] != 1:
            raise ValueError(f"Batch size must be 1, got {token_ids.shape[0]}")

        for _ in range(max_new_tokens):
            logits = self.forward(token_ids)          # (1, seq, V)
            next_logits = logits[0, -1, :]            # (V,)

            if temperature > 1e-6:
                # Temperature-scaled sampling
                scaled = next_logits / temperature
                probs = softmax(scaled)                # (V,)

                # Top-k filtering
                if top_k > 0:
                    indices = np.argpartition(probs, -top_k)[-top_k:]
                    filtered = np.zeros_like(probs)
                    filtered[indices] = probs[indices]
                    filtered /= filtered.sum()
                    probs = filtered

                next_token = int(np.random.choice(len(probs), p=probs))
            else:
                # Greedy
                next_token = int(np.argmax(next_logits))

            token_ids = np.concatenate(
                [token_ids, np.array([[next_token]], dtype=np.int32)],
                axis=1,
            )

        return token_ids
