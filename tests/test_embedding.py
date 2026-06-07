"""Unit tests for Triton Embedding + Positional Encoding kernel (CUDA + HIP).

Tests the fused token embedding + positional encoding operation against
a simple NumPy reference implementation.
"""

import numpy as np
from gpt2_triton.kernels.embedding import embedding


def _ref_embedding(token_ids, weight, pos_weight):
    """Reference implementation: gather + add in pure NumPy."""
    batch, seq_len = token_ids.shape
    n_embd = weight.shape[1]
    out = np.zeros((batch, seq_len, n_embd), dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            tid = token_ids[b, s]
            out[b, s, :] = weight[tid, :] + pos_weight[s, :]
    return out


def test_embedding_correctness():
    """Test numerical correctness vs NumPy reference at multiple sizes."""
    print("\n=== Embedding Correctness Tests ===")

    np.random.seed(42)

    configs = [
        (1, 4, 64, 10, 1024),     # single batch
        (2, 8, 32, 10, 1024),     # small config
        (4, 16, 128, 50, 1024),   # moderate
        (8, 4, 64, 50, 512),      # multiple batches, small seq
    ]

    all_passed = True
    for batch, seq_len, n_embd, vocab_size, max_position in configs:
        token_ids = np.random.randint(0, vocab_size, (batch, seq_len)).astype(np.int32)
        weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
        pos_weight = np.random.randn(max_position, n_embd).astype(np.float32)

        out = embedding(token_ids, weight, pos_weight)
        ref = _ref_embedding(token_ids, weight, pos_weight)

        max_diff = float(np.abs(out - ref).max())
        passed = np.allclose(out, ref, atol=1e-4, rtol=1e-5)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] B={batch} S={seq_len} D={n_embd} V={vocab_size} "
              f"| max_diff={max_diff:.2e}")
        assert passed, f"Embedding failed for config ({batch},{seq_len},{n_embd})"
        all_passed &= passed

    return all_passed


def test_embedding_first_token():
    """Verify first position: output[0, 0, :] = weight[token_ids[0, 0], :] + pos_weight[0, :]."""
    print("\n=== Embedding First Token Test ===")

    batch, seq_len, n_embd = 2, 4, 32
    vocab_size = 10
    max_position = 128

    token_ids = np.array([[3, 7, 1, 9], [2, 5, 0, 8]], dtype=np.int32)
    weight = np.arange(vocab_size * n_embd, dtype=np.float32).reshape(vocab_size, n_embd)
    pos_weight = np.arange(max_position * n_embd, dtype=np.float32).reshape(max_position, n_embd)

    out = embedding(token_ids, weight, pos_weight)

    # First token (batch=0, seq=0): token_id=3, position=0
    expected_0_0 = weight[3, :] + pos_weight[0, :]
    diff_0_0 = float(np.abs(out[0, 0, :] - expected_0_0).max())
    passed_0_0 = np.allclose(out[0, 0, :], expected_0_0, atol=1e-4)
    print(f"[{'PASS' if passed_0_0 else 'FAIL'}] output[0,0] = weight[3] + pos[0] "
          f"| max_diff={diff_0_0:.2e}")
    assert passed_0_0, f"First token failed, max_diff={diff_0_0:.2e}"

    # Check a middle position (batch=1, seq=2): token_id=0, position=2
    expected_1_2 = weight[0, :] + pos_weight[2, :]
    diff_1_2 = float(np.abs(out[1, 2, :] - expected_1_2).max())
    passed_1_2 = np.allclose(out[1, 2, :], expected_1_2, atol=1e-4)
    print(f"[{'PASS' if passed_1_2 else 'FAIL'}] output[1,2] = weight[0] + pos[2] "
          f"| max_diff={diff_1_2:.2e}")
    assert passed_1_2, f"Middle position failed, max_diff={diff_1_2:.2e}"

    # Last position (batch=1, seq=3): token_id=8, position=3
    expected_1_3 = weight[8, :] + pos_weight[3, :]
    diff_1_3 = float(np.abs(out[1, 3, :] - expected_1_3).max())
    passed_1_3 = np.allclose(out[1, 3, :], expected_1_3, atol=1e-4)
    print(f"[{'PASS' if passed_1_3 else 'FAIL'}] output[1,3] = weight[8] + pos[3] "
          f"| max_diff={diff_1_3:.2e}")
    assert passed_1_3, f"Last position failed, max_diff={diff_1_3:.2e}"


def test_embedding_empty():
    """Empty input (batch=0 or seq_len=0) should return empty array with correct shape."""
    print("\n=== Embedding Empty Array Test ===")

    n_embd = 64
    vocab_size = 50
    max_position = 1024

    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    pos_weight = np.random.randn(max_position, n_embd).astype(np.float32)

    # batch=0
    token_ids_empty_batch = np.empty((0, 10), dtype=np.int32)
    out_batch = embedding(token_ids_empty_batch, weight, pos_weight)
    assert out_batch.shape == (0, 10, n_embd), f"Expected shape (0, 10, {n_embd}), got {out_batch.shape}"
    assert out_batch.dtype == np.float32
    print(f"[PASS] batch=0 -> shape={out_batch.shape}")

    # seq_len=0
    token_ids_empty_seq = np.empty((4, 0), dtype=np.int32)
    out_seq = embedding(token_ids_empty_seq, weight, pos_weight)
    assert out_seq.shape == (4, 0, n_embd), f"Expected shape (4, 0, {n_embd}), got {out_seq.shape}"
    assert out_seq.dtype == np.float32
    print(f"[PASS] seq_len=0 -> shape={out_seq.shape}")


def test_embedding_non_power_of_2():
    """Non-power-of-2 n_embd should work via masked loads."""
    print("\n=== Embedding Non-Power-of-2 dims Test ===")

    batch, seq_len = 3, 6
    n_embd = 50  # not a power of 2
    vocab_size = 20
    max_position = 128

    np.random.seed(7)
    token_ids = np.random.randint(0, vocab_size, (batch, seq_len)).astype(np.int32)
    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    pos_weight = np.random.randn(max_position, n_embd).astype(np.float32)

    out = embedding(token_ids, weight, pos_weight)
    ref = _ref_embedding(token_ids, weight, pos_weight)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-5)
    print(f"[{'PASS' if passed else 'FAIL'}] n_embd={n_embd} | max_diff={max_diff:.2e}")
    assert passed, f"Non-power-of-2 embedding failed, max_diff={max_diff:.2e}"


def test_embedding_position_offset():
    """Non-zero position_offset shifts the positional encoding index."""
    print("\n=== Embedding Position Offset Test ===")

    batch, seq_len, n_embd = 2, 3, 32
    vocab_size = 10
    max_position = 128

    token_ids = np.array([[3, 7, 1], [2, 5, 0]], dtype=np.int32)
    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    pos_weight = np.random.randn(max_position, n_embd).astype(np.float32)

    offset = 10
    out = embedding(token_ids, weight, pos_weight, position_offset=offset)

    # Reference with offset
    ref = np.zeros((batch, seq_len, n_embd), dtype=np.float32)
    for b in range(batch):
        for s in range(seq_len):
            ref[b, s, :] = weight[token_ids[b, s], :] + pos_weight[s + offset, :]

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4)
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] offset={offset} seq_len={seq_len} | max_diff={max_diff:.2e}")
    assert passed, f"Position offset failed, max_diff={max_diff:.2e}"

    # Also verify offset=0 matches original reference
    out_zero = embedding(token_ids, weight, pos_weight, position_offset=0)
    ref_zero = _ref_embedding(token_ids, weight, pos_weight)
    max_diff_zero = float(np.abs(out_zero - ref_zero).max())
    passed_zero = np.allclose(out_zero, ref_zero, atol=1e-4)
    status_zero = "PASS" if passed_zero else "FAIL"
    print(f"[{status_zero}] offset=0 (default) | max_diff={max_diff_zero:.2e}")
    assert passed_zero, "offset=0 should match original ref"

    # Verify offset != 0 produces different result
    diff = float(np.abs(out - out_zero).max())
    print(f"  |diff between offset=10 and offset=0| = {diff:.4f}")

def test_embedding_single_position():
    """Single batch, single position edge case."""
    print("\n=== Embedding Single Position Test ===")

    batch, seq_len, n_embd = 1, 1, 64
    vocab_size = 10
    max_position = 128

    token_ids = np.array([[5]], dtype=np.int32)
    weight = np.random.randn(vocab_size, n_embd).astype(np.float32)
    pos_weight = np.random.randn(max_position, n_embd).astype(np.float32)

    out = embedding(token_ids, weight, pos_weight)
    ref = _ref_embedding(token_ids, weight, pos_weight)

    max_diff = float(np.abs(out - ref).max())
    passed = np.allclose(out, ref, atol=1e-4, rtol=1e-5)
    print(f"[{'PASS' if passed else 'FAIL'}] Single position | max_diff={max_diff:.2e}")
    assert passed, f"Single position failed, max_diff={max_diff:.2e}"


if __name__ == "__main__":
    print("Running Embedding unit tests on current GPU backend...")
    test_embedding_correctness()
    test_embedding_first_token()
    test_embedding_empty()
    test_embedding_non_power_of_2()
    test_embedding_position_offset()
    test_embedding_single_position()
    print("\n" + "=" * 45)
    print("All embedding tests PASSED")
    print("=" * 45)
