# State — triton-llm GPT-2 pure Triton inference

**Snapshot**: 2026-05-16  
**Current Phase**: Phase 1 — Component implementation (LayerNorm done)  
**Next**: Implement GELU / GEMM kernels + tests

## Completed Components
- LayerNorm kernel + correctness test (max diff < 1e-3 vs torch reference)

## Open PRs
None

## Live resources
- Local: ~/code/triton-llm
- GitHub: https://github.com/ziliangpeng/triton-llm

## Resume commands
cd ~/code/triton-llm
python -m pytest gpt2_triton/tests/test_layernorm.py   # (will need CUDA)

## Completed Components (updated)
- LayerNorm kernel + test
- GELU kernel + test

## Completed Components (updated 2026-05-16)
- LayerNorm kernel + test
- GELU kernel + test
- Basic GEMM kernel + test (max diff < 1e-2)
