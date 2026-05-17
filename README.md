# triton-llm

Pure Python + Triton (no PyTorch) implementation of GPT-2 inference.

## Goal
Build a minimal, from-scratch GPT-2 inference engine using only:
- Python standard library + numpy for orchestration
- NVIDIA Triton for all GPU kernels (GEMM, attention, layernorm, etc.)

## Current Status
- Repo initialized
- Working toward Phase 1: core model implementation

## Directory Structure
```
triton-llm/
├── gpt2_triton/          # main package
│   ├── model.py          # Python model definition
│   ├── kernels/          # Triton kernels
│   │   ├── gemm.py
│   │   ├── attention.py
│   │   └── ...
│   └── utils.py
├── tests/
├── scripts/
├── STATE.md
└── README.md
```

See STATE.md for live progress.
