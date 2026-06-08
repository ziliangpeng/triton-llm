#!/usr/bin/env python3
"""Get git diff of PR #28 (main vs feat/http-server)"""
import subprocess, sys

repo = "/Users/victor.peng/code/triton-llm"

# Get diff between main and feat/http-server
result = subprocess.run(
    ["git", "diff", "2fe285c2a9a00386a236841bc10d7b93c5e37b46", "f8c07fdc8d4b6834fadfa9cddf8c58dcc38a0d38", "--stat"],
    cwd=repo, capture_output=True, text=True
)
print("=== STAT ===")
print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

result = subprocess.run(
    ["git", "log", "--oneline", "2fe285c2a9a00386a236841bc10d7b93c5e37b46..f8c07fdc8d4b6834fadfa9cddf8c58dcc38a0d38"],
    cwd=repo, capture_output=True, text=True
)
print("=== COMMITS ===")
print(result.stdout)

result = subprocess.run(
    ["git", "diff", "2fe285c2a9a00386a236841bc10d7b93c5e37b46", "f8c07fdc8d4b6834fadfa9cddf8c58dcc38a0d38"],
    cwd=repo, capture_output=True, text=True
)
print("=== FULL DIFF ===")
print(result.stdout[:15000])
if len(result.stdout) > 15000:
    print(f"... (truncated, total {len(result.stdout)} chars)")
