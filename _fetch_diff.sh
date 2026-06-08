#!/bin/bash
cd /Users/victor.peng/code/triton-llm
echo "=== Current branch ==="
git branch -a 2>&1
echo "=== Current HEAD ==="
git log --oneline -1
echo "=== Local diff: feat/http-server vs main ==="
git diff main...feat/http-server --stat 2>&1
echo "=== FULL DIFF (local) ==="
git diff main...feat/http-server 2>&1
echo ""
echo "=== Fetching remote ==="
git fetch origin feat/http-server 2>&1 || echo "Fetch failed, using local ref"
echo "=== Remote diff: origin/main...origin/feat/http-server ==="
git diff origin/main...origin/feat/http-server --stat 2>&1
echo "---"
git diff origin/main...origin/feat/http-server 2>&1
