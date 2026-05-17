# AGENTS.md

This document defines the guidelines for contributing to this repository, especially for writing PR descriptions and handling tests.

## 1. PR Description Guidelines

Every PR should contain the following sections (in order):

### Summary
- Clearly state what this PR does.
- Mention which step of the overall plan this PR corresponds to (link to the tracking Issue if applicable).

### Goal
- Explain the purpose of this change in the context of the larger project.

### Implementation
- Briefly describe how the feature or fix was implemented.
- Keep it high-level; avoid exposing internal environment details.

### Testing
- Describe how the changes were tested.
- Include the test command used (generalized, without exposing internal paths or cluster names).
- Paste the test output (cleaned of sensitive information).

### Scope
- Clearly define what is included and what is not included in this PR.
- This helps reviewers understand the boundaries of the change.

## 2. Writing Style

- Always write PR descriptions, commit messages, and code comments in **English**.
- Keep the tone professional and concise.
- Do not expose internal environment details such as:
  - Cluster names (e.g., gcp5)
  - Specific node names or Slurm partitions
  - Absolute file paths on internal machines
  - Usernames or home directory structures

## 3. Test Output in PRs

When including test results in a PR:

- Use a clean, generalized command format.
- Only show relevant output.
- Remove any lines that reveal internal infrastructure.

Example format:

```bash
# Run tests on a GPU node
python -m pytest tests/test_gpu_allocator.py -q
```

**Test Output:**
```
All allocator tests passed!
```

## 4. Branch and PR Naming

- Use the naming pattern: `feat/step-N-<short-description>`
- Each PR should correspond to one step in the tracking Issue.
- Reference the Issue number in the PR description.

## 5. Future Updates

This file should be updated whenever new conventions or best practices are established during development.