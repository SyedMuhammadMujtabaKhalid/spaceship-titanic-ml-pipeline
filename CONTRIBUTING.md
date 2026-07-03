# Contributing Guidelines

Thank you for your interest in contributing to this project!

## How to Contribute

1. **Fork** the repository and clone it locally.
2. Create a new **feature branch**: `git checkout -b feature/your-improvement`
3. Make your changes with clear, focused commits.
4. Ensure no data leakage is introduced in any preprocessing changes.
5. Run the full notebook top-to-bottom to verify it executes without errors.
6. Open a **Pull Request** with a clear description of what you changed and why.

## Code Standards

- Follow [PEP 8](https://pep8.org/) style guidelines.
- All preprocessing logic must remain inside the `preprocess_data` function.
- GroupKFold validation must not be replaced with random splits.
- Do not commit raw data files or output CSVs.
- Do not commit `catboost_info/` or `__pycache__/` directories.

## Scope

This repository focuses on:
- Leakage-free tabular ML pipelines
- Hyperparameter optimization with Optuna
- Ensemble techniques and threshold tuning
- Model explainability with SHAP

Out of scope: deep learning approaches, external data sources, or significant feature expansion without ablation evidence.

## Reporting Issues

Open a GitHub Issue describing:
- What you expected to happen
- What actually happened
- Steps to reproduce the problem
