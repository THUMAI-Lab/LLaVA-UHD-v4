"""Minimal base model package for Stage 4 SFT training.

Keep this package initializer side-effect free. The training entrypoint imports
the concrete modules it needs directly, and importing retired VLU variants here
would force the minimal repo to keep their unused dependencies.
"""
