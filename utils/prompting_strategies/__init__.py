"""
Prompt strategy package housing experiment-specific prompt definitions.

Experiment A currently includes concrete zero-shot, few-shot, and chain-of-thought
factories. Experiments B and C are scaffolded for future prompt variants.
"""

from . import experiment_a, experiment_b, experiment_c

__all__ = [
    "experiment_a",
    "experiment_b",
    "experiment_c",
]
