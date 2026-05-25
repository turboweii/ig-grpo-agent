"""
IG-GRPO: Information Gain Guided GRPO for Long-Horizon Agents

This package implements the IG-GRPO algorithm for training multi-tool agents
on long-horizon tasks.

Core Components:
- JIG (Joint Information Gain): Exploration reward based on state-tool novelty
- G-Normalization: L^0.5 advantage normalization for bounded gradient variance
- Hierarchical Coverage Tracker: Three-layer Bloom Filter for state tracking
- Sustained Exploration Bonus: Prevents premature convergence

Reference: ig-grpo-project-plan.md
"""

__version__ = "1.0.0"
