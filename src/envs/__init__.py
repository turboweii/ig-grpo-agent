"""
IG-GRPO Environment Components

This package contains all environment-related components for IG-GRPO training:
- JIG (Joint Information Gain) components
- G-Normalization for advantage computation
- Tau-bench environment wrappers and interactions
"""

from .jig_components import (
    HierarchicalCoverageTracker,
    JointInformationGain,
    SustainedExplorationBonus,
    CurriculumScheduler,
)

from .g_normalization import (
    GNormalizer,
    AdaptiveGNormalizer,
    get_global_normalizer,
    set_global_normalizer,
)

from .async_entropy_estimator import (
    AsyncEntropyEstimator,
    GlobalEntropyEstimator,
)

from .tau_bench_interaction_ig import (
    TauBenchInteractionIG,
)

__all__ = [
    # JIG Components
    "HierarchicalCoverageTracker",
    "JointInformationGain",
    "SustainedExplorationBonus",
    "CurriculumScheduler",
    # G-Normalization
    "GNormalizer",
    "AdaptiveGNormalizer",
    "get_global_normalizer",
    "set_global_normalizer",
    # Async Entropy
    "AsyncEntropyEstimator",
    "GlobalEntropyEstimator",
    # Interaction
    "TauBenchInteractionIG",
]
