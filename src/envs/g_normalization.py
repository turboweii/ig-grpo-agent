"""
G-Normalization (L^0.5) for IG-GRPO

This module implements the G-Normalization technique to control gradient variance
in long-horizon trajectories. According to Lemma 2:

For a trajectory of length L, G-Norm (L^γ) keeps gradient variance bounded when γ ≤ 0.5.

We use γ = 0.5 (square root normalization) which provides:
- Bounded gradient variance for long trajectories
- Sufficient gradient signal for short trajectories
- Better balance than linear (1/L) normalization

Reference: ig-grpo-project-plan.md, Section 1.2 Lemma 2
"""
from __future__ import annotations

import math
from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class NormalizationResult:
    """Result of G-Normalization"""
    normalized_advantage: float      # Advantage after L^γ normalization
    raw_advantage: float             # Original advantage
    trajectory_length: int           # Length L
    gamma: float                     # Used γ value
    normalization_factor: float      # L^γ


class GNormalizer:
    """
    G-Normalization for IG-GRPO advantage computation.

    Formula:
        A_norm = A_raw / L^γ

    where:
        A_raw = raw advantage (group relative advantage)
        L = trajectory length (number of actions/steps)
        γ = normalization exponent (default 0.5)

    Properties:
    - When γ = 0: No normalization (raw advantage)
    - When γ = 0.5: Square root normalization (bounded variance)
    - When γ = 1.0: Linear normalization (standard GRPO)

    Theoretical guarantee (Lemma 2):
    - Gradient variance remains bounded when γ ≤ 0.5
    - For long trajectories (L → ∞), Var(∇θ J) remains O(1)
    """

    def __init__(
        self,
        gamma: float = 0.5,
        min_length: int = 1,
        max_length: Optional[int] = None,
        clip_value: Optional[float] = None,
    ):
        """
        Initialize G-Normalizer.

        Args:
            gamma: Normalization exponent (default 0.5 for sqrt)
            min_length: Minimum trajectory length for normalization
            max_length: Maximum trajectory length (caps normalization)
            clip_value: Clip normalized advantage to [-clip_value, clip_value]
        """
        self.gamma = gamma
        self.min_length = min_length
        self.max_length = max_length
        self.clip_value = clip_value

        # Statistics
        self.total_normalizations = 0
        self.accumulated_raw_adv = 0.0
        self.accumulated_norm_adv = 0.0

    def compute_normalization_factor(self, length: int) -> float:
        """
        Compute L^γ normalization factor.

        Args:
            length: Trajectory length L

        Returns:
            Normalization factor L^γ
        """
        # Clamp length to prevent division issues
        effective_length = max(self.min_length, length)
        if self.max_length is not None:
            effective_length = min(effective_length, self.max_length)

        return effective_length ** self.gamma

    def normalize(
        self,
        raw_advantage: float,
        trajectory_length: int,
    ) -> NormalizationResult:
        """
        Apply G-Normalization to raw advantage.

        Args:
            raw_advantage: Raw group relative advantage
            trajectory_length: Length of trajectory L

        Returns:
            NormalizationResult with normalized advantage and metadata
        """
        norm_factor = self.compute_normalization_factor(trajectory_length)
        normalized = raw_advantage / norm_factor

        # Optional clipping
        if self.clip_value is not None:
            normalized = max(-self.clip_value, min(self.clip_value, normalized))

        # Update statistics
        self.total_normalizations += 1
        self.accumulated_raw_adv += abs(raw_advantage)
        self.accumulated_norm_adv += abs(normalized)

        return NormalizationResult(
            normalized_advantage=normalized,
            raw_advantage=raw_advantage,
            trajectory_length=trajectory_length,
            gamma=self.gamma,
            normalization_factor=norm_factor,
        )

    def normalize_batch(
        self,
        raw_advantages: List[float],
        trajectory_lengths: List[int],
    ) -> List[NormalizationResult]:
        """
        Apply G-Normalization to a batch of advantages.

        Args:
            raw_advantages: List of raw advantages
            trajectory_lengths: Corresponding trajectory lengths

        Returns:
            List of NormalizationResult objects
        """
        return [
            self.normalize(adv, length)
            for adv, length in zip(raw_advantages, trajectory_lengths)
        ]

    def compute_advantage_with_jig(
        self,
        raw_advantage: float,
        trajectory_length: int,
        jig_reward: float,
        outcome_reward: float,
        curriculum_alpha: float = 0.3,
    ) -> Dict[str, float]:
        """
        Compute final advantage combining outcome + JIG with G-Normalization.

        Formula:
            A_raw = outcome + α * JIG
            A_final = A_raw / L^γ

        Args:
            raw_advantage: Raw group relative advantage
            trajectory_length: Length of trajectory L
            jig_reward: JIG (Joint Information Gain) reward
            outcome_reward: Binary outcome reward (0 or 1)
            curriculum_alpha: Curriculum learning weight for JIG

        Returns:
            Dict with normalized_advantage, components, and metadata
        """
        # Total reward before normalization
        total_reward = outcome_reward + curriculum_alpha * jig_reward

        # Apply G-Normalization
        result = self.normalize(raw_advantage, trajectory_length)

        return {
            "normalized_advantage": result.normalized_advantage,
            "raw_advantage": raw_advantage,
            "total_reward": total_reward,
            "outcome_reward": outcome_reward,
            "jig_reward": jig_reward,
            "trajectory_length": trajectory_length,
            "normalization_factor": result.normalization_factor,
            "gamma": self.gamma,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get normalization statistics"""
        avg_raw = (
            self.accumulated_raw_adv / self.total_normalizations
            if self.total_normalizations > 0
            else 0.0
        )
        avg_norm = (
            self.accumulated_norm_adv / self.total_normalizations
            if self.total_normalizations > 0
            else 0.0
        )

        return {
            "gamma": self.gamma,
            "total_normalizations": self.total_normalizations,
            "avg_raw_advantage": avg_raw,
            "avg_normalized_advantage": avg_norm,
            "compression_ratio": avg_norm / avg_raw if avg_raw > 0 else 0.0,
        }

    def reset_stats(self):
        """Reset statistics"""
        self.total_normalizations = 0
        self.accumulated_raw_adv = 0.0
        self.accumulated_norm_adv = 0.0


class AdaptiveGNormalizer(GNormalizer):
    """
    Adaptive G-Normalizer that adjusts gamma based on training progress.

    Uses curriculum learning to gradually increase gamma:
    - Early training: Lower gamma (less normalization) = stronger exploration signal
    - Late training: Higher gamma (more normalization) = stable convergence

    Formula:
        γ(t) = γ_min + (γ_max - γ_min) * (t / T)^β

    where:
        t = current step
        T = total steps
        β = curriculum exponent
    """

    def __init__(
        self,
        gamma_min: float = 0.3,
        gamma_max: float = 0.7,
        total_steps: int = 300,
        curriculum_beta: float = 1.0,
        **kwargs,
    ):
        super().__init__(gamma=gamma_min, **kwargs)
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.total_steps = total_steps
        self.curriculum_beta = curriculum_beta
        self.current_step = 0

    def get_gamma(self, step: Optional[int] = None) -> float:
        """Get current gamma based on curriculum"""
        if step is None:
            step = self.current_step
        else:
            self.current_step = step

        if step >= self.total_steps:
            return self.gamma_max

        progress = step / self.total_steps
        gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * (progress ** self.curriculum_beta)
        self.gamma = gamma
        return gamma

    def normalize(self, raw_advantage: float, trajectory_length: int) -> NormalizationResult:
        """Normalize with current curriculum gamma"""
        self.get_gamma()  # Update gamma based on current step
        return super().normalize(raw_advantage, trajectory_length)

    def step(self):
        """Advance curriculum step"""
        self.current_step += 1

    def reset(self):
        """Reset curriculum"""
        self.current_step = 0
        self.gamma = self.gamma_min


# Global instance for use in training
_global_normalizer: Optional[GNormalizer] = None


def get_global_normalizer() -> GNormalizer:
    """Get global G-Normalizer instance"""
    global _global_normalizer
    if _global_normalizer is None:
        _global_normalizer = GNormalizer(gamma=0.5)
    return _global_normalizer


def set_global_normalizer(normalizer: GNormalizer):
    """Set global G-Normalizer instance"""
    global _global_normalizer
    _global_normalizer = normalizer
