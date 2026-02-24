//! Pluggable fitness criteria for arena evaluation.
//!
//! Each criterion defines what "creature A beats creature B" means.
//! Different selection pressures shape diversity differently — testing
//! whether diversity-fitness tradeoffs are robust across criteria is
//! an interesting experimental variable (per Claudius's observation).

use serde::{Deserialize, Serialize};

/// A fitness criterion that can evaluate creature performance.
///
/// Implementations define how raw simulation data becomes a scalar score.
/// The arena doesn't care how scores are produced — it just compares them.
pub trait FitnessCriterion: Send + Sync {
    /// Human-readable name for this criterion.
    fn name(&self) -> &str;

    /// Compare two scores: returns positive if score_a wins, negative if score_b wins.
    /// Default implementation: simple subtraction.
    fn compare(&self, score_a: f32, score_b: f32) -> f32 {
        score_a - score_b
    }

    /// Convert a raw score difference to a win probability in [0, 1].
    /// Used for Elo updates. Default: logistic with scale 1.0.
    fn win_probability(&self, score_diff: f32) -> f32 {
        1.0 / (1.0 + (-score_diff).exp())
    }
}

/// Identifier for a fitness criterion, serializable for tournament records.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CriterionId {
    LocomotionDistance,
    EnergyEfficiency,
    ObstacleNavigation,
    PerturbationRobustness,
    Custom(String),
}

impl std::fmt::Display for CriterionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CriterionId::LocomotionDistance => write!(f, "locomotion_distance"),
            CriterionId::EnergyEfficiency => write!(f, "energy_efficiency"),
            CriterionId::ObstacleNavigation => write!(f, "obstacle_navigation"),
            CriterionId::PerturbationRobustness => write!(f, "perturbation_robustness"),
            CriterionId::Custom(name) => write!(f, "{}", name),
        }
    }
}

// ============================================================================
// Concrete Criteria
// ============================================================================

/// Locomotion distance: horizontal distance traveled per unit time.
/// The simplest and most natural criterion for virtual creatures.
pub struct LocomotionDistance;

impl FitnessCriterion for LocomotionDistance {
    fn name(&self) -> &str {
        "locomotion_distance"
    }
}

/// Energy efficiency: distance traveled per unit energy expended.
/// Favors creatures that move well without wasting energy on
/// flailing limbs or excessive joint torque.
pub struct EnergyEfficiency {
    /// How much to weight distance vs energy. Higher = more weight on distance.
    pub distance_weight: f32,
}

impl Default for EnergyEfficiency {
    fn default() -> Self {
        Self {
            distance_weight: 1.0,
        }
    }
}

impl FitnessCriterion for EnergyEfficiency {
    fn name(&self) -> &str {
        "energy_efficiency"
    }

    fn compare(&self, score_a: f32, score_b: f32) -> f32 {
        // Scores are already efficiency ratios (distance/energy)
        score_a - score_b
    }
}

/// Perturbation robustness: how well a creature maintains locomotion
/// when subjected to random forces. Measures the ratio of perturbed
/// performance to unperturbed performance.
pub struct PerturbationRobustness;

impl FitnessCriterion for PerturbationRobustness {
    fn name(&self) -> &str {
        "perturbation_robustness"
    }
}
