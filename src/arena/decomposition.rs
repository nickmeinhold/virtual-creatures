//! Transitive-cyclic decomposition of tournament payoff matrices.
//!
//! Based on Balduzzi et al. (2019) "Re-evaluating Evaluation":
//! - The **transitive** component captures what a linear Elo ranking explains:
//!   creature A > B > C with consistent score differences.
//! - The **cyclic** component captures intransitive dominance (rock-paper-scissors):
//!   creature A beats B, B beats C, but C beats A.
//!
//! Key insight from Claudius: the cyclic component isn't noise to filter —
//! it's diagnostic. Strong cycling means creatures have specialized into
//! niches that exploit each other's weaknesses. That's one of the most
//! interesting potential findings for our paper.

use serde::{Deserialize, Serialize};

/// Result of the transitive-cyclic decomposition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionResult {
    /// Transitive ratings (one per creature, in input order).
    /// Higher = stronger in the transitive component.
    pub transitive_ratings: Vec<f64>,

    /// The transitive component of the payoff matrix.
    /// T[i][j] = r_i - r_j where r are the transitive ratings.
    pub transitive_matrix: Vec<Vec<f64>>,

    /// The cyclic component of the payoff matrix.
    /// C = A - T where A is the antisymmetric payoff matrix.
    pub cyclic_matrix: Vec<Vec<f64>>,

    /// Cycle strength: ||C||_F / ||A||_F (Frobenius norm ratio).
    /// 0 = fully transitive, 1 = fully cyclic.
    pub cycle_strength: f64,

    /// Per-creature cycle participation: how much each creature
    /// is involved in intransitive cycles. Higher = more "niche" specialist.
    pub cycle_participation: Vec<f64>,

    /// Dominant cycles detected (sets of creature indices that form cycles).
    pub detected_cycles: Vec<Vec<usize>>,
}

/// Decompose a tournament payoff matrix into transitive and cyclic components.
pub struct TransitiveCyclicDecomposition;

impl TransitiveCyclicDecomposition {
    /// Decompose a payoff matrix.
    ///
    /// `payoff` is an NxN matrix where payoff[i][j] is the score of creature i
    /// against creature j, in [0, 1]. Self-play entries (diagonal) are ignored.
    pub fn decompose(payoff: &[Vec<f64>]) -> DecompositionResult {
        let n = payoff.len();
        assert!(n >= 2, "Need at least 2 creatures for decomposition");

        // Step 1: Build the antisymmetric payoff matrix A
        // A[i][j] = payoff[i][j] - payoff[j][i] = payoff[i][j] - (1 - payoff[i][j]) = 2*payoff[i][j] - 1
        // This gives A[i][j] in [-1, 1] where positive = i beats j.
        let antisymmetric = build_antisymmetric(payoff);

        // Step 2: Find transitive ratings via least squares.
        // We want to find ratings r such that r_i - r_j ≈ A[i][j] for all pairs.
        // This is a least-squares problem: minimize Σ (A[i][j] - (r_i - r_j))^2
        // subject to Σ r_i = 0 (center the ratings).
        let transitive_ratings = fit_transitive_ratings(&antisymmetric);

        // Step 3: Build the transitive matrix T[i][j] = r_i - r_j
        let transitive_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| transitive_ratings[i] - transitive_ratings[j])
                    .collect()
            })
            .collect();

        // Step 4: Cyclic component C = A - T
        let cyclic_matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| antisymmetric[i][j] - transitive_matrix[i][j])
                    .collect()
            })
            .collect();

        // Step 5: Compute cycle strength
        let norm_a = frobenius_norm(&antisymmetric);
        let norm_c = frobenius_norm(&cyclic_matrix);
        let cycle_strength = if norm_a > 1e-10 {
            norm_c / norm_a
        } else {
            0.0 // All draws = no signal
        };

        // Step 6: Per-creature cycle participation
        let cycle_participation = compute_cycle_participation(&cyclic_matrix);

        // Step 7: Detect dominant cycles (3-cycles where all three edges agree)
        let detected_cycles = detect_cycles(&cyclic_matrix);

        DecompositionResult {
            transitive_ratings,
            transitive_matrix,
            cyclic_matrix,
            cycle_strength,
            cycle_participation,
            detected_cycles,
        }
    }
}

/// Build the antisymmetric payoff matrix.
fn build_antisymmetric(payoff: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = payoff.len();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        0.0
                    } else {
                        // A[i][j] = p[i][j] - p[j][i]
                        // Since p[j][i] should be 1 - p[i][j] for zero-sum, this is 2*p[i][j] - 1
                        // But use the actual data in case there's noise
                        payoff[i][j] - payoff[j][i]
                    }
                })
                .collect()
        })
        .collect()
}

/// Fit transitive ratings via least squares.
///
/// Finds r_1, ..., r_n that minimize Σ_{i<j} (A[i][j] - (r_i - r_j))^2
/// subject to Σ r_i = 0.
///
/// This has a closed-form solution: r_i = (1/n) * Σ_j A[i][j]
/// (the mean of each row of the antisymmetric matrix).
fn fit_transitive_ratings(antisymmetric: &[Vec<f64>]) -> Vec<f64> {
    let n = antisymmetric.len();
    let n_f = n as f64;

    // r_i = (1/n) * Σ_j A[i][j]
    let ratings: Vec<f64> = (0..n)
        .map(|i| antisymmetric[i].iter().sum::<f64>() / n_f)
        .collect();

    // Center: subtract mean (should already be ~0 for antisymmetric, but be safe)
    let mean: f64 = ratings.iter().sum::<f64>() / n_f;
    ratings.iter().map(|r| r - mean).collect()
}

/// Frobenius norm of a matrix (sqrt of sum of squared elements).
fn frobenius_norm(matrix: &[Vec<f64>]) -> f64 {
    let sum_sq: f64 = matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| x * x)
        .sum();
    sum_sq.sqrt()
}

/// Compute per-creature cycle participation.
///
/// For each creature i, this is the L2 norm of the i-th row of the
/// cyclic matrix, normalized by the maximum across all creatures.
fn compute_cycle_participation(cyclic: &[Vec<f64>]) -> Vec<f64> {
    let row_norms: Vec<f64> = cyclic
        .iter()
        .map(|row| {
            let sum_sq: f64 = row.iter().map(|&x| x * x).sum();
            sum_sq.sqrt()
        })
        .collect();

    let max_norm = row_norms
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max);

    if max_norm > 1e-10 {
        row_norms.iter().map(|n| n / max_norm).collect()
    } else {
        vec![0.0; cyclic.len()]
    }
}

/// Detect 3-cycles in the cyclic matrix.
///
/// A 3-cycle (i, j, k) exists when:
/// - C[i][j] > 0 (i beats j in the cyclic component)
/// - C[j][k] > 0 (j beats k)
/// - C[k][i] > 0 (k beats i)
///
/// Returns cycles sorted by strength (product of edge weights).
fn detect_cycles(cyclic: &[Vec<f64>]) -> Vec<Vec<usize>> {
    let n = cyclic.len();
    let threshold = 0.05; // Minimum edge weight to count as a cycle edge

    let mut cycles: Vec<(Vec<usize>, f64)> = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                // Check both directions: i->j->k->i and i->k->j->i
                let forward = cyclic[i][j].min(cyclic[j][k]).min(cyclic[k][i]);
                let backward = cyclic[i][k].min(cyclic[k][j]).min(cyclic[j][i]);

                if forward > threshold {
                    let strength = cyclic[i][j] * cyclic[j][k] * cyclic[k][i];
                    cycles.push((vec![i, j, k], strength));
                } else if backward > threshold {
                    let strength = cyclic[i][k] * cyclic[k][j] * cyclic[j][i];
                    cycles.push((vec![i, k, j], strength));
                }
            }
        }
    }

    // Sort by strength (descending)
    cycles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cycles.into_iter().map(|(c, _)| c).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fully_transitive() {
        // Mathematically transitive payoff: margins proportional to skill gaps.
        // r = [0.4, 0, -0.4], p[i][j] = 0.5 + (r_i - r_j) / 2
        // This ensures the linear decomposition can perfectly reconstruct the matrix.
        let payoff = vec![
            vec![0.5, 0.7, 0.9], // creature 0: skill 0.4
            vec![0.3, 0.5, 0.7], // creature 1: skill 0.0
            vec![0.1, 0.3, 0.5], // creature 2: skill -0.4
        ];

        let result = TransitiveCyclicDecomposition::decompose(&payoff);

        // Should be fully transitive (cycle strength ≈ 0)
        assert!(
            result.cycle_strength < 0.01,
            "Mathematically transitive payoff should have near-zero cycle strength: {}",
            result.cycle_strength
        );

        // Transitive ratings should rank 0 > 1 > 2
        assert!(result.transitive_ratings[0] > result.transitive_ratings[1]);
        assert!(result.transitive_ratings[1] > result.transitive_ratings[2]);
    }

    #[test]
    fn test_rock_paper_scissors() {
        // Perfect rock-paper-scissors: 0 beats 1, 1 beats 2, 2 beats 0
        let payoff = vec![
            vec![0.5, 1.0, 0.0], // creature 0 beats 1, loses to 2
            vec![0.0, 0.5, 1.0], // creature 1 beats 2, loses to 0
            vec![1.0, 0.0, 0.5], // creature 2 beats 0, loses to 1
        ];

        let result = TransitiveCyclicDecomposition::decompose(&payoff);

        // Should be highly cyclic
        assert!(
            result.cycle_strength > 0.9,
            "Rock-paper-scissors should have high cycle strength: {}",
            result.cycle_strength
        );

        // Transitive ratings should be approximately equal
        let max_diff = result.transitive_ratings.iter()
            .flat_map(|&a| result.transitive_ratings.iter().map(move |&b| (a - b).abs()))
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 0.1,
            "RPS ratings should be equal, max diff: {}",
            max_diff
        );

        // Should detect at least one cycle
        assert!(
            !result.detected_cycles.is_empty(),
            "Should detect the RPS cycle"
        );

        // All creatures should participate equally in cycling
        let max_participation = result.cycle_participation.iter().cloned().fold(0.0_f64, f64::max);
        let min_participation = result.cycle_participation.iter().cloned().fold(1.0_f64, f64::min);
        assert!(
            (max_participation - min_participation).abs() < 0.1,
            "All creatures should participate equally in RPS: {:?}",
            result.cycle_participation
        );
    }

    #[test]
    fn test_mixed_transitive_and_cyclic() {
        // Mostly transitive but with some cycling
        let payoff = vec![
            vec![0.5, 0.8, 0.3, 0.9], // creature 0: strong overall but loses to 2
            vec![0.2, 0.5, 0.7, 0.8], // creature 1: medium, beats 2 and 3
            vec![0.7, 0.3, 0.5, 0.6], // creature 2: beats 0 but loses to 1
            vec![0.1, 0.2, 0.4, 0.5], // creature 3: weakest
        ];

        let result = TransitiveCyclicDecomposition::decompose(&payoff);

        // Should have moderate cycle strength (not fully transitive or cyclic)
        assert!(
            result.cycle_strength > 0.1 && result.cycle_strength < 0.9,
            "Mixed payoff should have moderate cycle strength: {}",
            result.cycle_strength
        );

        // Creature 3 should be ranked lowest transitively
        let min_rating = result.transitive_ratings.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(
            (result.transitive_ratings[3] - min_rating).abs() < 0.001,
            "Creature 3 should have lowest transitive rating"
        );
    }

    #[test]
    fn test_all_draws() {
        let payoff = vec![
            vec![0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5],
        ];

        let result = TransitiveCyclicDecomposition::decompose(&payoff);

        // No signal at all
        assert!(
            result.cycle_strength < 0.01,
            "All draws should have zero cycle strength: {}",
            result.cycle_strength
        );
    }

    #[test]
    fn test_antisymmetric_property() {
        let payoff = vec![
            vec![0.5, 0.7, 0.3],
            vec![0.3, 0.5, 0.8],
            vec![0.7, 0.2, 0.5],
        ];

        let a = build_antisymmetric(&payoff);

        // A should be antisymmetric: A[i][j] = -A[j][i]
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (a[i][j] + a[j][i]).abs() < 1e-10,
                    "A[{}][{}] + A[{}][{}] = {} + {} ≠ 0",
                    i, j, j, i, a[i][j], a[j][i]
                );
            }
        }
    }

    #[test]
    fn test_cyclic_matrix_antisymmetric() {
        let payoff = vec![
            vec![0.5, 0.7, 0.3],
            vec![0.3, 0.5, 0.8],
            vec![0.7, 0.2, 0.5],
        ];

        let result = TransitiveCyclicDecomposition::decompose(&payoff);

        // Cyclic matrix should also be antisymmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (result.cyclic_matrix[i][j] + result.cyclic_matrix[j][i]).abs() < 1e-10,
                    "C[{}][{}] + C[{}][{}] = {} + {} ≠ 0",
                    i, j, j, i,
                    result.cyclic_matrix[i][j], result.cyclic_matrix[j][i]
                );
            }
        }
    }
}
