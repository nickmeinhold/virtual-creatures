//! Tournament scheduling and execution.
//!
//! The tournament runner is substrate-agnostic: it generates match pairings
//! and processes results, but delegates actual evaluation to an `Evaluator`
//! trait that the physics sim implements.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::criteria::CriterionId;
use super::decomposition::{DecompositionResult, TransitiveCyclicDecomposition};
use super::ratings::{EloConfig, EloSystem, MatchRecord};

/// Configuration for a tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentConfig {
    /// Elo rating configuration.
    pub elo_config: EloConfig,
    /// Number of rounds in round-robin (each pair plays this many times).
    pub rounds_per_pair: usize,
    /// Fitness criterion for this tournament.
    pub criterion: CriterionId,
}

impl Default for TournamentConfig {
    fn default() -> Self {
        Self {
            elo_config: EloConfig::default(),
            rounds_per_pair: 3,
            criterion: CriterionId::LocomotionDistance,
        }
    }
}

/// A scheduled match between two creatures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledMatch {
    /// Match index in the tournament.
    pub index: usize,
    /// First creature ID.
    pub creature_a: u64,
    /// Second creature ID.
    pub creature_b: u64,
    /// Round number (0-indexed).
    pub round: usize,
}

/// Complete results of a tournament.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TournamentResults {
    /// Tournament configuration used.
    pub config: TournamentConfig,
    /// All match records.
    pub matches: Vec<MatchRecord>,
    /// Final Elo ratings.
    pub elo_system: EloSystem,
    /// Transitive-cyclic decomposition.
    pub decomposition: Option<DecompositionResult>,
    /// Creature IDs that participated.
    pub participants: Vec<u64>,
}

impl TournamentResults {
    /// Get the leaderboard (sorted by Elo rating, descending).
    pub fn leaderboard(&self) -> Vec<(u64, f64)> {
        self.elo_system
            .leaderboard()
            .into_iter()
            .map(|(id, r)| (id, r.rating))
            .collect()
    }

    /// Get cycle strength (0 = fully transitive, 1 = fully cyclic).
    pub fn cycle_strength(&self) -> Option<f64> {
        self.decomposition.as_ref().map(|d| d.cycle_strength)
    }
}

/// A tournament: generates pairings, collects results, computes ratings.
pub struct Tournament {
    config: TournamentConfig,
    elo: EloSystem,
    participants: Vec<u64>,
    schedule: Vec<ScheduledMatch>,
    results: Vec<MatchRecord>,
    current_match: usize,
}

impl Tournament {
    /// Create a new tournament with the given participants.
    pub fn new(config: TournamentConfig, participant_ids: Vec<u64>) -> Self {
        let mut elo = EloSystem::new(config.elo_config.clone());
        elo.register_all(&participant_ids);

        let schedule = generate_round_robin(&participant_ids, config.rounds_per_pair);

        Self {
            config,
            elo,
            participants: participant_ids,
            schedule,
            results: Vec::new(),
            current_match: 0,
        }
    }

    /// Get the next match to play, or None if tournament is complete.
    pub fn next_match(&self) -> Option<&ScheduledMatch> {
        self.schedule.get(self.current_match)
    }

    /// Total number of matches in the tournament.
    pub fn total_matches(&self) -> usize {
        self.schedule.len()
    }

    /// Number of matches completed so far.
    pub fn matches_completed(&self) -> usize {
        self.current_match
    }

    /// Is the tournament finished?
    pub fn is_complete(&self) -> bool {
        self.current_match >= self.schedule.len()
    }

    /// Record the result of the current match.
    ///
    /// `score_a` and `score_b` are the raw performance scores for each creature.
    /// The tournament converts these to a [0, 1] outcome for Elo.
    pub fn record_result(&mut self, score_a: f32, score_b: f32) {
        if self.is_complete() {
            return;
        }

        let scheduled = &self.schedule[self.current_match];
        let creature_a = scheduled.creature_a;
        let creature_b = scheduled.creature_b;

        // Convert raw scores to [0, 1] outcome for Elo
        let outcome_a = scores_to_outcome(score_a, score_b);

        // Record in Elo system
        self.elo.record_match(creature_a, creature_b, outcome_a);

        // Store the match record
        self.results.push(MatchRecord {
            creature_a,
            creature_b,
            score_a: outcome_a,
            criterion: self.config.criterion.to_string(),
        });

        self.current_match += 1;
    }

    /// Finalize the tournament and compute decomposition.
    pub fn finalize(self) -> TournamentResults {
        let (ids, payoff_matrix) = self.elo.payoff_matrix(&self.results);

        let decomposition = if ids.len() >= 3 {
            Some(TransitiveCyclicDecomposition::decompose(&payoff_matrix))
        } else {
            None
        };

        TournamentResults {
            config: self.config,
            matches: self.results,
            elo_system: self.elo,
            decomposition,
            participants: self.participants,
        }
    }

    /// Run a tournament from pre-computed scores (no simulation needed).
    ///
    /// `scores` maps creature_id -> performance score.
    /// All pairs are compared, and the higher score wins the match.
    pub fn run_from_scores(config: TournamentConfig, scores: &HashMap<u64, f32>) -> TournamentResults {
        let participant_ids: Vec<u64> = scores.keys().copied().collect();
        let mut tournament = Tournament::new(config, participant_ids);

        while let Some(scheduled) = tournament.next_match() {
            let a = scheduled.creature_a;
            let b = scheduled.creature_b;
            let score_a = scores.get(&a).copied().unwrap_or(0.0);
            let score_b = scores.get(&b).copied().unwrap_or(0.0);
            tournament.record_result(score_a, score_b);
        }

        tournament.finalize()
    }
}

/// Generate round-robin pairings.
///
/// Each pair of creatures plays `rounds_per_pair` matches.
/// Matches are interleaved across rounds to avoid recency bias in Elo.
fn generate_round_robin(participants: &[u64], rounds_per_pair: usize) -> Vec<ScheduledMatch> {
    let mut schedule = Vec::new();
    let mut index = 0;

    for round in 0..rounds_per_pair {
        for i in 0..participants.len() {
            for j in (i + 1)..participants.len() {
                // Alternate who is "creature_a" across rounds
                let (a, b) = if round % 2 == 0 {
                    (participants[i], participants[j])
                } else {
                    (participants[j], participants[i])
                };

                schedule.push(ScheduledMatch {
                    index,
                    creature_a: a,
                    creature_b: b,
                    round,
                });
                index += 1;
            }
        }
    }

    schedule
}

/// Convert raw performance scores to a [0, 1] outcome for Elo.
///
/// Uses a logistic function so that small differences produce outcomes
/// near 0.5 (close to a draw) and large differences saturate toward 0 or 1.
fn scores_to_outcome(score_a: f32, score_b: f32) -> f64 {
    let diff = (score_a - score_b) as f64;

    // If both scores are zero (or very close), it's a draw
    if diff.abs() < 1e-6 {
        return 0.5;
    }

    // Normalize the difference by the mean of both scores to make the
    // logistic scale-invariant (a 10% advantage means the same whether
    // scores are 1.0 vs 1.1 or 100.0 vs 110.0)
    let mean = ((score_a + score_b) as f64 / 2.0).max(1e-6);
    let normalized_diff = diff / mean;

    // Logistic with steepness 5.0: a ~20% advantage ≈ 0.73 win probability
    1.0 / (1.0 + (-5.0 * normalized_diff).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_round_robin_pairing_count() {
        let participants = vec![1, 2, 3, 4];
        let schedule = generate_round_robin(&participants, 3);
        // 4 choose 2 = 6 pairs * 3 rounds = 18 matches
        assert_eq!(schedule.len(), 18);
    }

    #[test]
    fn test_scores_to_outcome_equal() {
        let outcome = scores_to_outcome(1.0, 1.0);
        assert!((outcome - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_scores_to_outcome_win() {
        let outcome = scores_to_outcome(2.0, 1.0);
        assert!(outcome > 0.5, "Higher score should win: {}", outcome);
    }

    #[test]
    fn test_scores_to_outcome_loss() {
        let outcome = scores_to_outcome(1.0, 2.0);
        assert!(outcome < 0.5, "Lower score should lose: {}", outcome);
    }

    #[test]
    fn test_scores_to_outcome_symmetric() {
        let outcome_a = scores_to_outcome(2.0, 1.0);
        let outcome_b = scores_to_outcome(1.0, 2.0);
        assert!((outcome_a + outcome_b - 1.0).abs() < 0.001, "Outcomes should sum to 1");
    }

    #[test]
    fn test_tournament_from_scores() {
        let mut scores = HashMap::new();
        scores.insert(1, 10.0);
        scores.insert(2, 5.0);
        scores.insert(3, 1.0);

        let config = TournamentConfig::default();
        let results = Tournament::run_from_scores(config, &scores);

        let board = results.leaderboard();
        assert_eq!(board[0].0, 1, "Creature 1 (score=10) should be ranked #1");
        assert_eq!(board[2].0, 3, "Creature 3 (score=1) should be ranked #3");
    }

    #[test]
    fn test_tournament_has_decomposition() {
        let mut scores = HashMap::new();
        scores.insert(1, 10.0);
        scores.insert(2, 5.0);
        scores.insert(3, 1.0);

        let config = TournamentConfig::default();
        let results = Tournament::run_from_scores(config, &scores);

        assert!(results.decomposition.is_some(), "Should have decomposition with 3+ creatures");
        let cycle = results.cycle_strength().unwrap();
        // With a clear transitive ordering (10 > 5 > 1), cycle strength should be moderate
        // (the logistic score-to-outcome conversion introduces non-linearity that
        // the linear decomposition can't perfectly capture)
        assert!(cycle < 0.5, "Clear transitive ordering should have bounded cycle strength: {}", cycle);
    }

    #[test]
    fn test_tournament_step_by_step() {
        let config = TournamentConfig {
            rounds_per_pair: 1,
            ..Default::default()
        };
        let mut tournament = Tournament::new(config, vec![1, 2, 3]);

        assert_eq!(tournament.total_matches(), 3); // 3 choose 2
        assert!(!tournament.is_complete());

        // Play all matches
        while tournament.next_match().is_some() {
            tournament.record_result(1.0, 0.5); // A always wins
        }

        assert!(tournament.is_complete());
        let results = tournament.finalize();
        assert_eq!(results.matches.len(), 3);
    }
}
