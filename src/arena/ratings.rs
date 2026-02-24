//! Elo rating system for creature tournaments.
//!
//! Standard Elo with configurable K-factor. Used as the "transitive"
//! component of creature rankings — the part that a linear ordering
//! can explain. The residual (what Elo can't explain) is where
//! intransitive dynamics live.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Configuration for the Elo rating system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloConfig {
    /// Initial rating for new creatures.
    pub initial_rating: f64,
    /// K-factor: how much each match moves ratings.
    /// Higher = more volatile, lower = more stable.
    pub k_factor: f64,
    /// Rating difference that corresponds to ~76% expected win rate.
    /// Standard chess uses 400.
    pub scale: f64,
}

impl Default for EloConfig {
    fn default() -> Self {
        Self {
            initial_rating: 1500.0,
            k_factor: 32.0,
            scale: 400.0,
        }
    }
}

/// A creature's Elo rating with history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatureRating {
    /// Current Elo rating.
    pub rating: f64,
    /// Number of matches played.
    pub matches_played: usize,
    /// Win/loss/draw record.
    pub wins: usize,
    pub losses: usize,
    pub draws: usize,
    /// Rating history (after each match).
    pub history: Vec<f64>,
}

impl CreatureRating {
    pub fn new(initial_rating: f64) -> Self {
        Self {
            rating: initial_rating,
            matches_played: 0,
            wins: 0,
            losses: 0,
            draws: 0,
            history: vec![initial_rating],
        }
    }

    pub fn win_rate(&self) -> f64 {
        if self.matches_played == 0 {
            return 0.0;
        }
        (self.wins as f64 + 0.5 * self.draws as f64) / self.matches_played as f64
    }
}

/// The Elo rating system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloSystem {
    pub config: EloConfig,
    pub ratings: HashMap<u64, CreatureRating>,
}

impl EloSystem {
    pub fn new(config: EloConfig) -> Self {
        Self {
            config,
            ratings: HashMap::new(),
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(EloConfig::default())
    }

    /// Register a creature. No-op if already registered.
    pub fn register(&mut self, creature_id: u64) {
        self.ratings
            .entry(creature_id)
            .or_insert_with(|| CreatureRating::new(self.config.initial_rating));
    }

    /// Register multiple creatures at once.
    pub fn register_all(&mut self, creature_ids: &[u64]) {
        for &id in creature_ids {
            self.register(id);
        }
    }

    /// Expected score for creature A against creature B.
    /// Returns a value in [0, 1] where 1 = certain win for A.
    pub fn expected_score(&self, creature_a: u64, creature_b: u64) -> f64 {
        let ra = self.get_rating(creature_a);
        let rb = self.get_rating(creature_b);
        1.0 / (1.0 + 10.0_f64.powf((rb - ra) / self.config.scale))
    }

    /// Update ratings after a match.
    ///
    /// `actual_score_a` should be:
    /// - 1.0 if creature A won
    /// - 0.0 if creature B won
    /// - 0.5 for a draw
    /// - Or any value in [0, 1] for a continuous outcome
    pub fn record_match(&mut self, creature_a: u64, creature_b: u64, actual_score_a: f64) {
        self.register(creature_a);
        self.register(creature_b);

        let expected_a = self.expected_score(creature_a, creature_b);
        let expected_b = 1.0 - expected_a;
        let actual_b = 1.0 - actual_score_a;

        let k = self.config.k_factor;

        // Update A
        {
            let rating_a = self.ratings.get_mut(&creature_a).unwrap();
            rating_a.rating += k * (actual_score_a - expected_a);
            rating_a.matches_played += 1;
            if actual_score_a > 0.5 + f64::EPSILON {
                rating_a.wins += 1;
            } else if actual_score_a < 0.5 - f64::EPSILON {
                rating_a.losses += 1;
            } else {
                rating_a.draws += 1;
            }
            rating_a.history.push(rating_a.rating);
        }

        // Update B
        {
            let rating_b = self.ratings.get_mut(&creature_b).unwrap();
            rating_b.rating += k * (actual_b - expected_b);
            rating_b.matches_played += 1;
            if actual_b > 0.5 + f64::EPSILON {
                rating_b.wins += 1;
            } else if actual_b < 0.5 - f64::EPSILON {
                rating_b.losses += 1;
            } else {
                rating_b.draws += 1;
            }
            rating_b.history.push(rating_b.rating);
        }
    }

    /// Get a creature's current rating (or initial if unregistered).
    pub fn get_rating(&self, creature_id: u64) -> f64 {
        self.ratings
            .get(&creature_id)
            .map(|r| r.rating)
            .unwrap_or(self.config.initial_rating)
    }

    /// Get all ratings sorted by rating (descending).
    pub fn leaderboard(&self) -> Vec<(u64, &CreatureRating)> {
        let mut entries: Vec<_> = self.ratings.iter().map(|(&id, r)| (id, r)).collect();
        entries.sort_by(|a, b| b.1.rating.partial_cmp(&a.1.rating).unwrap_or(std::cmp::Ordering::Equal));
        entries
    }

    /// Build the payoff matrix for all registered creatures.
    /// Matrix[i][j] = actual score of creature i against creature j.
    /// Returns (creature_ids, matrix) where creature_ids maps index to ID.
    pub fn payoff_matrix(&self, results: &[MatchRecord]) -> (Vec<u64>, Vec<Vec<f64>>) {
        let ids: Vec<u64> = {
            let mut ids: Vec<_> = self.ratings.keys().copied().collect();
            ids.sort();
            ids
        };
        let n = ids.len();
        let id_to_idx: HashMap<u64, usize> = ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        // Accumulate scores and counts
        let mut total_scores = vec![vec![0.0_f64; n]; n];
        let mut match_counts = vec![vec![0usize; n]; n];

        for record in results {
            if let (Some(&i), Some(&j)) = (id_to_idx.get(&record.creature_a), id_to_idx.get(&record.creature_b)) {
                total_scores[i][j] += record.score_a;
                total_scores[j][i] += 1.0 - record.score_a;
                match_counts[i][j] += 1;
                match_counts[j][i] += 1;
            }
        }

        // Average scores
        let matrix: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| {
                        if match_counts[i][j] > 0 {
                            total_scores[i][j] / match_counts[i][j] as f64
                        } else if i == j {
                            0.5
                        } else {
                            0.5 // No data = assume even
                        }
                    })
                    .collect()
            })
            .collect();

        (ids, matrix)
    }
}

/// A record of a single match outcome.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchRecord {
    /// Creature A's ID.
    pub creature_a: u64,
    /// Creature B's ID.
    pub creature_b: u64,
    /// Score for creature A in [0, 1]. Score for B is 1 - score_a.
    pub score_a: f64,
    /// Which fitness criterion was used.
    pub criterion: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_ratings() {
        let mut elo = EloSystem::with_defaults();
        elo.register(1);
        elo.register(2);
        assert!((elo.get_rating(1) - 1500.0).abs() < f64::EPSILON);
        assert!((elo.get_rating(2) - 1500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_expected_score_equal() {
        let mut elo = EloSystem::with_defaults();
        elo.register(1);
        elo.register(2);
        let expected = elo.expected_score(1, 2);
        assert!((expected - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_win_updates_rating() {
        let mut elo = EloSystem::with_defaults();
        elo.register(1);
        elo.register(2);
        elo.record_match(1, 2, 1.0); // Creature 1 wins
        assert!(elo.get_rating(1) > 1500.0);
        assert!(elo.get_rating(2) < 1500.0);
    }

    #[test]
    fn test_elo_conservation() {
        let mut elo = EloSystem::with_defaults();
        elo.register(1);
        elo.register(2);
        elo.record_match(1, 2, 1.0);
        let total: f64 = elo.ratings.values().map(|r| r.rating).sum();
        assert!((total - 3000.0).abs() < 0.001, "Elo should be zero-sum");
    }

    #[test]
    fn test_strong_player_expected_to_win() {
        let mut elo = EloSystem::with_defaults();
        elo.register(1);
        elo.register(2);
        // Give creature 1 a much higher rating
        for _ in 0..20 {
            elo.record_match(1, 2, 1.0);
        }
        let expected = elo.expected_score(1, 2);
        assert!(expected > 0.8, "Strong player should be expected to win: {}", expected);
    }

    #[test]
    fn test_leaderboard_ordering() {
        let mut elo = EloSystem::with_defaults();
        elo.register(1);
        elo.register(2);
        elo.register(3);
        elo.record_match(1, 2, 1.0);
        elo.record_match(1, 3, 1.0);
        let board = elo.leaderboard();
        assert_eq!(board[0].0, 1, "Creature 1 should be top rated");
    }

    #[test]
    fn test_payoff_matrix() {
        let mut elo = EloSystem::with_defaults();
        elo.register(1);
        elo.register(2);

        let records = vec![
            MatchRecord { creature_a: 1, creature_b: 2, score_a: 1.0, criterion: "test".into() },
            MatchRecord { creature_a: 1, creature_b: 2, score_a: 1.0, criterion: "test".into() },
            MatchRecord { creature_a: 2, creature_b: 1, score_a: 0.0, criterion: "test".into() },
        ];

        let (ids, matrix) = elo.payoff_matrix(&records);
        assert_eq!(ids.len(), 2);
        // Creature 1 always beats creature 2
        assert!((matrix[0][1] - 1.0).abs() < 0.001);
        assert!((matrix[1][0] - 0.0).abs() < 0.001);
    }
}
