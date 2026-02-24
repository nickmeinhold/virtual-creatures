//! Save and load functionality for creature genotypes.

use std::fs;
use std::io;
use std::path::Path;

use super::CreatureGenotype;

/// A saved creature with metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedCreature {
    /// The creature's genotype
    pub genotype: CreatureGenotype,
    /// Fitness score when saved
    pub fitness: f32,
    /// Generation when saved
    pub generation: usize,
    /// Number of body parts
    pub part_count: usize,
}

impl SavedCreature {
    pub fn new(genotype: CreatureGenotype, fitness: f32, generation: usize, part_count: usize) -> Self {
        Self {
            genotype,
            fitness,
            generation,
            part_count,
        }
    }
}

/// A collection of saved creatures (e.g., hall of fame)
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CreatureArchive {
    /// Best creatures from evolution
    pub creatures: Vec<SavedCreature>,
}

impl CreatureArchive {
    pub fn new() -> Self {
        Self {
            creatures: Vec::new(),
        }
    }

    /// Add a creature to the archive
    pub fn add(&mut self, creature: SavedCreature) {
        self.creatures.push(creature);
    }

    /// Keep only the best N creatures
    pub fn keep_best(&mut self, n: usize) {
        self.creatures.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        self.creatures.truncate(n);
    }

    /// Save to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        fs::write(path, json)
    }

    /// Load from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(io::Error::other)
    }

    /// Get the best creature
    #[allow(dead_code)]
    pub fn best(&self) -> Option<&SavedCreature> {
        self.creatures.iter().max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }
}

/// Save a single creature to a file
#[allow(dead_code)]
pub fn save_creature<P: AsRef<Path>>(creature: &SavedCreature, path: P) -> io::Result<()> {
    let json = serde_json::to_string_pretty(creature).map_err(io::Error::other)?;
    fs::write(path, json)
}

/// Load a single creature from a file
#[allow(dead_code)]
pub fn load_creature<P: AsRef<Path>>(path: P) -> io::Result<SavedCreature> {
    let json = fs::read_to_string(path)?;
    serde_json::from_str(&json).map_err(io::Error::other)
}

// ============================================================================
// Population Snapshots — for experiment data collection
// ============================================================================

use crate::evolution::Lineage;
use super::analysis::MorphologyDescriptor;

/// A saved individual with full metadata for analysis
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SavedIndividual {
    /// Unique creature ID
    pub id: u64,
    /// The creature's genotype
    pub genotype: CreatureGenotype,
    /// Fitness score
    pub fitness: f32,
    /// Lineage information
    pub lineage: Lineage,
    /// Precomputed morphological descriptor
    pub descriptor: MorphologyDescriptor,
}

/// A snapshot of the entire population at a given generation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PopulationSnapshot {
    /// Generation number
    pub generation: usize,
    /// All individuals in the population
    pub individuals: Vec<SavedIndividual>,
}

impl PopulationSnapshot {
    /// Save to a JSON file
    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(io::Error::other)?;
        fs::write(path, json)
    }

    /// Load from a JSON file
    pub fn load<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let json = fs::read_to_string(path)?;
        serde_json::from_str(&json).map_err(io::Error::other)
    }

    /// Compute population-level diversity statistics
    pub fn diversity_stats(&self) -> DiversityStats {
        let descriptors: Vec<_> = self.individuals.iter().map(|i| &i.descriptor).collect();
        DiversityStats::compute(&descriptors)
    }
}

/// Aggregate diversity statistics for a population
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DiversityStats {
    /// Mean pairwise morphological distance
    pub mean_pairwise_distance: f32,
    /// Standard deviation of fitness
    pub fitness_std: f32,
    /// Mean fitness
    pub fitness_mean: f32,
    /// Best fitness
    pub fitness_best: f32,
    /// Mean node count
    pub mean_node_count: f32,
    /// Unique reproduction methods used
    pub reproduction_method_counts: ReproductionMethodCounts,
}

/// Counts of each reproduction method in the population
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ReproductionMethodCounts {
    pub random: usize,
    pub asexual: usize,
    pub crossover: usize,
    pub grafting: usize,
}

impl DiversityStats {
    pub fn compute(descriptors: &[&MorphologyDescriptor]) -> Self {
        let n = descriptors.len() as f32;
        if n == 0.0 {
            return Self {
                mean_pairwise_distance: 0.0,
                fitness_std: 0.0,
                fitness_mean: 0.0,
                fitness_best: 0.0,
                mean_node_count: 0.0,
                reproduction_method_counts: ReproductionMethodCounts::default(),
            };
        }

        // Mean pairwise distance
        let mut total_distance = 0.0;
        let mut pair_count = 0;
        for i in 0..descriptors.len() {
            for j in (i + 1)..descriptors.len() {
                total_distance += descriptors[i].distance(descriptors[j]);
                pair_count += 1;
            }
        }
        let mean_pairwise_distance = if pair_count > 0 {
            total_distance / pair_count as f32
        } else {
            0.0
        };

        let mean_node_count = descriptors.iter().map(|d| d.node_count as f32).sum::<f32>() / n;

        Self {
            mean_pairwise_distance,
            fitness_std: 0.0,  // Filled separately when fitness is available
            fitness_mean: 0.0,
            fitness_best: 0.0,
            mean_node_count,
            reproduction_method_counts: ReproductionMethodCounts::default(),
        }
    }
}
