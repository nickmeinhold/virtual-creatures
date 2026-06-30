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
    /// NEAT species id this creature belongs to (distinct genetic lineage).
    /// The archive keeps one champion per species so the gallery is diverse
    /// by construction rather than 10 near-clones of the fastest mover.
    #[serde(default)]
    pub species_id: usize,
}

impl SavedCreature {
    pub fn new(genotype: CreatureGenotype, fitness: f32, generation: usize, part_count: usize, species_id: usize) -> Self {
        Self {
            genotype,
            fitness,
            generation,
            part_count,
            species_id,
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
    #[allow(dead_code)]
    pub fn add(&mut self, creature: SavedCreature) {
        self.creatures.push(creature);
    }

    /// Insert or update the champion for a species lineage.
    ///
    /// Keeps exactly one entry per `species_id`: the highest-fitness creature
    /// ever seen in that lineage. This is what makes the gallery diverse —
    /// each entry is a distinct body-plan family, not a clone of the fastest.
    pub fn upsert_species_champion(&mut self, creature: SavedCreature) {
        match self.creatures.iter_mut().find(|c| c.species_id == creature.species_id) {
            Some(existing) if creature.fitness > existing.fitness => *existing = creature,
            Some(_) => {} // existing champion is stronger; keep it
            None => self.creatures.push(creature),
        }
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
