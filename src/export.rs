//! Export evolved creatures to a compact JSON format the web gallery plays.
//!
//! Creatures here are just articulated boxes, so rather than authoring glTF
//! animation tracks we bake each part's world transform per frame into a flat
//! JSON. A Three.js component on the site rebuilds a box per part and replays
//! the frames — identical visual result, a fraction of the machinery.

use std::fs;
use std::io;
use std::path::Path;

/// One rigid part of a creature (an oriented box).
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebPart {
    /// Full box dimensions [x, y, z] in metres.
    pub dims: [f32; 3],
}

/// A single recorded creature: its parts plus a per-frame pose track.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebCreature {
    pub id: usize,
    pub fitness: f32,
    pub generation: usize,
    pub species_id: usize,
    /// Parts in a stable order; matches the inner arrays of `frames`.
    pub parts: Vec<WebPart>,
    /// frames[f][p] = [px, py, pz, qx, qy, qz, qw] — world pose of part `p`
    /// at frame `f`. Uniform timestep of 1/fps seconds between frames.
    pub frames: Vec<Vec<[f32; 7]>>,
}

/// One generation's worth of recorded creatures, in playback order.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebGenSnapshot {
    pub gen: usize,
    /// Creatures of this generation, ranked best-first.
    pub creatures: Vec<WebCreature>,
}

/// One evolution objective (what the population was selected for), carrying its
/// human label, fitness unit, and the per-generation creatures the gallery
/// steps through.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebObjective {
    /// Stable key, e.g. "distance".
    pub key: String,
    /// Human label, e.g. "distance travelled".
    pub label: String,
    /// Fitness unit, e.g. "m/s".
    pub unit: String,
    /// Playback rate the frames were sampled at.
    pub fps: u32,
    /// Generations in ascending order; each holds that generation's creatures.
    pub generations: Vec<WebGenSnapshot>,
}

/// Top-level gallery document written to disk: every objective, every
/// generation, baked and ready for the web viewer to replay the evolutionary
/// arc. The label/unit live here (and in code) so the file is reproducible
/// rather than hand-assembled.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebMultiGallery {
    pub objectives: Vec<WebObjective>,
}

impl WebMultiGallery {
    /// Assemble from a flat list of baked creatures tagged with their objective
    /// index. `objectives_meta` is `(key, label, unit)` indexed by `obj_idx`.
    /// Within each objective, creatures are grouped by `generation` and sorted
    /// ascending; within a generation they are ranked best-first by fitness.
    pub fn assemble(
        fps: u32,
        objectives_meta: &[(String, String, String)],
        tagged: Vec<(usize, WebCreature)>,
    ) -> Self {
        let mut objectives: Vec<WebObjective> = objectives_meta
            .iter()
            .map(|(key, label, unit)| WebObjective {
                key: key.clone(),
                label: label.clone(),
                unit: unit.clone(),
                fps,
                generations: Vec::new(),
            })
            .collect();

        for (obj_idx, creature) in tagged {
            let Some(obj) = objectives.get_mut(obj_idx) else { continue };
            let gen = creature.generation;
            match obj.generations.iter_mut().find(|g| g.gen == gen) {
                Some(snap) => snap.creatures.push(creature),
                None => obj.generations.push(WebGenSnapshot { gen, creatures: vec![creature] }),
            }
        }

        for obj in &mut objectives {
            obj.generations.sort_by_key(|g| g.gen);
            for snap in &mut obj.generations {
                snap.creatures
                    .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap_or(std::cmp::Ordering::Equal));
            }
        }

        // Drop objectives that produced nothing (e.g. a missing history file).
        objectives.retain(|o| !o.generations.is_empty());
        Self { objectives }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let json = serde_json::to_string(self).map_err(io::Error::other)?;
        fs::write(path, json)
    }
}
