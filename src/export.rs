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

/// Top-level gallery document written to disk.
#[derive(Debug, Clone, serde::Serialize)]
pub struct WebGallery {
    /// Playback rate the frames were sampled at.
    pub fps: u32,
    pub creatures: Vec<WebCreature>,
}

impl WebGallery {
    pub fn new(fps: u32) -> Self {
        Self { fps, creatures: Vec::new() }
    }

    pub fn save<P: AsRef<Path>>(&self, path: P) -> io::Result<()> {
        let json = serde_json::to_string(self).map_err(io::Error::other)?;
        fs::write(path, json)
    }
}
