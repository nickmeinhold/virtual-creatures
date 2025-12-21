//! Genotype representation for virtual creatures.
//!
//! Based on Karl Sims' directed graph approach where:
//! - Morphology is a directed graph of body parts and connections
//! - Neural control is nested within each body part node
//! - Graphs can be recursive/cyclic for fractal-like structures

mod graph;
mod morphology;
mod neural;

pub use graph::*;
pub use morphology::*;
pub use neural::*;
