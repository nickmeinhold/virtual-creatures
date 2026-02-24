//! Arena tournament system for creature Elo ratings.
//!
//! Substrate-agnostic: works with any evaluation backend (Bevy/Rapier sim,
//! pre-recorded scores, or hypothetical matchups). The tournament system
//! handles scheduling, Elo computation, and Balduzzi et al.'s transitive-cyclic
//! decomposition for detecting intransitive dominance.

pub mod criteria;
pub mod decomposition;
pub mod ratings;
pub mod tournament;

pub use criteria::*;
pub use decomposition::*;
pub use ratings::*;
pub use tournament::*;
