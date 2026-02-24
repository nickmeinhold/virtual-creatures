//! Morphological analysis and diversity metrics for creature genotypes.
//!
//! Provides computed descriptors that summarize a genotype's structure,
//! enabling diversity measurement across populations — critical for
//! studying diversity-fitness tradeoffs in variable-topology evolution.

use serde::{Deserialize, Serialize};

use super::morphology::CreatureGenotype;
use super::graph::NodeId;

/// A compact descriptor summarizing a genotype's morphology.
///
/// Used for computing pairwise distances between creatures and measuring
/// population diversity without comparing full genotype graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyDescriptor {
    /// Number of nodes in the genotype graph
    pub node_count: usize,
    /// Number of connections in the genotype graph
    pub connection_count: usize,
    /// Total volume of all parts (sum of x*y*z)
    pub total_volume: f32,
    /// Mean volume per part
    pub mean_volume: f32,
    /// Volume standard deviation
    pub volume_std: f32,
    /// Max depth of the morphology tree from root
    pub max_depth: usize,
    /// Branching factor: mean number of outgoing connections per node
    pub mean_branching: f32,
    /// Number of leaf nodes (no outgoing connections)
    pub leaf_count: usize,
    /// Total degrees of freedom across all joints
    pub total_dof: usize,
    /// Number of neurons across all parts
    pub total_neurons: usize,
    /// Number of sensors across all parts
    pub total_sensors: usize,
    /// Number of effectors across all parts
    pub total_effectors: usize,
    /// Symmetry score: how bilaterally symmetric the creature is (0-1)
    pub symmetry_score: f32,
    /// Distribution of joint types [rigid, revolute, twist, universal, bendtwist, twistbend, spherical]
    pub joint_type_distribution: [f32; 7],
}

impl MorphologyDescriptor {
    /// Compute descriptor from a creature genotype
    pub fn from_genotype(genotype: &CreatureGenotype) -> Self {
        let node_count = genotype.morphology.node_count();
        let connection_count = genotype.morphology.connection_count();

        // Volume statistics
        let volumes: Vec<f32> = genotype
            .morphology
            .nodes()
            .map(|(_, n)| n.volume())
            .collect();

        let total_volume: f32 = volumes.iter().sum();
        let mean_volume = if !volumes.is_empty() {
            total_volume / volumes.len() as f32
        } else {
            0.0
        };
        let volume_std = if volumes.len() > 1 {
            let variance = volumes
                .iter()
                .map(|v| (v - mean_volume).powi(2))
                .sum::<f32>()
                / (volumes.len() - 1) as f32;
            variance.sqrt()
        } else {
            0.0
        };

        // Depth calculation via BFS
        let max_depth = compute_max_depth(genotype);

        // Branching statistics
        let branching_factors: Vec<usize> = genotype
            .morphology
            .nodes()
            .map(|(id, _)| genotype.morphology.connections_from(id).count())
            .collect();

        let mean_branching = if !branching_factors.is_empty() {
            branching_factors.iter().sum::<usize>() as f32 / branching_factors.len() as f32
        } else {
            0.0
        };

        let leaf_count = branching_factors.iter().filter(|&&b| b == 0).count();

        // Neural statistics
        let mut total_dof = 0;
        let mut total_neurons = 0;
        let mut total_sensors = 0;
        let mut total_effectors = 0;
        let mut joint_counts = [0usize; 7];

        for (_, node) in genotype.morphology.nodes() {
            total_dof += node.joint_type.dof();
            total_neurons += node.neural.neurons.len();
            total_sensors += node.neural.sensors.len();
            total_effectors += node.neural.effectors.len();

            let jt_idx = match node.joint_type {
                super::morphology::JointType::Rigid => 0,
                super::morphology::JointType::Revolute => 1,
                super::morphology::JointType::Twist => 2,
                super::morphology::JointType::Universal => 3,
                super::morphology::JointType::BendTwist => 4,
                super::morphology::JointType::TwistBend => 5,
                super::morphology::JointType::Spherical => 6,
            };
            joint_counts[jt_idx] += 1;
        }

        let joint_type_distribution = if node_count > 0 {
            let n = node_count as f32;
            [
                joint_counts[0] as f32 / n,
                joint_counts[1] as f32 / n,
                joint_counts[2] as f32 / n,
                joint_counts[3] as f32 / n,
                joint_counts[4] as f32 / n,
                joint_counts[5] as f32 / n,
                joint_counts[6] as f32 / n,
            ]
        } else {
            [0.0; 7]
        };

        // Symmetry score: check if connections come in reflected pairs
        let symmetry_score = compute_symmetry_score(genotype);

        Self {
            node_count,
            connection_count,
            total_volume,
            mean_volume,
            volume_std,
            max_depth,
            mean_branching,
            leaf_count,
            total_dof,
            total_neurons,
            total_sensors,
            total_effectors,
            symmetry_score,
            joint_type_distribution,
        }
    }

    /// Compute a normalized distance between two morphological descriptors.
    ///
    /// Returns a value in [0, inf) where 0 means identical descriptors.
    /// Each feature is normalized by its typical scale before computing
    /// Euclidean distance.
    pub fn distance(&self, other: &MorphologyDescriptor) -> f32 {
        let mut sum_sq = 0.0;

        // Structural features (normalized by typical ranges)
        sum_sq += ((self.node_count as f32 - other.node_count as f32) / 5.0).powi(2);
        sum_sq += ((self.connection_count as f32 - other.connection_count as f32) / 5.0).powi(2);
        sum_sq += ((self.max_depth as f32 - other.max_depth as f32) / 5.0).powi(2);
        sum_sq += ((self.leaf_count as f32 - other.leaf_count as f32) / 3.0).powi(2);
        sum_sq += ((self.mean_branching - other.mean_branching) / 2.0).powi(2);

        // Volume features
        sum_sq += ((self.total_volume - other.total_volume) / 2.0).powi(2);
        sum_sq += ((self.mean_volume - other.mean_volume) / 0.5).powi(2);

        // Neural features
        sum_sq += ((self.total_dof as f32 - other.total_dof as f32) / 5.0).powi(2);
        sum_sq += ((self.total_neurons as f32 - other.total_neurons as f32) / 5.0).powi(2);

        // Joint type distribution (cosine-distance-like)
        for i in 0..7 {
            sum_sq += (self.joint_type_distribution[i] - other.joint_type_distribution[i]).powi(2);
        }

        // Symmetry
        sum_sq += (self.symmetry_score - other.symmetry_score).powi(2);

        sum_sq.sqrt()
    }
}

/// Compute the maximum depth of the morphology tree from root
fn compute_max_depth(genotype: &CreatureGenotype) -> usize {
    fn depth_from(genotype: &CreatureGenotype, node: NodeId, visited: &mut Vec<NodeId>, depth: usize) -> usize {
        if depth >= 10 || visited.contains(&node) {
            return depth;
        }
        visited.push(node);

        let mut max = depth;
        for conn in genotype.morphology.connections_from(node) {
            let child_depth = depth_from(genotype, conn.to, visited, depth + 1);
            if child_depth > max {
                max = child_depth;
            }
        }

        visited.pop();
        max
    }

    let mut visited = Vec::new();
    depth_from(genotype, genotype.root, &mut visited, 0)
}

/// Compute a rough symmetry score for the creature.
///
/// Measures how many connections have a reflected counterpart with a similar
/// position (flipped on X axis — bilateral symmetry). Returns 0-1 where
/// 1 means perfectly symmetric and 0 means completely asymmetric.
fn compute_symmetry_score(genotype: &CreatureGenotype) -> f32 {
    let connections: Vec<_> = genotype.morphology.connections().collect();
    if connections.is_empty() {
        return 1.0; // Single node is trivially symmetric
    }

    let mut matched = 0;
    let mut total = 0;

    for conn in &connections {
        total += 1;
        let pos = conn.data.position;
        let reflected_x = bevy::prelude::Vec3::new(-pos.x, pos.y, pos.z);

        // Check if any other connection from the same parent has a reflected position
        for other in &connections {
            if std::ptr::eq(*conn, *other) {
                continue;
            }
            if conn.from == other.from {
                let dist = (other.data.position - reflected_x).length();
                if dist < 0.3 {
                    matched += 1;
                    break;
                }
            }
        }
    }

    matched as f32 / total as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genotype::morphology::{MorphologyNode, JointType};
    use bevy::prelude::Vec3;

    #[test]
    fn test_descriptor_single_node() {
        let root = MorphologyNode::new(Vec3::new(1.0, 0.5, 0.5), JointType::Rigid);
        let genotype = CreatureGenotype::new(root);
        let desc = MorphologyDescriptor::from_genotype(&genotype);

        assert_eq!(desc.node_count, 1);
        assert_eq!(desc.connection_count, 0);
        assert_eq!(desc.max_depth, 0);
        assert_eq!(desc.leaf_count, 1);
        assert!((desc.total_volume - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_distance_identical() {
        let root = MorphologyNode::new(Vec3::new(1.0, 0.5, 0.5), JointType::Rigid);
        let genotype = CreatureGenotype::new(root);
        let desc = MorphologyDescriptor::from_genotype(&genotype);

        assert!((desc.distance(&desc) - 0.0).abs() < 0.001);
    }

    #[test]
    fn test_distance_different() {
        let g1 = CreatureGenotype::new(
            MorphologyNode::new(Vec3::new(1.0, 0.5, 0.5), JointType::Rigid),
        );
        let g2 = CreatureGenotype::new(
            MorphologyNode::new(Vec3::new(2.0, 1.0, 1.0), JointType::Spherical),
        );
        let d1 = MorphologyDescriptor::from_genotype(&g1);
        let d2 = MorphologyDescriptor::from_genotype(&g2);

        assert!(d1.distance(&d2) > 0.1);
    }
}
