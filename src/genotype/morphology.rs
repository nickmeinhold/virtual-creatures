//! Morphology representation for creature body structure.
//!
//! The morphology is a directed graph where:
//! - Nodes define body parts (dimensions, joint type, neural circuitry)
//! - Connections define how parts attach to each other
//! - The graph can be recursive for fractal-like structures

use bevy::prelude::*;
use serde::{Deserialize, Serialize};

use super::graph::{DirectedGraph, NodeId};
use super::neural::{CentralNervousSystem, NeuralGraph};

/// Joint types from Karl Sims' paper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum JointType {
    /// No movement allowed
    Rigid,
    /// 1 DOF: rotation around one axis
    Revolute,
    /// 1 DOF: rotation around the attachment axis
    Twist,
    /// 2 DOF: rotation around two perpendicular axes
    Universal,
    /// 2 DOF: bend then twist
    BendTwist,
    /// 2 DOF: twist then bend
    TwistBend,
    /// 3 DOF: full rotation freedom
    Spherical,
}

impl JointType {
    /// Number of degrees of freedom for this joint type
    pub fn dof(&self) -> usize {
        match self {
            JointType::Rigid => 0,
            JointType::Revolute | JointType::Twist => 1,
            JointType::Universal | JointType::BendTwist | JointType::TwistBend => 2,
            JointType::Spherical => 3,
        }
    }
}

/// Joint limits for each degree of freedom
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLimits {
    /// (min, max) angle in radians for each DOF
    pub limits: Vec<(f32, f32)>,
    /// Spring stiffness when limit is exceeded
    pub stiffness: f32,
}

impl JointLimits {
    pub fn new(dof: usize) -> Self {
        Self {
            limits: vec![(-std::f32::consts::PI, std::f32::consts::PI); dof],
            stiffness: 100.0,
        }
    }
}

/// A body part node in the morphology graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyNode {
    /// Dimensions of the box-shaped part
    pub dimensions: Vec3,
    /// Type of joint connecting this part to its parent
    pub joint_type: JointType,
    /// Limits for each joint DOF
    pub joint_limits: JointLimits,
    /// How many times to recurse when this node is in a cycle
    pub recursive_limit: u8,
    /// Local neural circuitry for this part
    pub neural: NeuralGraph,
}

impl MorphologyNode {
    pub fn new(dimensions: Vec3, joint_type: JointType) -> Self {
        let dof = joint_type.dof();
        Self {
            dimensions,
            joint_type,
            joint_limits: JointLimits::new(dof),
            recursive_limit: 1,
            neural: NeuralGraph::new(),
        }
    }

    /// Volume of this part (for mass calculation)
    pub fn volume(&self) -> f32 {
        self.dimensions.x * self.dimensions.y * self.dimensions.z
    }

    /// Maximum cross-sectional area (for effector strength scaling)
    pub fn max_cross_section(&self) -> f32 {
        let xy = self.dimensions.x * self.dimensions.y;
        let xz = self.dimensions.x * self.dimensions.z;
        let yz = self.dimensions.y * self.dimensions.z;
        xy.max(xz).max(yz)
    }
}

/// How a child part attaches to its parent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MorphologyConnection {
    /// Position on parent's surface where child attaches
    /// Normalized to [-1, 1] for each axis, clamped to surface
    pub position: Vec3,
    /// Orientation of child relative to parent
    pub orientation: Quat,
    /// Scale factor applied to child
    pub scale: f32,
    /// Which axes to reflect (for symmetry)
    pub reflection: Vec3,
    /// If true, only create this connection at recursive limit
    pub terminal_only: bool,
}

impl MorphologyConnection {
    pub fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 1.0), // attach at +Z face by default
            orientation: Quat::IDENTITY,
            scale: 1.0,
            reflection: Vec3::ONE, // no reflection
            terminal_only: false,
        }
    }

    /// Create a reflected version of this connection
    pub fn reflected(&self, axis: ReflectAxis) -> Self {
        let mut conn = self.clone();
        match axis {
            ReflectAxis::X => {
                conn.position.x = -conn.position.x;
                conn.reflection.x = -conn.reflection.x;
            }
            ReflectAxis::Y => {
                conn.position.y = -conn.position.y;
                conn.reflection.y = -conn.reflection.y;
            }
            ReflectAxis::Z => {
                conn.position.z = -conn.position.z;
                conn.reflection.z = -conn.reflection.z;
            }
        }
        conn
    }
}

impl Default for MorphologyConnection {
    fn default() -> Self {
        Self::new()
    }
}

/// Axis for reflection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReflectAxis {
    X,
    Y,
    Z,
}

/// Complete genotype for a creature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreatureGenotype {
    /// The morphology graph
    pub morphology: DirectedGraph<MorphologyNode, MorphologyConnection>,
    /// The root node of the morphology
    pub root: NodeId,
    /// Centralized neurons not associated with any part
    pub central_nervous_system: CentralNervousSystem,
}

impl CreatureGenotype {
    /// Create a new creature with a single root part
    pub fn new(root_node: MorphologyNode) -> Self {
        let mut morphology = DirectedGraph::new();
        let root = morphology.add_node(root_node);
        Self {
            morphology,
            root,
            central_nervous_system: CentralNervousSystem::new(),
        }
    }

    /// Add a part connected to an existing part
    pub fn add_part(
        &mut self,
        parent: NodeId,
        node: MorphologyNode,
        connection: MorphologyConnection,
    ) -> NodeId {
        let child = self.morphology.add_node(node);
        self.morphology.add_connection(parent, child, connection);
        child
    }

    /// Get the root node
    pub fn root_node(&self) -> &MorphologyNode {
        &self.morphology[self.root]
    }

    /// Number of part types (nodes in genotype, not instantiated parts)
    pub fn part_type_count(&self) -> usize {
        self.morphology.node_count()
    }
}
