//! Phenotype builder - spawns creatures from genotypes.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::collections::HashMap;

use crate::genotype::*;

/// Marker component for creature parts
#[derive(Component)]
pub struct CreaturePart {
    /// Which creature this part belongs to
    pub creature_id: Entity,
    /// Index of the genotype node this was spawned from
    pub node_id: NodeId,
    /// Instance index (for parts spawned multiple times from same node)
    pub instance: usize,
}

/// Marker component for the root of a creature
#[derive(Component)]
pub struct CreatureRoot;

/// Tracks all spawned parts for a creature
#[derive(Component)]
pub struct CreatureBody {
    /// All part entities in this creature
    pub parts: Vec<Entity>,
}

/// Result of spawning a creature
pub struct SpawnedCreature {
    /// The root entity
    pub root: Entity,
    /// All part entities
    pub parts: Vec<Entity>,
}

/// Builder for spawning creatures from genotypes
pub struct PhenotypeBuilder<'a> {
    commands: &'a mut Commands<'a, 'a>,
    meshes: &'a mut Assets<Mesh>,
    materials: &'a mut Assets<StandardMaterial>,
}

impl<'a> PhenotypeBuilder<'a> {
    /// Spawn a creature from a genotype at the given position
    pub fn spawn(
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        genotype: &CreatureGenotype,
        position: Vec3,
    ) -> SpawnedCreature {
        let mut spawned_parts = Vec::new();
        let mut node_instances: HashMap<NodeId, usize> = HashMap::new();

        // Create a placeholder root entity first
        let creature_id = commands.spawn_empty().id();

        // Spawn the root part
        let root_node = &genotype.morphology[genotype.root];
        let root_entity = Self::spawn_part(
            commands,
            meshes,
            materials,
            creature_id,
            genotype.root,
            0,
            root_node,
            Transform::from_translation(position),
            None,
            Vec3::ONE, // no reflection
        );
        spawned_parts.push(root_entity);
        node_instances.insert(genotype.root, 1);

        // Recursively spawn children
        Self::spawn_children(
            commands,
            meshes,
            materials,
            genotype,
            creature_id,
            genotype.root,
            root_entity,
            root_node,
            Transform::from_translation(position),
            Vec3::ONE,
            &mut node_instances,
            &mut spawned_parts,
            0, // recursion depth
        );

        // Update the root entity with creature components
        commands.entity(creature_id).insert((
            CreatureRoot,
            CreatureBody {
                parts: spawned_parts.clone(),
            },
        ));

        // Also mark the root part
        commands.entity(root_entity).insert(CreatureRoot);

        SpawnedCreature {
            root: root_entity,
            parts: spawned_parts,
        }
    }

    fn spawn_children(
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        genotype: &CreatureGenotype,
        creature_id: Entity,
        parent_node_id: NodeId,
        parent_entity: Entity,
        parent_node: &MorphologyNode,
        parent_transform: Transform,
        parent_reflection: Vec3,
        node_instances: &mut HashMap<NodeId, usize>,
        spawned_parts: &mut Vec<Entity>,
        depth: usize,
    ) {
        const MAX_DEPTH: usize = 10; // prevent infinite recursion
        if depth >= MAX_DEPTH {
            return;
        }

        for conn in genotype.morphology.connections_from(parent_node_id) {
            let child_node = &genotype.morphology[conn.to];

            // Check recursive limit
            let instance_count = node_instances.get(&conn.to).copied().unwrap_or(0);
            if instance_count >= child_node.recursive_limit as usize {
                continue;
            }

            // Handle terminal_only flag
            let at_terminal = instance_count + 1 >= child_node.recursive_limit as usize;
            if conn.data.terminal_only && !at_terminal {
                continue;
            }

            // Compute child transform
            let combined_reflection = parent_reflection * conn.data.reflection;
            let child_transform = Self::compute_child_transform(
                &parent_transform,
                parent_node,
                &conn.data,
                combined_reflection,
            );

            // Spawn the child part
            let instance = instance_count;
            let child_entity = Self::spawn_part(
                commands,
                meshes,
                materials,
                creature_id,
                conn.to,
                instance,
                child_node,
                child_transform,
                Some((parent_entity, parent_node, &conn.data)),
                combined_reflection,
            );
            spawned_parts.push(child_entity);
            *node_instances.entry(conn.to).or_insert(0) += 1;

            // Recurse to spawn grandchildren
            Self::spawn_children(
                commands,
                meshes,
                materials,
                genotype,
                creature_id,
                conn.to,
                child_entity,
                child_node,
                child_transform,
                combined_reflection,
                node_instances,
                spawned_parts,
                depth + 1,
            );
        }
    }

    fn spawn_part(
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        creature_id: Entity,
        node_id: NodeId,
        instance: usize,
        node: &MorphologyNode,
        transform: Transform,
        parent_info: Option<(Entity, &MorphologyNode, &MorphologyConnection)>,
        _reflection: Vec3,
    ) -> Entity {
        let dims = node.dimensions;

        // Create mesh and material
        let mesh = meshes.add(Cuboid::new(dims.x, dims.y, dims.z));
        let material = materials.add(Color::srgb(
            0.5 + 0.3 * (node_id.0 as f32 * 0.7).sin(),
            0.5 + 0.3 * (node_id.0 as f32 * 1.3).cos(),
            0.5 + 0.3 * (node_id.0 as f32 * 2.1).sin(),
        ));

        // Spawn the entity
        let mut entity_commands = commands.spawn((
            Mesh3d(mesh),
            MeshMaterial3d(material),
            transform,
            RigidBody::Dynamic,
            Collider::cuboid(dims.x / 2.0, dims.y / 2.0, dims.z / 2.0),
            ColliderMassProperties::Density(1.0),
            CreaturePart {
                creature_id,
                node_id,
                instance,
            },
        ));

        let entity = entity_commands.id();

        // Add joint to parent if this isn't the root
        if let Some((parent_entity, parent_node, connection)) = parent_info {
            let joint = Self::create_joint(node, parent_node, connection);
            entity_commands.insert(ImpulseJoint::new(parent_entity, joint));
        }

        entity
    }

    fn compute_child_transform(
        parent_transform: &Transform,
        parent_node: &MorphologyNode,
        connection: &MorphologyConnection,
        reflection: Vec3,
    ) -> Transform {
        // Connection position is normalized [-1, 1], map to surface
        let parent_half = parent_node.dimensions / 2.0;

        // Apply reflection to connection position
        let conn_pos = connection.position * reflection;

        // Clamp to surface of parent box
        let attach_local = Self::clamp_to_surface(conn_pos, parent_half);

        // Transform to world space
        let attach_world = parent_transform.transform_point(attach_local);

        // Apply connection orientation (with reflection)
        let orientation = if reflection.x * reflection.y * reflection.z < 0.0 {
            // Odd number of reflections - mirror the orientation
            Quat::from_xyzw(
                -connection.orientation.x,
                connection.orientation.y,
                connection.orientation.z,
                -connection.orientation.w,
            )
        } else {
            connection.orientation
        };

        let child_rotation = parent_transform.rotation * orientation;

        Transform {
            translation: attach_world,
            rotation: child_rotation,
            scale: Vec3::splat(connection.scale),
        }
    }

    fn clamp_to_surface(pos: Vec3, half_extents: Vec3) -> Vec3 {
        // Find which face we're closest to and clamp to it
        let scaled = pos * half_extents;
        let abs_scaled = scaled.abs();

        if abs_scaled.x >= abs_scaled.y && abs_scaled.x >= abs_scaled.z {
            // X face
            Vec3::new(half_extents.x * pos.x.signum(), scaled.y, scaled.z)
        } else if abs_scaled.y >= abs_scaled.z {
            // Y face
            Vec3::new(scaled.x, half_extents.y * pos.y.signum(), scaled.z)
        } else {
            // Z face
            Vec3::new(scaled.x, scaled.y, half_extents.z * pos.z.signum())
        }
    }

    fn create_joint(
        child_node: &MorphologyNode,
        parent_node: &MorphologyNode,
        connection: &MorphologyConnection,
    ) -> impl Into<TypedJoint> {
        let parent_half = parent_node.dimensions / 2.0;
        let child_half = child_node.dimensions / 2.0;

        // Anchor on parent surface
        let parent_anchor = Self::clamp_to_surface(connection.position, parent_half);

        // Anchor on child - opposite face from connection direction
        let conn_dir = connection.position.normalize_or_zero();
        let child_anchor = -conn_dir * child_half;

        // We use SphericalJoint for all types for now, as it's the most flexible
        // Different joint types can be simulated by adjusting motor targets
        SphericalJointBuilder::new()
            .local_anchor1(parent_anchor)
            .local_anchor2(child_anchor)
    }
}
