//! Brain simulation - runs the neural control system.
//!
//! Each creature has a brain that:
//! - Reads sensor values from physics
//! - Evaluates neurons to compute outputs
//! - Applies effector forces to joints

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::collections::HashMap;

use crate::genotype::*;
use crate::phenotype::{CreatureBody, CreaturePart};

/// Safely get direction vectors from a transform, returning defaults if NaN
struct SafeTransform {
    translation: Vec3,
    right: Vec3,
    up: Vec3,
    forward: Vec3,
}

impl SafeTransform {
    fn from_global(transform: &GlobalTransform) -> Self {
        let translation = transform.translation();

        // Check for NaN in translation - if so, return safe defaults
        if translation.is_nan() {
            return Self {
                translation: Vec3::ZERO,
                right: Vec3::X,
                up: Vec3::Y,
                forward: Vec3::NEG_Z,
            };
        }

        // Try to get rotation, use identity if invalid
        let rotation = transform.rotation();
        if rotation.is_nan() {
            return Self {
                translation,
                right: Vec3::X,
                up: Vec3::Y,
                forward: Vec3::NEG_Z,
            };
        }

        // Compute direction vectors manually to avoid panic on denormalized quaternions
        let right = rotation * Vec3::X;
        let up = rotation * Vec3::Y;
        let forward = rotation * Vec3::NEG_Z;

        Self {
            translation,
            right: if right.is_nan() { Vec3::X } else { right },
            up: if up.is_nan() { Vec3::Y } else { up },
            forward: if forward.is_nan() { Vec3::NEG_Z } else { forward },
        }
    }
}

/// Plugin for brain simulation
pub struct BrainPlugin;

impl Plugin for BrainPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Update, run_brains);
    }
}

/// Runtime state for a creature's brain
#[derive(Component)]
pub struct Brain {
    /// The genotype this brain is running
    pub genotype: CreatureGenotype,
    /// Neuron output values (indexed by (node_id, instance), then neuron index)
    pub neuron_outputs: HashMap<(usize, usize), Vec<f32>>,
    /// Internal state for stateful neurons (keyed by (node_id, instance, neuron_idx))
    pub neuron_state: HashMap<(usize, usize, usize), f32>,
    /// Simulation time for oscillators
    pub time: f32,
    /// Central neuron outputs
    pub central_outputs: Vec<f32>,
    /// Central neuron state (keyed by neuron_idx)
    pub central_state: HashMap<usize, f32>,
    /// Sensor values per part (keyed by (node_id, instance), then sensor index)
    pub sensor_values: HashMap<(usize, usize), Vec<f32>>,
}

impl Brain {
    pub fn new(genotype: CreatureGenotype) -> Self {
        // Initialize central outputs
        let central_count = genotype.central_nervous_system.neurons.len();
        Self {
            genotype,
            neuron_outputs: HashMap::new(),
            neuron_state: HashMap::new(),
            time: 0.0,
            central_outputs: vec![0.0; central_count],
            central_state: HashMap::new(),
            sensor_values: HashMap::new(),
        }
    }
}

/// Sanitize a float value, replacing NaN/Inf with 0
fn sanitize(v: f32) -> f32 {
    if v.is_nan() || v.is_infinite() { 0.0 } else { v }
}

/// Evaluate a single neuron function
fn evaluate_neuron(
    func: NeuronFunc,
    inputs: &[f32],
    state: &mut f32,
    time: f32,
    dt: f32,
) -> f32 {
    // Sanitize state if it became NaN
    if state.is_nan() || state.is_infinite() {
        *state = 0.0;
    }

    // Sanitize inputs
    let inputs: Vec<f32> = inputs.iter().map(|&x| sanitize(x)).collect();

    let result = match func {
        NeuronFunc::Sum => inputs.iter().sum(),
        NeuronFunc::Product => inputs.iter().product(),
        NeuronFunc::Divide => {
            if inputs.len() >= 2 && inputs[1].abs() > 0.001 {
                inputs[0] / inputs[1]
            } else {
                0.0
            }
        }
        NeuronFunc::SumThreshold => {
            let sum: f32 = inputs.iter().sum();
            if sum > 0.0 { 1.0 } else { -1.0 }
        }
        NeuronFunc::GreaterThan => {
            if inputs.len() >= 2 && inputs[0] > inputs[1] {
                1.0
            } else {
                -1.0
            }
        }
        NeuronFunc::SignOf => {
            if inputs.is_empty() {
                0.0
            } else if inputs[0] > 0.0 {
                1.0
            } else if inputs[0] < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        NeuronFunc::Min => inputs.iter().cloned().fold(f32::MAX, f32::min),
        NeuronFunc::Max => inputs.iter().cloned().fold(f32::MIN, f32::max),
        NeuronFunc::Abs => inputs.first().map(|x| x.abs()).unwrap_or(0.0),
        NeuronFunc::If => {
            if inputs.len() >= 3 {
                if inputs[0] > 0.0 { inputs[1] } else { inputs[2] }
            } else {
                0.0
            }
        }
        NeuronFunc::Interpolate => {
            if inputs.len() >= 3 {
                let t = (inputs[0] + 1.0) / 2.0; // map [-1, 1] to [0, 1]
                inputs[1] * (1.0 - t) + inputs[2] * t
            } else {
                0.0
            }
        }
        NeuronFunc::Sin => inputs.first().map(|x| x.sin()).unwrap_or(0.0),
        NeuronFunc::Cos => inputs.first().map(|x| x.cos()).unwrap_or(0.0),
        NeuronFunc::Atan => inputs.first().map(|x| x.atan()).unwrap_or(0.0),
        NeuronFunc::Log => inputs.first().map(|x| x.abs().max(0.001).ln()).unwrap_or(0.0),
        NeuronFunc::Exp => inputs.first().map(|x| x.clamp(-10.0, 10.0).exp()).unwrap_or(1.0),
        NeuronFunc::Sigmoid => {
            inputs.first().map(|x| 1.0 / (1.0 + (-x).exp())).unwrap_or(0.5)
        }
        NeuronFunc::Integrate => {
            let input = inputs.first().copied().unwrap_or(0.0);
            *state += input * dt;
            *state = state.clamp(-10.0, 10.0);
            *state
        }
        NeuronFunc::Differentiate => {
            let input = inputs.first().copied().unwrap_or(0.0);
            let diff = (input - *state) / dt.max(0.001);
            *state = input;
            diff.clamp(-10.0, 10.0)
        }
        NeuronFunc::Smooth => {
            let input = inputs.first().copied().unwrap_or(0.0);
            let alpha = 0.1; // smoothing factor
            *state = *state * (1.0 - alpha) + input * alpha;
            *state
        }
        NeuronFunc::Memory => {
            let input = inputs.first().copied().unwrap_or(0.0);
            if input.abs() > 0.5 {
                *state = input;
            }
            *state
        }
        NeuronFunc::OscillateWave => {
            let freq = inputs.first().copied().unwrap_or(1.0).abs().max(0.1);
            (time * freq * std::f32::consts::TAU).sin()
        }
        NeuronFunc::OscillateSaw => {
            let freq = inputs.first().copied().unwrap_or(1.0).abs().max(0.1);
            let phase = (time * freq) % 1.0;
            phase * 2.0 - 1.0
        }
    };

    // Final sanitization of result
    sanitize(result)
}

/// Tracks parent-child relationships for neural connections
struct PartRelations {
    /// Parent node id for each node (None for root)
    parent: HashMap<usize, usize>,
    /// Child node ids for each node, indexed by connection order
    children: HashMap<usize, Vec<usize>>,
}

/// Get input value for a neuron from its source
fn get_input_value(
    input: &NeuralInput,
    weight: f32,
    part_key: (usize, usize),
    sensor_values: &HashMap<(usize, usize), Vec<f32>>,
    neuron_outputs: &HashMap<(usize, usize), Vec<f32>>,
    central_outputs: &[f32],
    relations: &PartRelations,
) -> f32 {
    let raw = match input {
        NeuralInput::Constant(v) => *v,
        NeuralInput::Sensor(idx) => {
            sensor_values.get(&part_key)
                .and_then(|s| s.get(*idx))
                .copied()
                .unwrap_or(0.0)
        }
        NeuralInput::LocalNeuron(idx) => {
            neuron_outputs.get(&part_key)
                .and_then(|n| n.get(*idx))
                .copied()
                .unwrap_or(0.0)
        }
        NeuralInput::ParentNeuron(idx) => {
            // Get parent node and look up its neuron output
            relations.parent.get(&part_key.0)
                .and_then(|parent_node| {
                    // Use instance 0 for parent (simplified)
                    neuron_outputs.get(&(*parent_node, 0))
                })
                .and_then(|n| n.get(*idx))
                .copied()
                .unwrap_or(0.0)
        }
        NeuralInput::ChildNeuron { connection, neuron } => {
            // Get child at specified connection index
            relations.children.get(&part_key.0)
                .and_then(|children| children.get(*connection))
                .and_then(|child_node| {
                    // Use instance 0 for child (simplified)
                    neuron_outputs.get(&(*child_node, 0))
                })
                .and_then(|n| n.get(*neuron))
                .copied()
                .unwrap_or(0.0)
        }
        NeuralInput::CentralNeuron(idx) => {
            central_outputs.get(*idx).copied().unwrap_or(0.0)
        }
    };
    raw * weight
}

/// System to run all creature brains
fn run_brains(
    time: Res<Time>,
    mut brains: Query<(Entity, &mut Brain, &CreatureBody)>,
    mut parts_query: Query<(&CreaturePart, &mut ImpulseJoint, &GlobalTransform)>,
) {
    let dt = time.delta_secs();

    for (creature_entity, mut brain, body) in brains.iter_mut() {
        brain.time += dt;
        let brain_time = brain.time;

        // Collect part info for this creature
        let mut part_joints: HashMap<(usize, usize), Entity> = HashMap::new();

        for &part_entity in &body.parts {
            if let Ok((part, _, _)) = parts_query.get(part_entity) {
                if part.creature_id == creature_entity {
                    let key = (part.node_id.0, part.instance);
                    part_joints.insert(key, part_entity);
                }
            }
        }

        // Build parent-child relationships from morphology graph
        let mut relations = PartRelations {
            parent: HashMap::new(),
            children: HashMap::new(),
        };
        for conn in brain.genotype.morphology.connections() {
            // Record parent relationship
            relations.parent.insert(conn.to.0, conn.from.0);
            // Record child relationship (append to children list)
            relations.children.entry(conn.from.0).or_default().push(conn.to.0);
        }

        // Read sensor values for each part
        for (&part_key, &part_entity) in &part_joints {
            let node_id = NodeId(part_key.0);
            if let Some(node) = brain.genotype.morphology.get_node(node_id) {
                let mut sensors = Vec::new();

                for sensor in &node.neural.sensors {
                    let value = match sensor {
                        SensorType::JointAngle { dof } => {
                            // Return joint motor velocity for the specified DOF
                            // In a full implementation, we'd track actual joint angles
                            // For now, use position-based approximation from transform
                            if let Ok((_, joint, transform)) = parts_query.get(part_entity) {
                                let axis = match dof {
                                    0 => JointAxis::AngX,
                                    1 => JointAxis::AngY,
                                    _ => JointAxis::AngZ,
                                };
                                // Get motor target velocity as proxy for angle
                                let motor = joint.data.as_ref().motor(axis);
                                motor.map(|m| m.target_vel).unwrap_or_else(|| {
                                    // Fallback: use orientation component
                                    let rotation = transform.rotation();
                                    if rotation.is_nan() {
                                        0.0
                                    } else {
                                        let euler = rotation.to_euler(EulerRot::XYZ);
                                        let angle = match dof {
                                            0 => euler.0,
                                            1 => euler.1,
                                            _ => euler.2,
                                        };
                                        if angle.is_nan() { 0.0 } else { angle }
                                    }
                                })
                            } else {
                                0.0
                            }
                        }
                        SensorType::Contact { face } => {
                            // Contact sensing based on face direction
                            // Use simplified ground contact detection
                            if let Ok((_, _, transform)) = parts_query.get(part_entity) {
                                let safe = SafeTransform::from_global(transform);
                                // Check if face is pointing downward and close to ground
                                let face_normal = match face {
                                    Face::PosX => safe.right,
                                    Face::NegX => -safe.right,
                                    Face::PosY => safe.up,
                                    Face::NegY => -safe.up,
                                    Face::PosZ => safe.forward,
                                    Face::NegZ => -safe.forward,
                                };
                                // Activate if face is pointing down and part is near ground
                                if face_normal.y < -0.5 && safe.translation.y < 0.5 {
                                    1.0
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            }
                        }
                        SensorType::PhotoSensor { axis } => {
                            // Simplified photosensor: detect light direction
                            // Returns how much the specified axis points toward light (up)
                            if let Ok((_, _, transform)) = parts_query.get(part_entity) {
                                let safe = SafeTransform::from_global(transform);
                                match axis {
                                    SensorAxis::X => safe.up.x,
                                    SensorAxis::Y => safe.up.y,
                                    SensorAxis::Z => safe.up.z,
                                }
                            } else {
                                0.0
                            }
                        }
                    };
                    sensors.push(value);
                }

                brain.sensor_values.insert(part_key, sensors);
            }
        }

        // Evaluate central nervous system first
        // Clone the CNS neurons to avoid borrow issues
        let cns_neurons: Vec<_> = brain.genotype.central_nervous_system.neurons.clone();
        let mut new_central_outputs = vec![0.0; cns_neurons.len()];

        for (neuron_idx, neuron) in cns_neurons.iter().enumerate() {
            let mut inputs = Vec::new();
            for weighted_input in &neuron.inputs {
                let value = get_input_value(
                    &weighted_input.source,
                    weighted_input.weight,
                    (0, 0), // Central neurons don't have a part
                    &brain.sensor_values,
                    &brain.neuron_outputs,
                    &brain.central_outputs,
                    &relations,
                );
                inputs.push(value);
            }

            let state = brain.central_state.entry(neuron_idx).or_insert(0.0);
            new_central_outputs[neuron_idx] = evaluate_neuron(
                neuron.func,
                &inputs,
                state,
                brain_time,
                dt,
            );
        }
        brain.central_outputs = new_central_outputs;

        // Evaluate each part's neural network
        // First collect what we need to avoid borrow issues
        let part_neurons: Vec<_> = part_joints.keys()
            .filter_map(|&part_key| {
                let node_id = NodeId(part_key.0);
                brain.genotype.morphology.get_node(node_id)
                    .map(|node| (part_key, node.neural.neurons.clone()))
            })
            .collect();

        for (part_key, neurons) in part_neurons {
            let mut outputs = vec![0.0; neurons.len()];

            // Evaluate neurons in order (assumes topological ordering)
            for (neuron_idx, neuron) in neurons.iter().enumerate() {
                let mut inputs = Vec::new();
                for weighted_input in &neuron.inputs {
                    let value = get_input_value(
                        &weighted_input.source,
                        weighted_input.weight,
                        part_key,
                        &brain.sensor_values,
                        &brain.neuron_outputs,
                        &brain.central_outputs,
                        &relations,
                    );
                    inputs.push(value);
                }

                let state_key = (part_key.0, part_key.1, neuron_idx);
                let state = brain.neuron_state.entry(state_key).or_insert(0.0);
                outputs[neuron_idx] = evaluate_neuron(
                    neuron.func,
                    &inputs,
                    state,
                    brain_time,
                    dt,
                );
            }

            brain.neuron_outputs.insert(part_key, outputs);
        }

        // Apply effector outputs to joints
        for (&part_key, &part_entity) in &part_joints {
            let node_id = NodeId(part_key.0);
            if let Some(node) = brain.genotype.morphology.get_node(node_id) {
                if let Ok((_, mut joint, _)) = parts_query.get_mut(part_entity) {
                    for effector in &node.neural.effectors {
                        // Get effector input value
                        let value = get_input_value(
                            &effector.input.source,
                            effector.input.weight,
                            part_key,
                            &brain.sensor_values,
                            &brain.neuron_outputs,
                            &brain.central_outputs,
                            &relations,
                        );

                        // Apply to the appropriate joint DOF
                        let axis = match effector.dof {
                            0 => JointAxis::AngX,
                            1 => JointAxis::AngY,
                            _ => JointAxis::AngZ,
                        };

                        // Set motor velocity based on effector output
                        // Sanitize and clamp to prevent physics instability
                        let motor_velocity = sanitize(value * 5.0).clamp(-50.0, 50.0);
                        let max_force = sanitize(effector.max_force).clamp(0.0, 1000.0);
                        let raw = joint.data.as_mut();
                        raw.set_motor_velocity(axis, motor_velocity, 0.8);
                        raw.set_motor_max_force(axis, max_force);
                    }
                }
            }
        }
    }
}
