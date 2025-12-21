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
use crate::phenotype::CreaturePart;

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
    /// Neuron output values (indexed by part instance, then neuron index)
    pub neuron_outputs: HashMap<usize, Vec<f32>>,
    /// Internal state for stateful neurons (oscillators, integrators, etc.)
    pub neuron_state: HashMap<(usize, usize), f32>,
    /// Simulation time for oscillators
    pub time: f32,
    /// Central neuron outputs
    pub central_outputs: Vec<f32>,
    /// Central neuron state
    pub central_state: HashMap<usize, f32>,
}

impl Brain {
    pub fn new() -> Self {
        Self {
            neuron_outputs: HashMap::new(),
            neuron_state: HashMap::new(),
            time: 0.0,
            central_outputs: Vec::new(),
            central_state: HashMap::new(),
        }
    }
}

impl Default for Brain {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluate a single neuron function
fn evaluate_neuron(
    func: NeuronFunc,
    inputs: &[f32],
    state: &mut f32,
    time: f32,
    dt: f32,
) -> f32 {
    match func {
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
    }
}

/// System to run all creature brains
fn run_brains(
    time: Res<Time>,
    mut brains: Query<&mut Brain>,
    mut joints: Query<(&CreaturePart, &mut ImpulseJoint)>,
) {
    let dt = time.delta_secs();

    for mut brain in brains.iter_mut() {
        brain.time += dt;

        // For now, just run simple oscillators to make creatures move
        // Full neural network evaluation would go here

        // Apply simple oscillating forces to all joints
        let osc = (brain.time * 3.0).sin();

        for (_part, mut joint) in joints.iter_mut() {
            // Apply motor to the joint based on oscillation
            let motor_velocity = osc * 5.0;

            // Set motor on the underlying generic joint
            let raw = joint.data.as_mut();
            raw.set_motor_velocity(JointAxis::AngX, motor_velocity, 0.8);
            raw.set_motor_max_force(JointAxis::AngX, 100.0);
        }
    }
}
