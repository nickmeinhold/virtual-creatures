//! Evolution system for virtual creatures.
//!
//! Implements:
//! - Random genotype generation
//! - Mutation operators
//! - Crossover and grafting
//! - Fitness evaluation
//! - Evolution loop with selection

use bevy::prelude::*;
use rand::prelude::*;

use crate::genotype::*;

/// Configuration for evolution
#[derive(Resource, Clone)]
pub struct EvolutionConfig {
    /// Population size
    pub population_size: usize,
    /// Fraction that survives each generation
    pub survival_ratio: f32,
    /// Probability of asexual reproduction
    pub asexual_prob: f32,
    /// Probability of crossover
    pub crossover_prob: f32,
    /// Probability of grafting
    pub grafting_prob: f32,
    /// Mutation rate for parameters
    pub mutation_rate: f32,
    /// Duration of each fitness test in seconds
    pub test_duration: f32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            survival_ratio: 0.2,
            asexual_prob: 0.4,
            crossover_prob: 0.3,
            grafting_prob: 0.3,
            mutation_rate: 0.3,
            test_duration: 10.0,
        }
    }
}

/// An individual in the population
#[derive(Clone)]
pub struct Individual {
    pub genotype: CreatureGenotype,
    pub fitness: f32,
}

/// Current state of evolution
#[derive(Resource)]
pub struct EvolutionState {
    pub population: Vec<Individual>,
    pub generation: usize,
    pub best_fitness: f32,
    pub current_individual: usize,
    pub test_start_time: f32,
    pub test_start_position: Vec3,
}

impl Default for EvolutionState {
    fn default() -> Self {
        Self {
            population: Vec::new(),
            generation: 0,
            best_fitness: 0.0,
            current_individual: 0,
            test_start_time: 0.0,
            test_start_position: Vec3::ZERO,
        }
    }
}

// ============================================================================
// Random Genotype Generation
// ============================================================================

/// Generate a random genotype
pub fn random_genotype(rng: &mut impl Rng) -> CreatureGenotype {
    // Random root node
    let root = random_morphology_node(rng, true);
    let mut genotype = CreatureGenotype::new(root);

    // Add 1-5 child parts
    let num_parts = rng.gen_range(1..=5);
    let mut parent_options = vec![genotype.root];

    for _ in 0..num_parts {
        let parent = *parent_options.choose(rng).unwrap();
        let node = random_morphology_node(rng, false);
        let connection = random_connection(rng);

        let child = genotype.add_part(parent, node, connection);
        parent_options.push(child);

        // Maybe add symmetric counterpart
        if rng.gen_bool(0.5) {
            let sym_node = random_morphology_node(rng, false);
            // Choose reflection axis - X is most common for bilateral symmetry
            let reflect_axis = match rng.gen_range(0..10) {
                0 => ReflectAxis::Y,  // Less common
                1 => ReflectAxis::Z,  // Less common
                _ => ReflectAxis::X,  // Most common - bilateral symmetry
            };
            let sym_conn = genotype.morphology.connections_from(parent)
                .last()
                .map(|c| c.data.reflected(reflect_axis))
                .unwrap_or_else(|| random_connection(rng));
            genotype.add_part(parent, sym_node, sym_conn);
        }
    }

    // Add sensors and neural oscillators to each part
    for (_node_id, node) in genotype.morphology.nodes_mut() {
        let dof = node.joint_type.dof();

        // Add joint angle sensors for each DOF
        for d in 0..dof {
            node.neural.add_sensor(SensorType::JointAngle { dof: d });
        }

        // Maybe add a photosensor
        if rng.gen_bool(0.3) {
            let axis = match rng.gen_range(0..3) {
                0 => SensorAxis::X,
                1 => SensorAxis::Y,
                _ => SensorAxis::Z,
            };
            node.neural.add_sensor(SensorType::PhotoSensor { axis });
        }

        // Maybe add contact sensors on different faces
        if rng.gen_bool(0.2) {
            let face = match rng.gen_range(0..6) {
                0 => Face::PosX,
                1 => Face::NegX,
                2 => Face::PosY,
                3 => Face::NegY,
                4 => Face::PosZ,
                _ => Face::NegZ,
            };
            node.neural.add_sensor(SensorType::Contact { face });
        }

        if node.joint_type != JointType::Rigid {
            // Add an oscillator neuron with random frequency
            let osc_func = if rng.gen_bool(0.7) {
                NeuronFunc::OscillateWave
            } else {
                NeuronFunc::OscillateSaw
            };

            let oscillator = Neuron {
                func: osc_func,
                inputs: vec![WeightedInput {
                    source: NeuralInput::Constant(rng.gen_range(0.5..3.0)),
                    weight: 1.0,
                }],
            };
            let osc_idx = node.neural.add_neuron(oscillator);

            // Maybe add a processing neuron that uses sensor input
            let output_idx = if !node.neural.sensors.is_empty() && rng.gen_bool(0.5) {
                // Add a neuron that modulates based on sensor
                let sensor_idx = rng.gen_range(0..node.neural.sensors.len());
                // Use the full range of neuron functions
                let proc_func = match rng.gen_range(0..21) {
                    0 => NeuronFunc::Sum,
                    1 => NeuronFunc::Product,
                    2 => NeuronFunc::Divide,
                    3 => NeuronFunc::SumThreshold,
                    4 => NeuronFunc::GreaterThan,
                    5 => NeuronFunc::SignOf,
                    6 => NeuronFunc::Min,
                    7 => NeuronFunc::Max,
                    8 => NeuronFunc::Abs,
                    9 => NeuronFunc::If,
                    10 => NeuronFunc::Interpolate,
                    11 => NeuronFunc::Sin,
                    12 => NeuronFunc::Cos,
                    13 => NeuronFunc::Atan,
                    14 => NeuronFunc::Log,
                    15 => NeuronFunc::Exp,
                    16 => NeuronFunc::Sigmoid,
                    17 => NeuronFunc::Integrate,
                    18 => NeuronFunc::Differentiate,
                    19 => NeuronFunc::Smooth,
                    _ => NeuronFunc::Memory,
                };

                // Build appropriate inputs based on function arity
                let inputs = match proc_func.num_inputs() {
                    1 => vec![
                        WeightedInput { source: NeuralInput::LocalNeuron(osc_idx), weight: 1.0 },
                    ],
                    3 => vec![
                        WeightedInput { source: NeuralInput::Sensor(sensor_idx), weight: 1.0 },
                        WeightedInput { source: NeuralInput::Constant(-1.0), weight: 1.0 },
                        WeightedInput { source: NeuralInput::LocalNeuron(osc_idx), weight: 1.0 },
                    ],
                    _ => vec![
                        WeightedInput { source: NeuralInput::LocalNeuron(osc_idx), weight: 1.0 },
                        WeightedInput { source: NeuralInput::Sensor(sensor_idx), weight: rng.gen_range(0.1..1.0) },
                    ],
                };

                let proc_neuron = Neuron { func: proc_func, inputs };
                node.neural.add_neuron(proc_neuron)
            } else {
                osc_idx
            };

            // Add effectors for each DOF, scaling by cross-section for realism
            let force_scale = node.max_cross_section();
            for d in 0..dof {
                let effector = Effector {
                    dof: d,
                    input: WeightedInput {
                        source: NeuralInput::LocalNeuron(output_idx),
                        weight: rng.gen_range(1.0..5.0),
                    },
                    max_force: rng.gen_range(50.0..200.0) * force_scale,
                };
                node.neural.add_effector(effector);
            }
        }
    }

    // Add central nervous system neurons for global coordination
    let num_parts = genotype.part_type_count();
    if num_parts > 1 && rng.gen_bool(0.5) {
        // Add a central oscillator that parts can reference
        let central_osc = Neuron {
            func: NeuronFunc::OscillateWave,
            inputs: vec![WeightedInput {
                source: NeuralInput::Constant(rng.gen_range(0.3..1.5)),
                weight: 1.0,
            }],
        };
        genotype.central_nervous_system.neurons.push(central_osc);

        // Maybe add a processing neuron that combines inputs
        if rng.gen_bool(0.3) {
            let proc_neuron = Neuron {
                func: NeuronFunc::Smooth,
                inputs: vec![WeightedInput {
                    source: NeuralInput::CentralNeuron(0),
                    weight: 1.0,
                }],
            };
            genotype.central_nervous_system.neurons.push(proc_neuron);
        }
    }

    // Add inter-part neural connections to some parts
    // This creates more complex coordination between body parts
    for (node_id, node) in genotype.morphology.nodes_mut() {
        if node_id.0 > 0 && !node.neural.neurons.is_empty() && rng.gen_bool(0.3) {
            // Add a neuron that receives input from parent
            let parent_input_neuron = Neuron {
                func: NeuronFunc::Sum,
                inputs: vec![
                    WeightedInput {
                        source: NeuralInput::ParentNeuron(0),
                        weight: rng.gen_range(0.5..1.5),
                    },
                    WeightedInput {
                        source: NeuralInput::LocalNeuron(0),
                        weight: rng.gen_range(0.5..1.5),
                    },
                ],
            };
            node.neural.add_neuron(parent_input_neuron);
        }

        // Maybe add child neuron input
        if rng.gen_bool(0.2) && !node.neural.neurons.is_empty() {
            let child_input_neuron = Neuron {
                func: NeuronFunc::Sum,
                inputs: vec![
                    WeightedInput {
                        source: NeuralInput::ChildNeuron { connection: 0, neuron: 0 },
                        weight: rng.gen_range(0.3..1.0),
                    },
                    WeightedInput {
                        source: NeuralInput::LocalNeuron(0),
                        weight: 1.0,
                    },
                ],
            };
            node.neural.add_neuron(child_input_neuron);
        }

        // Maybe reference central nervous system
        if !genotype.central_nervous_system.neurons.is_empty() && rng.gen_bool(0.4) {
            let cns_input_neuron = Neuron {
                func: NeuronFunc::Product,
                inputs: vec![
                    WeightedInput {
                        source: NeuralInput::CentralNeuron(0),
                        weight: 1.0,
                    },
                    WeightedInput {
                        source: NeuralInput::LocalNeuron(0),
                        weight: 1.0,
                    },
                ],
            };
            node.neural.add_neuron(cns_input_neuron);
        }
    }

    genotype
}

fn random_morphology_node(rng: &mut impl Rng, is_root: bool) -> MorphologyNode {
    let dimensions = Vec3::new(
        rng.gen_range(0.2..1.0),
        rng.gen_range(0.2..1.0),
        rng.gen_range(0.2..1.0),
    );

    let joint_type = if is_root {
        JointType::Rigid
    } else {
        match rng.gen_range(0..7) {
            0 => JointType::Revolute,
            1 => JointType::Twist,
            2 => JointType::Universal,
            3 => JointType::BendTwist,
            4 => JointType::TwistBend,
            5 => JointType::Spherical,
            _ => JointType::Revolute, // Default to simple revolute
        }
    };

    let mut node = MorphologyNode::new(dimensions, joint_type);
    node.recursive_limit = rng.gen_range(1..=3);
    node
}

fn random_connection(rng: &mut impl Rng) -> MorphologyConnection {
    let mut conn = MorphologyConnection::new();

    // Random position on parent surface
    conn.position = Vec3::new(
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
        rng.gen_range(-1.0..1.0),
    );

    // Random orientation
    conn.orientation = Quat::from_euler(
        EulerRot::XYZ,
        rng.gen_range(-0.5..0.5),
        rng.gen_range(-0.5..0.5),
        rng.gen_range(-0.5..0.5),
    );

    conn.scale = rng.gen_range(0.5..1.2);
    conn.terminal_only = rng.gen_bool(0.1);

    conn
}

// ============================================================================
// Mutation
// ============================================================================

/// Mutate a genotype in place
pub fn mutate(genotype: &mut CreatureGenotype, rng: &mut impl Rng, rate: f32) {
    // Scale mutation rate by complexity (consider both nodes and connections)
    let complexity = genotype.morphology.node_count() + genotype.morphology.connection_count();
    let scale = 1.0 / (complexity as f32).sqrt();
    let adjusted_rate = rate * scale;

    // Mutate each node
    for (_, node) in genotype.morphology.nodes_mut() {
        mutate_node(node, rng, adjusted_rate);
    }

    // Mutate connections
    for conn in genotype.morphology.connections_mut() {
        mutate_connection(&mut conn.data, rng, adjusted_rate);
    }

    // Maybe add a new part
    if rng.gen_bool((adjusted_rate * 0.2) as f64) && genotype.morphology.node_count() < 10 {
        let parents: Vec<_> = genotype.morphology.nodes().map(|(id, _)| id).collect();
        if let Some(&parent) = parents.choose(rng) {
            // Validate the parent node exists before adding part
            if genotype.morphology.is_valid(parent) {
                let new_node = random_morphology_node(rng, false);
                let new_conn = random_connection(rng);
                genotype.add_part(parent, new_node, new_conn);
            }
        }
    }

    // Maybe mutate a specific node's recursive limit using get_node_mut
    if rng.gen_bool((adjusted_rate * 0.1) as f64) {
        let node_id = NodeId(rng.gen_range(0..genotype.morphology.node_count()));
        if let Some(node) = genotype.morphology.get_node_mut(node_id) {
            node.recursive_limit = rng.gen_range(1..=4);
        }
    }

    // Maybe remove a part (not root)
    if rng.gen_bool((adjusted_rate * 0.1) as f64) && genotype.morphology.node_count() > 2 {
        // Note: actual removal would require more complex graph surgery
        // For now, we just reduce recursive_limit of a random node
        let _nodes: Vec<_> = genotype.morphology.nodes_mut()
            .filter(|(id, _)| *id != genotype.root)
            .collect();
        // Can't easily mutate here due to borrow, skip for now
    }
}

fn mutate_node(node: &mut MorphologyNode, rng: &mut impl Rng, rate: f32) {
    // Mutate dimensions
    if rng.gen_bool(rate as f64) {
        node.dimensions.x *= rng.gen_range(0.8..1.25);
        node.dimensions.x = node.dimensions.x.clamp(0.1, 2.0);
    }
    if rng.gen_bool(rate as f64) {
        node.dimensions.y *= rng.gen_range(0.8..1.25);
        node.dimensions.y = node.dimensions.y.clamp(0.1, 2.0);
    }
    if rng.gen_bool(rate as f64) {
        node.dimensions.z *= rng.gen_range(0.8..1.25);
        node.dimensions.z = node.dimensions.z.clamp(0.1, 2.0);
    }

    // Mutate joint type
    if rng.gen_bool((rate * 0.1) as f64) {
        node.joint_type = match rng.gen_range(0..5) {
            0 => JointType::Rigid,
            1 => JointType::Revolute,
            2 => JointType::Twist,
            3 => JointType::Universal,
            _ => JointType::Spherical,
        };
    }

    // Mutate recursive limit
    if rng.gen_bool((rate * 0.2) as f64) {
        node.recursive_limit = rng.gen_range(1..=4);
    }

    // Mutate neural parameters
    for neuron in &mut node.neural.neurons {
        for input in &mut neuron.inputs {
            if rng.gen_bool(rate as f64) {
                input.weight *= rng.gen_range(0.8..1.25);
                input.weight = input.weight.clamp(-10.0, 10.0);
            }
            // Mutate constant inputs
            if let NeuralInput::Constant(ref mut val) = input.source {
                if rng.gen_bool(rate as f64) {
                    *val *= rng.gen_range(0.8..1.25);
                    *val = val.clamp(0.1, 10.0);
                }
            }
        }
    }

    for effector in &mut node.neural.effectors {
        if rng.gen_bool(rate as f64) {
            effector.input.weight *= rng.gen_range(0.8..1.25);
            effector.input.weight = effector.input.weight.clamp(-10.0, 10.0);
        }
    }
}

fn mutate_connection(conn: &mut MorphologyConnection, rng: &mut impl Rng, rate: f32) {
    // Mutate position
    if rng.gen_bool(rate as f64) {
        conn.position.x += rng.gen_range(-0.2..0.2);
        conn.position.x = conn.position.x.clamp(-1.0, 1.0);
    }
    if rng.gen_bool(rate as f64) {
        conn.position.y += rng.gen_range(-0.2..0.2);
        conn.position.y = conn.position.y.clamp(-1.0, 1.0);
    }
    if rng.gen_bool(rate as f64) {
        conn.position.z += rng.gen_range(-0.2..0.2);
        conn.position.z = conn.position.z.clamp(-1.0, 1.0);
    }

    // Mutate orientation
    if rng.gen_bool(rate as f64) {
        let delta = Quat::from_euler(
            EulerRot::XYZ,
            rng.gen_range(-0.2..0.2),
            rng.gen_range(-0.2..0.2),
            rng.gen_range(-0.2..0.2),
        );
        conn.orientation = (conn.orientation * delta).normalize();
    }

    // Mutate scale
    if rng.gen_bool(rate as f64) {
        conn.scale *= rng.gen_range(0.9..1.1);
        conn.scale = conn.scale.clamp(0.3, 2.0);
    }
}

// ============================================================================
// Crossover and Grafting
// ============================================================================

/// Crossover: combine two genotypes by swapping node sequences
pub fn crossover(
    parent1: &CreatureGenotype,
    parent2: &CreatureGenotype,
    rng: &mut impl Rng,
) -> CreatureGenotype {
    // Simple crossover: take root from parent1, add some parts from parent2
    let mut child = parent1.clone();

    // Try to graft a random subtree from parent2
    if parent2.morphology.node_count() > 1 {
        let p2_nodes: Vec<_> = parent2.morphology.nodes()
            .filter(|(id, _)| *id != parent2.root)
            .collect();

        if let Some(&(node_id, node)) = p2_nodes.choose(rng) {
            // Get the connection for this node
            if let Some(conn) = parent2.morphology.connections_to(node_id).next() {
                // Add to a random parent in child
                let child_parents: Vec<_> = child.morphology.nodes().map(|(id, _)| id).collect();
                if let Some(&parent) = child_parents.choose(rng) {
                    child.add_part(parent, node.clone(), conn.data.clone());
                }
            }
        }
    }

    child
}

/// Grafting: attach a subtree from one genotype to another
pub fn graft(
    base: &CreatureGenotype,
    donor: &CreatureGenotype,
    rng: &mut impl Rng,
) -> CreatureGenotype {
    let mut child = base.clone();

    // Pick a random node from donor (not root)
    let donor_nodes: Vec<_> = donor.morphology.nodes()
        .filter(|(id, _)| *id != donor.root)
        .collect();

    if let Some(&(node_id, node)) = donor_nodes.choose(rng) {
        // Get connection data
        if let Some(conn) = donor.morphology.connections_to(node_id).next() {
            // Attach to random node in base
            let base_nodes: Vec<_> = child.morphology.nodes().map(|(id, _)| id).collect();
            if let Some(&parent) = base_nodes.choose(rng) {
                child.add_part(parent, node.clone(), conn.data.clone());
            }
        }
    }

    child
}

// ============================================================================
// Fitness Evaluation
// ============================================================================

/// Calculate fitness based on distance traveled
pub fn calculate_fitness(start_pos: Vec3, end_pos: Vec3, duration: f32) -> f32 {
    // Horizontal distance traveled (ignore Y to avoid rewarding falling)
    let horizontal_dist = Vec2::new(end_pos.x - start_pos.x, end_pos.z - start_pos.z).length();

    // Reward forward movement (positive X)
    let forward_dist = end_pos.x - start_pos.x;

    // Combine: mostly forward progress, some total distance
    let fitness = forward_dist.max(0.0) * 0.7 + horizontal_dist * 0.3;

    // Normalize by time
    fitness / duration.max(1.0)
}

// ============================================================================
// Evolution Loop
// ============================================================================

/// Initialize the population
pub fn init_population(config: &EvolutionConfig) -> Vec<Individual> {
    let mut rng = rand::thread_rng();
    (0..config.population_size)
        .map(|_| Individual {
            genotype: random_genotype(&mut rng),
            fitness: 0.0,
        })
        .collect()
}

/// Select survivors based on fitness
pub fn select_survivors(population: &mut Vec<Individual>, survival_ratio: f32) {
    // Sort by fitness (descending)
    population.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

    // Keep top fraction
    let num_survivors = ((population.len() as f32) * survival_ratio).ceil() as usize;
    population.truncate(num_survivors.max(1));
}

/// Reproduce to fill population back up
pub fn reproduce(population: &mut Vec<Individual>, target_size: usize, config: &EvolutionConfig) {
    let mut rng = rand::thread_rng();
    let survivors = population.clone();

    while population.len() < target_size {
        let roll: f32 = rng.gen();

        let child_genotype = if roll < config.asexual_prob {
            // Asexual: mutate a copy
            let parent = survivors.choose(&mut rng).unwrap();
            let mut child = parent.genotype.clone();
            mutate(&mut child, &mut rng, config.mutation_rate);
            child
        } else if roll < config.asexual_prob + config.crossover_prob {
            // Crossover
            let p1 = survivors.choose(&mut rng).unwrap();
            let p2 = survivors.choose(&mut rng).unwrap();
            let mut child = crossover(&p1.genotype, &p2.genotype, &mut rng);
            mutate(&mut child, &mut rng, config.mutation_rate * 0.5);
            child
        } else if roll < config.asexual_prob + config.crossover_prob + config.grafting_prob {
            // Grafting: take a subtree from one parent and attach to another
            let p1 = survivors.choose(&mut rng).unwrap();
            let p2 = survivors.choose(&mut rng).unwrap();
            let mut child = graft(&p1.genotype, &p2.genotype, &mut rng);
            mutate(&mut child, &mut rng, config.mutation_rate * 0.5);
            child
        } else {
            // Fallback: asexual reproduction
            let parent = survivors.choose(&mut rng).unwrap();
            let mut child = parent.genotype.clone();
            mutate(&mut child, &mut rng, config.mutation_rate);
            child
        };

        population.push(Individual {
            genotype: child_genotype,
            fitness: 0.0,
        });
    }
}

/// Run one generation of evolution
pub fn evolve_generation(state: &mut EvolutionState, config: &EvolutionConfig) {
    // Select survivors
    select_survivors(&mut state.population, config.survival_ratio);

    // Record best
    if let Some(best) = state.population.first() {
        state.best_fitness = best.fitness;
    }

    // Reproduce
    reproduce(&mut state.population, config.population_size, config);

    state.generation += 1;
    state.current_individual = 0;

    println!(
        "Generation {}: best fitness = {:.3}",
        state.generation, state.best_fitness
    );
}
