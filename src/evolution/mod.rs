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
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

use crate::genotype::*;

/// Tunables read once from the environment so we can sweep evolution regimes
/// without recompiling. Lets a single binary run several configs in parallel.
pub struct Tuning {
    /// Hard cap on distinct morphology nodes (part-types). Higher = wilder,
    /// more complex bodies become reachable.
    pub max_nodes: usize,
    /// Multiplier on the structural (add-part) mutation probability. >1 pushes
    /// against the 1/sqrt(complexity) damping so bodies actually grow.
    pub struct_boost: f64,
}

// Parse an env override, warning loudly on a malformed value rather than
// silently running the default — a typo'd sweep config should not quietly run
// the wrong experiment.
fn env_f64(key: &str, default: f64) -> f64 {
    match std::env::var(key) {
        Ok(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("WARNING: {key}='{v}' is not a valid number; using default {default}");
            default
        }),
        Err(_) => default,
    }
}
fn env_usize(key: &str, default: usize) -> usize {
    match std::env::var(key) {
        Ok(v) => v.parse().unwrap_or_else(|_| {
            eprintln!("WARNING: {key}='{v}' is not a valid integer; using default {default}");
            default
        }),
        Err(_) => default,
    }
}

pub fn tuning() -> &'static Tuning {
    static T: OnceLock<Tuning> = OnceLock::new();
    T.get_or_init(|| Tuning {
        max_nodes: env_usize("VC_MAXNODES", 6),
        struct_boost: env_f64("VC_ADDPART", 1.0),
    })
}

/// Configuration for evolution
#[derive(Resource, Clone)]
pub struct EvolutionConfig {
    /// Population size
    pub population_size: usize,
    /// Probability of asexual reproduction (vs crossover/grafting)
    pub asexual_prob: f32,
    /// Probability of crossover
    pub crossover_prob: f32,
    /// Mutation rate for parameters
    pub mutation_rate: f32,
    /// Duration of each fitness test in seconds
    pub test_duration: f32,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            // Larger population sustains many NEAT species at once, which both
            // escapes the slow-shuffle local optimum (more parallel exploration)
            // and fills more gallery slots (one champion is archived per species).
            population_size: env_usize("VC_POP", 100),
            asexual_prob: 0.4,
            crossover_prob: 0.3,
            mutation_rate: env_f64("VC_MUT", 0.3) as f32,
            test_duration: 10.0,
        }
    }
}

/// Monotonic counter for assigning globally unique gene IDs.
/// Each new gene (morphology node) created by mutation or random generation
/// gets the next ID. Inherited genes keep their original ID through cloning,
/// crossover, and grafting — this is how we track gene lineage.
#[derive(Debug, Clone)]
pub struct InnovationCounter(pub u64);

impl InnovationCounter {
    pub fn new() -> Self {
        Self(0)
    }

    /// Allocate the next unique innovation ID
    pub fn next(&mut self) -> u64 {
        let id = self.0;
        self.0 += 1;
        id
    }
}

// ============================================================================
// Speciation (NEAT-style)
// ============================================================================

/// Configuration for speciation
#[derive(Clone)]
pub struct SpeciationConfig {
    /// Weight for disjoint/excess gene count in compatibility distance
    pub disjoint_coeff: f32,
    /// Weight for average parameter difference in matching genes
    pub weight_diff_coeff: f32,
    /// Compatibility threshold — creatures below this distance are same species
    pub compatibility_threshold: f32,
    /// How many generations a species can stagnate before being eliminated
    pub stagnation_limit: usize,
    /// Probability of inheriting disjoint genes from the less-fit parent
    /// during crossover. Higher = more genetic diversity, lower = more
    /// conservative (only proven genes survive). 0.0-1.0.
    pub disjoint_inherit_prob: f64,
}

impl Default for SpeciationConfig {
    fn default() -> Self {
        Self {
            disjoint_coeff: 1.0,
            weight_diff_coeff: 0.4,
            // Low threshold to encourage speciation in small populations.
            // With ~5-12 genes per creature and disjoint_coeff=1.0, the
            // disjoint term alone typically ranges 0.1-0.8, so 0.5 ensures
            // creatures with meaningfully different topologies get separated.
            compatibility_threshold: env_f64("VC_COMPAT", 0.5) as f32,
            stagnation_limit: env_usize("VC_STAGNATION", 15),
            disjoint_inherit_prob: 0.3,
        }
    }
}

/// A species: a group of genetically similar creatures that compete within
/// their niche rather than globally. This protects novel topologies.
#[derive(Clone)]
pub struct Species {
    /// Representative genotype (random member from previous generation)
    pub representative: CreatureGenotype,
    /// Indices into the population vector
    pub members: Vec<usize>,
    /// Best fitness ever achieved by this species
    pub best_fitness: f32,
    /// Generation when best_fitness last improved
    pub last_improved: usize,
    /// Unique ID for this species
    pub id: usize,
}

/// Compute compatibility distance between two genotypes using innovation IDs.
///
/// The distance measures structural difference (how many genes are unshared)
/// and parametric difference (how different the shared genes' weights are).
pub fn compatibility_distance(
    g1: &CreatureGenotype,
    g2: &CreatureGenotype,
    config: &SpeciationConfig,
) -> f32 {
    use std::collections::HashMap;

    // Build innovation_id → node maps
    let genes1: HashMap<u64, &MorphologyNode> = g1.morphology.nodes()
        .map(|(_, node)| (node.innovation_id, node))
        .collect();
    let genes2: HashMap<u64, &MorphologyNode> = g2.morphology.nodes()
        .map(|(_, node)| (node.innovation_id, node))
        .collect();

    let mut matching = 0;
    let mut disjoint = 0;
    let mut weight_diff_sum = 0.0f32;

    // Check all genes in g1
    for (&id, node1) in &genes1 {
        if let Some(node2) = genes2.get(&id) {
            matching += 1;
            // Parameter difference: dimensions + joint type mismatch
            let dim_diff = (node1.dimensions - node2.dimensions).length();
            let joint_diff = if node1.joint_type != node2.joint_type { 1.0 } else { 0.0 };
            weight_diff_sum += dim_diff + joint_diff;
        } else {
            disjoint += 1;
        }
    }

    // Genes only in g2
    for id in genes2.keys() {
        if !genes1.contains_key(id) {
            disjoint += 1;
        }
    }

    // Normalize by the size of the larger genome
    let max_genes = genes1.len().max(genes2.len()).max(1) as f32;
    let avg_weight_diff = if matching > 0 { weight_diff_sum / matching as f32 } else { 0.0 };

    (config.disjoint_coeff * disjoint as f32 / max_genes)
        + (config.weight_diff_coeff * avg_weight_diff)
}

/// Assign each individual in the population to a species.
/// Returns the updated species list.
pub fn speciate(
    population: &[Individual],
    existing_species: &[Species],
    config: &SpeciationConfig,
    generation: usize,
    next_species_id: &mut usize,
) -> Vec<Species> {
    // Start with empty species, keeping representatives from last generation
    let mut species: Vec<Species> = existing_species.iter().map(|s| {
        Species {
            representative: s.representative.clone(),
            members: Vec::new(),
            best_fitness: s.best_fitness,
            last_improved: s.last_improved,
            id: s.id,
        }
    }).collect();

    // Assign each individual to the first compatible species
    for (idx, individual) in population.iter().enumerate() {
        let mut assigned = false;
        for s in &mut species {
            let dist = compatibility_distance(&individual.genotype, &s.representative, config);
            if dist < config.compatibility_threshold {
                s.members.push(idx);
                assigned = true;
                break;
            }
        }

        // No compatible species found — create a new one
        if !assigned {
            let id = *next_species_id;
            *next_species_id += 1;
            species.push(Species {
                representative: individual.genotype.clone(),
                members: vec![idx],
                best_fitness: 0.0,
                last_improved: generation,
                id,
            });
        }
    }

    // Remove empty species (went extinct)
    species.retain(|s| !s.members.is_empty());

    // Update representatives: random member from current generation
    let mut rng = rand::thread_rng();
    for s in &mut species {
        if let Some(&member_idx) = s.members.choose(&mut rng) {
            s.representative = population[member_idx].genotype.clone();
        }
    }

    species
}

/// Apply fitness sharing: divide each creature's fitness by its species size.
/// This prevents large species from dominating and gives small (novel) species
/// a fair chance to reproduce.
pub fn apply_fitness_sharing(population: &mut [Individual], species: &[Species]) {
    for s in species {
        let species_size = s.members.len() as f32;
        for &idx in &s.members {
            population[idx].fitness /= species_size;
        }
    }
}

/// Update species stagnation tracking after fitness evaluation
pub fn update_species_fitness(species: &mut [Species], population: &[Individual], generation: usize) {
    for s in species {
        let best_in_species = s.members.iter()
            .map(|&idx| population[idx].fitness)
            .fold(0.0f32, f32::max);

        if best_in_species > s.best_fitness {
            s.best_fitness = best_in_species;
            s.last_improved = generation;
        }
    }
}

/// Remove species that have stagnated (no fitness improvement for too long)
pub fn cull_stagnant_species(species: &mut Vec<Species>, generation: usize, stagnation_limit: usize) {
    // Always keep at least one species
    if species.len() <= 1 {
        return;
    }

    // Sort by best_fitness descending so the best species is first
    species.sort_by(|a, b| b.best_fitness.partial_cmp(&a.best_fitness).unwrap_or(std::cmp::Ordering::Equal));

    // Cull stagnant species, but always preserve the best one
    let best = species[0].clone();
    species.retain(|s| generation - s.last_improved < stagnation_limit);

    // Ensure at least the best species survives total extinction
    if species.is_empty() {
        species.push(best);
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
    /// Archive of best creatures (optional)
    pub archive: Option<CreatureArchive>,
    /// Path to save archive (if set, saves after each generation)
    pub save_path: Option<String>,
    /// Frames to wait before spawning first creature (let physics initialize)
    pub frames_before_spawn: u32,
    /// Global counter for assigning unique gene innovation IDs
    pub innovation_counter: InnovationCounter,
    /// Current species in the population
    pub species: Vec<Species>,
    /// Next species ID to assign
    pub next_species_id: usize,
    /// Speciation configuration
    pub speciation_config: SpeciationConfig,
    /// Gene analytics: tracks building-block genes across generations
    pub gene_tracker: GeneTracker,
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
            archive: Some(CreatureArchive::new()),
            // Env-overridable so parallel sweep runs don't clobber each other.
            save_path: Some(std::env::var("VC_OUT").unwrap_or_else(|_| "creatures.json".to_string())),
            frames_before_spawn: 2,
            innovation_counter: InnovationCounter::new(),
            species: Vec::new(),
            next_species_id: 0,
            speciation_config: SpeciationConfig::default(),
            gene_tracker: GeneTracker::new(),
        }
    }
}

// ============================================================================
// Random Genotype Generation
// ============================================================================

/// Generate a random genotype
pub fn random_genotype(rng: &mut impl Rng, counter: &mut InnovationCounter) -> CreatureGenotype {
    // Random root node
    let root = random_morphology_node(rng, true, counter);
    let mut genotype = CreatureGenotype::new(root);

    // Add 1-3 child parts (kept small — recursion can multiply these)
    let num_parts = rng.gen_range(1..=3);
    let mut parent_options = vec![genotype.root];

    for _ in 0..num_parts {
        let parent = *parent_options.choose(rng).unwrap();
        let node = random_morphology_node(rng, false, counter);
        let connection = random_connection(rng);

        let child = genotype.add_part(parent, node, connection);
        parent_options.push(child);

        // Maybe add symmetric counterpart
        if rng.gen_bool(0.5) {
            let sym_node = random_morphology_node(rng, false, counter);
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
                // Use the available neuron functions (17 total)
                let proc_func = match rng.gen_range(0..17) {
                    0 => NeuronFunc::Sum,
                    1 => NeuronFunc::Product,
                    2 => NeuronFunc::SumThreshold,
                    3 => NeuronFunc::GreaterThan,
                    4 => NeuronFunc::SignOf,
                    5 => NeuronFunc::Min,
                    6 => NeuronFunc::Max,
                    7 => NeuronFunc::Abs,
                    8 => NeuronFunc::If,
                    9 => NeuronFunc::Interpolate,
                    10 => NeuronFunc::Sin,
                    11 => NeuronFunc::Cos,
                    12 => NeuronFunc::Sigmoid,
                    13 => NeuronFunc::Integrate,
                    14 => NeuronFunc::Smooth,
                    15 => NeuronFunc::OscillateWave,
                    _ => NeuronFunc::OscillateSaw,
                };

                // Build appropriate inputs based on function arity
                let inputs = match proc_func.num_inputs() {
                    1 => vec![
                        WeightedInput { source: NeuralInput::Neuron { part: PartRef::Local, index: osc_idx }, weight: 1.0 },
                    ],
                    3 => vec![
                        WeightedInput { source: NeuralInput::Sensor(sensor_idx), weight: 1.0 },
                        WeightedInput { source: NeuralInput::Constant(-1.0), weight: 1.0 },
                        WeightedInput { source: NeuralInput::Neuron { part: PartRef::Local, index: osc_idx }, weight: 1.0 },
                    ],
                    _ => vec![
                        WeightedInput { source: NeuralInput::Neuron { part: PartRef::Local, index: osc_idx }, weight: 1.0 },
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
                        source: NeuralInput::Neuron { part: PartRef::Local, index: output_idx },
                        weight: rng.gen_range(1.0..5.0),
                    },
                    max_force: rng.gen_range(50.0..200.0) * force_scale,
                };
                node.neural.add_effector(effector);
            }
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
                        source: NeuralInput::Neuron { part: PartRef::Parent, index: 0 },
                        weight: rng.gen_range(0.5..1.5),
                    },
                    WeightedInput {
                        source: NeuralInput::Neuron { part: PartRef::Local, index: 0 },
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
                        source: NeuralInput::Neuron { part: PartRef::Child(0), index: 0 },
                        weight: rng.gen_range(0.3..1.0),
                    },
                    WeightedInput {
                        source: NeuralInput::Neuron { part: PartRef::Local, index: 0 },
                        weight: 1.0,
                    },
                ],
            };
            node.neural.add_neuron(child_input_neuron);
        }
    }

    genotype
}

fn random_morphology_node(rng: &mut impl Rng, is_root: bool, counter: &mut InnovationCounter) -> MorphologyNode {
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

    let mut node = MorphologyNode::new(counter.next(), dimensions, joint_type);
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
pub fn mutate(genotype: &mut CreatureGenotype, rng: &mut impl Rng, rate: f32, counter: &mut InnovationCounter) {
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

    // Maybe add a new part (new gene = new innovation ID). The structural boost
    // pushes against the 1/sqrt(complexity) damping so bodies can actually grow,
    // and the node cap is env-tunable to allow wilder morphologies.
    let add_part_prob = ((adjusted_rate as f64) * 0.2 * tuning().struct_boost).clamp(0.0, 1.0);
    if rng.gen_bool(add_part_prob) && genotype.morphology.node_count() < tuning().max_nodes {
        let parents: Vec<_> = genotype.morphology.nodes().map(|(id, _)| id).collect();
        if let Some(&parent) = parents.choose(rng) {
            // Validate the parent node exists before adding part
            if genotype.morphology.is_valid(parent) {
                let new_node = random_morphology_node(rng, false, counter);
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

/// NEAT-style aligned crossover using innovation IDs.
///
/// Genes (morphology nodes) are matched by their innovation_id:
/// - **Matching genes**: randomly inherit from either parent
/// - **Disjoint/excess genes**: inherited from the fitter parent (parent1)
///
/// The topology (connections) follows the inherited nodes. When a node is
/// taken from parent2, its connection data comes along, re-attached to the
/// closest valid parent in the child.
pub fn crossover(
    parent1: &CreatureGenotype,
    parent2: &CreatureGenotype,
    rng: &mut impl Rng,
    speciation_config: &SpeciationConfig,
) -> CreatureGenotype {
    use std::collections::HashMap;

    // Build innovation_id → (NodeId, node, connection_data, parent_innovation_id) maps.
    // The parent_innovation_id lets us preserve topology when assembling the child.
    fn build_gene_map(genotype: &CreatureGenotype) -> HashMap<u64, (NodeId, MorphologyNode, MorphologyConnection, Option<u64>)> {
        genotype.morphology.nodes().map(|(id, node)| {
            let conn = genotype.morphology.connections_to(id).next();
            let parent_innovation = conn.map(|c| genotype.morphology[c.from].innovation_id);
            let conn_data = conn.map(|c| c.data.clone()).unwrap_or_default();
            (node.innovation_id, (id, node.clone(), conn_data, parent_innovation))
        }).collect()
    }

    let p1_genes = build_gene_map(parent1);
    let p2_genes = build_gene_map(parent2);

    // Start with the root from parent1 (fitter parent)
    let root_node = parent1.root_node().clone();
    let mut child = CreatureGenotype::new(root_node);

    // Track which innovation IDs are in the child and their NodeId
    let root_innovation = parent1.root_node().innovation_id;
    let mut child_innovation_to_node: HashMap<u64, NodeId> = HashMap::new();
    child_innovation_to_node.insert(root_innovation, child.root);

    // Collect all innovation IDs from both parents (excluding root)
    let mut all_ids: Vec<u64> = p1_genes.keys()
        .chain(p2_genes.keys())
        .copied()
        .filter(|id| *id != root_innovation)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    all_ids.sort(); // Deterministic ordering by innovation ID

    for innovation_id in all_ids {
        // Decide which parent to take the gene from, and get its source info
        let (chosen_node, chosen_conn, parent_innov) = match (p1_genes.get(&innovation_id), p2_genes.get(&innovation_id)) {
            // Matching gene: randomly pick from either parent
            (Some((_, n1, c1, pi1)), Some((_, n2, c2, pi2))) => {
                if rng.gen_bool(0.5) {
                    (n1.clone(), c1.clone(), *pi1)
                } else {
                    (n2.clone(), c2.clone(), *pi2)
                }
            }
            // Disjoint/excess: only in fitter parent (parent1) — include
            (Some((_, node, conn, pi)), None) => {
                (node.clone(), conn.clone(), *pi)
            }
            // Only in parent2 (less fit) — include with configurable probability
            (None, Some((_, node, conn, pi))) => {
                if rng.gen_bool(speciation_config.disjoint_inherit_prob) {
                    (node.clone(), conn.clone(), *pi)
                } else {
                    continue;
                }
            }
            (None, None) => unreachable!(),
        };

        // Topology-preserving attachment: try to attach to the same parent
        // gene that this node was connected to in its source parent.
        // Fall back to random attachment if that parent isn't in the child.
        let attach_point = parent_innov
            .and_then(|pi| child_innovation_to_node.get(&pi).copied())
            .unwrap_or_else(|| {
                let child_nodes: Vec<_> = child.morphology.nodes().map(|(id, _)| id).collect();
                *child_nodes.choose(rng).unwrap_or(&child.root)
            });

        let new_node_id = child.add_part(attach_point, chosen_node, chosen_conn);
        child_innovation_to_node.insert(innovation_id, new_node_id);
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
pub fn calculate_fitness(
    start_pos: Vec3,
    end_pos: Vec3,
    duration: f32,
    peak_height: f32,
    max_part_speed: f32,
) -> f32 {
    // --- Physics-cheat disqualifiers -------------------------------------
    // A creature that exploits the solver (vibration energy-leak, launches)
    // isn't locomoting — it's a glitch, and a glitch in the gallery looks fake.
    // Disqualify outright (fitness 0) so it can't out-reproduce honest gaits.

    // 1. Solver-energy-leak / explosion: no legitimate gait whips a body part
    //    faster than this. Vibration exploits spike well past it.
    const MAX_PLAUSIBLE_PART_SPEED: f32 = 25.0; // m/s
    if !max_part_speed.is_finite() || max_part_speed > MAX_PLAUSIBLE_PART_SPEED {
        return 0.0;
    }

    // 2. Ballistic launch: anything that flings its center of mass this high
    //    is a projectile, not a walker — catch it even if it lands low again
    //    (which the end-of-run height check below would miss).
    const MAX_PLAUSIBLE_PEAK_HEIGHT: f32 = 5.0; // metres
    if !peak_height.is_finite() || peak_height > MAX_PLAUSIBLE_PEAK_HEIGHT {
        return 0.0;
    }

    // --- Honest locomotion score -----------------------------------------
    // Horizontal distance traveled in any direction (ignore Y to avoid rewarding falling)
    let horizontal_dist = Vec2::new(end_pos.x - start_pos.x, end_pos.z - start_pos.z).length();

    // Soft penalty for finishing airborne (in case it ends mid-hop just under
    // the hard cap above).
    let max_reasonable_height = 3.0;
    let height_penalty = (end_pos.y - max_reasonable_height).max(0.0);

    // Normalize by time to get speed, then subtract height penalty
    let speed = horizontal_dist / duration.max(1.0);
    (speed - height_penalty).max(0.0)
}

// ============================================================================
// Gene Analytics
// ============================================================================

/// Per-gene statistics tracked across generations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GeneStats {
    /// How many times this gene has appeared in the population (across all generations)
    pub total_appearances: usize,
    /// How many times this gene appeared in the top 20% of performers
    pub top_performer_appearances: usize,
    /// Running average fitness of individuals carrying this gene
    pub avg_fitness: f32,
    /// The generation this gene was first seen
    pub first_seen: usize,
    /// The generation this gene was last seen
    pub last_seen: usize,
}

/// Tracks gene frequency and fitness contribution across the entire evolutionary run.
/// This reveals which innovation IDs are "building blocks" — genes that consistently
/// appear in high-fitness individuals.
#[derive(Debug, Clone, Default)]
pub struct GeneTracker {
    pub stats: std::collections::HashMap<u64, GeneStats>,
}

impl GeneTracker {
    pub fn new() -> Self {
        Self { stats: std::collections::HashMap::new() }
    }

    /// Record one generation's worth of data
    pub fn record_generation(&mut self, population: &[Individual], generation: usize) {
        // Find the top 20% fitness threshold
        let mut fitnesses: Vec<f32> = population.iter().map(|i| i.fitness).collect();
        fitnesses.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let top_threshold = fitnesses.get(fitnesses.len() / 5).copied().unwrap_or(0.0);

        for individual in population {
            let is_top = individual.fitness >= top_threshold && top_threshold > 0.0;

            for (_, node) in individual.genotype.morphology.nodes() {
                let entry = self.stats.entry(node.innovation_id).or_insert(GeneStats {
                    first_seen: generation,
                    ..Default::default()
                });

                entry.total_appearances += 1;
                entry.last_seen = generation;

                // Incremental average: avg = avg + (new - avg) / n
                let n = entry.total_appearances as f32;
                entry.avg_fitness += (individual.fitness - entry.avg_fitness) / n;

                if is_top {
                    entry.top_performer_appearances += 1;
                }
            }
        }
    }

    /// Get the top N building-block genes ranked by frequency in top performers
    pub fn top_building_blocks(&self, n: usize) -> Vec<(u64, &GeneStats)> {
        let mut genes: Vec<_> = self.stats.iter().map(|(&id, stats)| (id, stats)).collect();
        genes.sort_by(|a, b| {
            b.1.top_performer_appearances.cmp(&a.1.top_performer_appearances)
                .then_with(|| b.1.avg_fitness.partial_cmp(&a.1.avg_fitness).unwrap_or(std::cmp::Ordering::Equal))
        });
        genes.truncate(n);
        genes
    }

    /// Print a summary of building-block genes
    pub fn print_summary(&self, top_n: usize) {
        let blocks = self.top_building_blocks(top_n);
        if blocks.is_empty() { return; }

        println!("  Top building-block genes:");
        for (id, stats) in blocks {
            let elite_rate = if stats.total_appearances > 0 {
                stats.top_performer_appearances as f32 / stats.total_appearances as f32 * 100.0
            } else {
                0.0
            };
            println!(
                "    Gene #{}: seen {}x, elite {:.0}%, avg fitness {:.3}, gens {}-{}",
                id, stats.total_appearances, elite_rate, stats.avg_fitness,
                stats.first_seen, stats.last_seen,
            );
        }
    }
}

// ============================================================================
// Surrogate Fitness Encoding
// ============================================================================

/// Features extracted from a single morphology node for the surrogate model.
/// This is the per-gene "slot" in the fixed-length feature vector.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct GeneFeatures {
    /// Whether this gene is present (1.0) or absent (0.0)
    pub present: f32,
    /// Volume of the body part (proxy for mass)
    pub volume: f32,
    /// Joint degrees of freedom (0-3)
    pub joint_dof: f32,
    /// Number of neurons in this part's neural circuit
    pub neuron_count: f32,
    /// Number of effectors (motor outputs)
    pub effector_count: f32,
    /// Number of sensors
    pub sensor_count: f32,
    /// Recursive limit (how many times this gene can repeat)
    pub recursive_limit: f32,
}

impl GeneFeatures {
    /// Number of f32 values per gene slot
    pub const DIMENSION: usize = 7;

    /// Convert to a flat array for model input
    pub fn to_array(&self) -> [f32; Self::DIMENSION] {
        [
            self.present,
            self.volume,
            self.joint_dof,
            self.neuron_count,
            self.effector_count,
            self.sensor_count,
            self.recursive_limit,
        ]
    }
}

/// Encode a genotype as a fixed-length feature vector for the surrogate model.
///
/// The vector has `counter.0 * GeneFeatures::DIMENSION` elements — one slot per
/// innovation ID ever issued. Each gene's innovation_id maps to its slot; absent
/// genes are zero-filled. Using the counter as the dimension guarantees no gene
/// is silently dropped.
#[allow(dead_code)]
pub fn encode_genotype(genotype: &CreatureGenotype, counter: &InnovationCounter) -> Vec<f32> {
    let num_slots = (counter.0 as usize).max(1);
    let mut features = vec![0.0f32; num_slots * GeneFeatures::DIMENSION];

    for (_, node) in genotype.morphology.nodes() {
        let slot = node.innovation_id as usize;
        if slot >= num_slots { continue; }

        let gene = GeneFeatures {
            present: 1.0,
            volume: node.volume(),
            joint_dof: node.joint_type.dof() as f32,
            neuron_count: node.neural.neurons.len() as f32,
            effector_count: node.neural.effectors.len() as f32,
            sensor_count: node.neural.sensors.len() as f32,
            recursive_limit: node.recursive_limit as f32,
        };

        let offset = slot * GeneFeatures::DIMENSION;
        let arr = gene.to_array();
        features[offset..offset + GeneFeatures::DIMENSION].copy_from_slice(&arr);
    }

    features
}

/// Encode an entire population into a matrix (one row per individual)
/// along with their fitness values. Ready for training a surrogate model.
/// Uses the innovation counter to determine vector width, ensuring all
/// genes that have ever existed have a stable slot.
#[allow(dead_code)]
pub fn encode_population(population: &[Individual], counter: &InnovationCounter) -> (Vec<Vec<f32>>, Vec<f32>) {
    let features: Vec<Vec<f32>> = population.iter()
        .map(|ind| encode_genotype(&ind.genotype, counter))
        .collect();
    let fitnesses: Vec<f32> = population.iter()
        .map(|ind| ind.fitness)
        .collect();
    (features, fitnesses)
}

// ============================================================================
// Evolution Loop
// ============================================================================

/// Initialize the population
pub fn init_population(config: &EvolutionConfig, counter: &mut InnovationCounter) -> Vec<Individual> {
    let mut rng = rand::thread_rng();
    (0..config.population_size)
        .map(|_| Individual {
            genotype: random_genotype(&mut rng, counter),
            fitness: 0.0,
        })
        .collect()
}

/// Run one generation of evolution
pub fn evolve_generation(state: &mut EvolutionState, config: &EvolutionConfig) {
    // Step 1: Speciate the population based on genetic similarity
    state.species = speciate(
        &state.population,
        &state.species,
        &state.speciation_config,
        state.generation,
        &mut state.next_species_id,
    );

    // Step 2: Update species fitness tracking (for stagnation detection)
    update_species_fitness(&mut state.species, &state.population, state.generation);

    // Step 3: Record gene analytics (before fitness sharing modifies values)
    state.gene_tracker.record_generation(&state.population, state.generation);

    // Step 3b: Record global best (before fitness sharing modifies values)
    // Sort by raw fitness to find the true best
    let global_best = state.population.iter()
        .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
        .cloned();

    if let Some(ref best) = global_best {
        state.best_fitness = best.fitness;
    }

    // Archive the champion of each species — one entry per distinct lineage —
    // so the saved gallery is behaviorally diverse rather than 10 near-clones
    // of whatever moves fastest. Read raw fitness here, BEFORE fitness sharing
    // (below) rescales it.
    if let Some(ref mut archive) = state.archive {
        for s in &state.species {
            // Champion = highest raw-fitness member of this species.
            let champion = s.members.iter()
                .map(|&idx| &state.population[idx])
                .max_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

            if let Some(champ) = champion {
                // Only archive lineages that actually do something — keeps
                // the barely-moving blobs out of the gallery. (Honest gaits
                // start slow now that physics-cheats are disqualified.)
                if champ.fitness < 0.04 { continue; }

                let part_count = count_spawned_parts(&champ.genotype);
                let saved = SavedCreature::new(
                    champ.genotype.clone(),
                    champ.fitness,
                    state.generation,
                    part_count,
                    s.id,
                );
                archive.upsert_species_champion(saved);
            }
        }

        // Cap to the strongest distinct lineages so the file stays manageable;
        // every survivor is still a different species, so diversity is kept.
        archive.keep_best(24);

        if let Some(ref path) = state.save_path {
            if let Err(e) = archive.save(path) {
                eprintln!("Warning: Failed to save creatures: {}", e);
            }
        }
    }

    // Step 4: Apply fitness sharing (divide by species size)
    // This ensures small novel species get a fair share of reproduction.
    apply_fitness_sharing(&mut state.population, &state.species);

    // Step 5: Cull stagnant species
    cull_stagnant_species(&mut state.species, state.generation, state.speciation_config.stagnation_limit);

    // Step 6: Reproduce proportionally to species' adjusted fitness
    let mut new_population = Vec::new();
    let mut rng = rand::thread_rng();

    // Calculate total adjusted fitness per species
    let species_fitness: Vec<f32> = state.species.iter().map(|s| {
        s.members.iter()
            .map(|&idx| state.population[idx].fitness.max(0.0))
            .sum::<f32>()
    }).collect();
    let total_fitness: f32 = species_fitness.iter().sum();

    // Each species gets offspring proportional to its share of total adjusted fitness
    for (s_idx, s) in state.species.iter().enumerate() {
        if s.members.is_empty() { continue; }

        // Number of offspring for this species
        let offspring_count = if total_fitness > 0.0 {
            ((species_fitness[s_idx] / total_fitness) * config.population_size as f32).round() as usize
        } else {
            // Equal distribution if all fitness is zero (e.g., first generation)
            config.population_size / state.species.len()
        };

        // Get this species' members sorted by fitness (descending)
        let mut species_members: Vec<&Individual> = s.members.iter()
            .map(|&idx| &state.population[idx])
            .collect();
        species_members.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Keep the champion of each species (elitism)
        if let Some(&champion) = species_members.first() {
            new_population.push(Individual {
                genotype: champion.genotype.clone(),
                fitness: 0.0,
            });
        }

        // Fill remaining slots with offspring
        for _ in 1..offspring_count {
            let roll: f32 = rng.gen();

            let child_genotype = if roll < config.asexual_prob {
                let parent = species_members.choose(&mut rng).unwrap();
                let mut child = parent.genotype.clone();
                mutate(&mut child, &mut rng, config.mutation_rate, &mut state.innovation_counter);
                child
            } else if roll < config.asexual_prob + config.crossover_prob {
                let p1 = species_members.choose(&mut rng).unwrap();
                let p2 = species_members.choose(&mut rng).unwrap();
                let mut child = crossover(&p1.genotype, &p2.genotype, &mut rng, &state.speciation_config);
                mutate(&mut child, &mut rng, config.mutation_rate * 0.5, &mut state.innovation_counter);
                child
            } else {
                let p1 = species_members.choose(&mut rng).unwrap();
                let p2 = species_members.choose(&mut rng).unwrap();
                let mut child = graft(&p1.genotype, &p2.genotype, &mut rng);
                mutate(&mut child, &mut rng, config.mutation_rate * 0.5, &mut state.innovation_counter);
                child
            };

            new_population.push(Individual {
                genotype: child_genotype,
                fitness: 0.0,
            });
        }
    }

    // Ensure we hit exact population size (rounding can leave us short or over)
    while new_population.len() < config.population_size {
        // Fill with mutated copies of random existing members
        if let Some(parent) = new_population.choose(&mut rng).cloned() {
            let mut child = parent.genotype;
            mutate(&mut child, &mut rng, config.mutation_rate, &mut state.innovation_counter);
            new_population.push(Individual { genotype: child, fitness: 0.0 });
        }
    }
    new_population.truncate(config.population_size);

    state.population = new_population;
    state.generation += 1;
    state.current_individual = 0;

    println!(
        "Generation {}: best fitness = {:.3}, species = {}",
        state.generation, state.best_fitness, state.species.len()
    );

    // Print building block summary every 10 generations
    if state.generation % 10 == 0 {
        state.gene_tracker.print_summary(5);
    }
}

/// Count how many parts would be spawned from a genotype
fn count_spawned_parts(genotype: &CreatureGenotype) -> usize {
    use std::collections::HashMap;

    fn count_recursive(
        genotype: &CreatureGenotype,
        node_id: NodeId,
        instances: &mut HashMap<NodeId, usize>,
        depth: usize,
    ) -> usize {
        if depth >= 10 {
            return 0;
        }

        let mut count = 1; // This node

        for conn in genotype.morphology.connections_from(node_id) {
            let child_node = &genotype.morphology[conn.to];
            let instance_count = instances.get(&conn.to).copied().unwrap_or(0);

            if instance_count >= child_node.recursive_limit as usize {
                continue;
            }

            let at_terminal = instance_count + 1 >= child_node.recursive_limit as usize;
            if conn.data.terminal_only && !at_terminal {
                continue;
            }

            *instances.entry(conn.to).or_insert(0) += 1;
            count += count_recursive(genotype, conn.to, instances, depth + 1);
        }

        count
    }

    let mut instances = HashMap::new();
    instances.insert(genotype.root, 1);
    count_recursive(genotype, genotype.root, &mut instances, 0)
}
