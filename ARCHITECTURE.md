# Virtual Creatures Architecture

This document describes the architecture of the Virtual Creatures project, an implementation extending Karl Sims' 1994 paper "Evolving Virtual Creatures."

## Overview

The project uses a genetic algorithm to evolve both the body morphology and neural control systems of creatures that are simulated in a 3D physics environment. Creatures are represented as directed graphs (genotypes) that get instantiated into Bevy/Rapier physics entities (phenotypes), evaluated on forward walking speed, and evolved through mutation, crossover, and grafting operations.

**Tech Stack:**
- Rust
- Bevy 0.15 (ECS game engine)
- Rapier 0.28 (physics engine via bevy_rapier3d)
- Serde (serialization)

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    EVOLUTION                             │
│  random_genotype() → mutate() → crossover() → graft()   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              GENOTYPE (CreatureGenotype)                 │
│  morphology: DirectedGraph<MorphologyNode, Connection>   │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼  PhenotypeBuilder::spawn()
┌─────────────────────────────────────────────────────────┐
│           PHENOTYPE (Bevy/Rapier Entities)               │
│  RigidBody + Collider + ImpulseJoint + CreaturePart      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  BRAIN (Neural Control)                  │
│  Sensors → Neurons → Effectors → Joint Motors            │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
src/
├── main.rs                 # Entry point, simulation loop, Bevy app setup
├── genotype/               # Genetic information representation
│   ├── mod.rs              # Module exports
│   ├── graph.rs            # Generic directed graph data structure
│   ├── morphology.rs       # Body structure (nodes=parts, connections=joints)
│   ├── neural.rs           # Neural circuit definitions
│   └── io.rs               # Save/load functionality
├── phenotype/              # Instantiation of genotypes to physics
│   ├── mod.rs              # Module exports
│   └── builder.rs          # Genotype → Bevy/Rapier entity spawning
├── brain/                  # Neural simulation and motor control
│   └── mod.rs              # Brain component, neuron evaluation
└── evolution/              # Genetic algorithm
    └── mod.rs              # Population, selection, reproduction, mutation
```

---

## Genome Structure

The genome is defined in `src/genotype/` with three key layers:

### CreatureGenotype

The top-level structure representing a complete creature:

```rust
pub struct CreatureGenotype {
    pub morphology: DirectedGraph<MorphologyNode, MorphologyConnection>,
    pub root: NodeId,
}
```

- **morphology**: A directed graph where nodes are body parts and edges are joint attachments
- **root**: The starting node (typically the torso/body)

### DirectedGraph

An arena-based graph structure (`graph.rs`) that avoids `Rc<RefCell<>>` complexity:

```rust
pub struct DirectedGraph<N, C> {
    nodes: Vec<N>,                    // Arena of nodes
    connections: Vec<Connection<C>>,  // All edges
}

pub struct NodeId(pub usize);         // Type-safe index

pub struct Connection<C> {
    pub from: NodeId,
    pub to: NodeId,
    pub data: C,
}
```

Key methods:
- `add_node(node) -> NodeId` - Add node and return its ID
- `add_connection(from, to, data)` - Add directed edge
- `connections_from(id)` - Get outgoing edges
- `connections_to(id)` - Get incoming edges

### MorphologyNode (Body Parts)

Each node represents a physical body part:

```rust
pub struct MorphologyNode {
    pub dimensions: Vec3,          // Box size (x, y, z)
    pub joint_type: JointType,     // How this part connects to parent
    pub joint_limits: JointLimits, // Min/max rotation per DOF
    pub recursive_limit: u8,       // Max instantiations if cyclic
    pub neural: NeuralGraph,       // Local brain for this part
}
```

### Joint Types

Based on Karl Sims' original paper:

| Type | DOF | Description |
|------|-----|-------------|
| Rigid | 0 | Fixed, no movement |
| Revolute | 1 | Rotate around X axis |
| Twist | 1 | Rotate around Z axis (attachment direction) |
| Universal | 2 | Rotate around X and Y |
| BendTwist | 2 | Bend (XY), then twist (Z) |
| TwistBend | 2 | Twist (Z), then bend (XY) |
| Spherical | 3 | Full rotation freedom |

### MorphologyConnection (Attachments)

Each edge describes how a child part attaches to its parent:

```rust
pub struct MorphologyConnection {
    pub position: Vec3,       // [-1,1] normalized position on parent surface
    pub orientation: Quat,    // Child rotation relative to parent
    pub scale: f32,           // Size multiplier for child
    pub reflection: Vec3,     // Axes to mirror (for bilateral symmetry)
    pub terminal_only: bool,  // Only attach at final recursion level
}
```

### Neural Layer

Each body part has its own neural network (`neural.rs`):

```rust
pub struct NeuralGraph {
    pub sensors: Vec<SensorType>,    // Input sensors
    pub neurons: Vec<Neuron>,        // Processing nodes
    pub effectors: Vec<Effector>,    // Motor outputs
}
```

#### Sensor Types

```rust
pub enum SensorType {
    JointAngle { dof: usize },       // Current angle of joint DOF
    Contact { face: Face },          // Touch detection on a face
    PhotoSensor { axis: SensorAxis }, // Light direction sensing
}
```

#### Neuron Functions

17 computation types (simplified from Karl Sims' original 23):

| Category | Functions |
|----------|-----------|
| Arithmetic | Sum, Product |
| Comparison | SumThreshold, GreaterThan, SignOf, Min, Max, Abs, If |
| Interpolation | Interpolate |
| Trigonometric | Sin, Cos |
| Activation | Sigmoid |
| Temporal (stateful) | Integrate, Smooth |
| Oscillators | OscillateWave, OscillateSaw |

#### Neural Input Sources

Neurons can read from 3 different source types:

```rust
pub enum PartRef {
    Local,           // This body part
    Parent,          // Parent body part
    Child(usize),    // Child body part (by connection index)
}

pub enum NeuralInput {
    Constant(f32),                       // Fixed genetic value
    Sensor(usize),                       // Local sensor by index
    Neuron { part: PartRef, index: usize }, // Neuron from any part
}
```

#### Effectors

Motor control for joints:

```rust
pub struct Effector {
    pub dof: usize,              // Which DOF of joint to control
    pub input: WeightedInput,    // Input signal source
    pub max_force: f32,          // Maximum torque
}
```

---

## Genotype to Phenotype Conversion

The conversion from genetic representation to physical simulation happens in `src/phenotype/builder.rs`.

### PhenotypeBuilder

```rust
pub struct PhenotypeBuilder;

impl PhenotypeBuilder {
    pub fn spawn(
        commands: &mut Commands,
        meshes: &mut Assets<Mesh>,
        materials: &mut Assets<StandardMaterial>,
        genotype: &CreatureGenotype,
        position: Vec3,
    ) -> SpawnedCreature;
}
```

### Conversion Algorithm

```
PhenotypeBuilder::spawn(genotype, position):

1. Create creature_id entity (container)

2. Spawn root part:
   - Create Cuboid mesh from node.dimensions
   - Add RigidBody::Dynamic
   - Add Collider (box with half-extents)
   - Add CollisionGroups (GROUP_1 for creatures)
   - Add CreaturePart metadata
   - Mass = volume (dimensions.x * y * z)

3. Recursively spawn children:
   For each connection FROM current node:

   a. Check recursive_limit (prevent infinite loops)
   b. Check terminal_only flag

   c. Compute child transform:
      - clamp_to_surface(position) → find parent's face
      - Apply reflection vector for symmetry
      - Transform to parent's world space
      - Apply connection.orientation
      - Apply connection.scale to dimensions

   d. Spawn child entity:
      - Mesh + Material (color varies by node_id)
      - RigidBody::Dynamic + Collider
      - ImpulseJoint connecting to parent

   e. Create joint based on joint_type:
      - Rigid: all axes locked
      - Revolute: AngX free, others locked
      - Twist: AngZ free, others locked
      - Universal/BendTwist/TwistBend: AngX+AngY free
      - Spherical: all angular axes free
      - Apply joint_limits to each free axis

   f. Recurse for grandchildren (max depth 10)

4. Attach Brain component with genotype reference

5. Return SpawnedCreature { creature_entity, root, parts }
```

### Surface Clamping

The `clamp_to_surface` function maps normalized [-1, 1] positions to the parent's surface:

```rust
fn clamp_to_surface(pos: Vec3, half_extents: Vec3) -> Vec3 {
    let scaled = pos * half_extents;
    let abs_scaled = scaled.abs();

    // Find which face is closest
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
```

### Reflection/Symmetry

Bilateral symmetry is achieved through the `reflection` vector:

```rust
// reflection = Vec3(±1, ±1, ±1)
// Common: Vec3(-1, 1, 1) for left/right symmetry

position_reflected = position * reflection;

// Odd number of reflections flips quaternion handedness
if reflection.x * reflection.y * reflection.z < 0.0 {
    orientation = Quat::from_xyzw(-q.x, q.y, q.z, -q.w);
}
```

### Collision Groups

Creature parts only collide with the ground, not each other:

```rust
// Creature parts
CollisionGroups::new(Group::GROUP_1, Group::GROUP_2)

// Ground
CollisionGroups::new(Group::GROUP_2, Group::GROUP_1)
```

---

## Brain and Neural Evaluation

The brain system (`src/brain/mod.rs`) evaluates neural networks and controls joint motors.

### Brain Component

```rust
pub struct Brain {
    pub genotype: CreatureGenotype,
    pub neuron_outputs: HashMap<(usize, usize), Vec<f32>>,  // (node_id, instance) -> outputs
    pub neuron_state: HashMap<(usize, usize, usize), f32>, // For temporal neurons
    pub time: f32,
    pub sensor_values: HashMap<(usize, usize), Vec<f32>>,
}
```

### Evaluation Pipeline

The `run_brains` system runs every frame:

```
1. Increment brain.time += delta_time

2. Read SENSORS for each part:
   - JointAngle: current motor velocity (proxy for angle)
   - Contact: check if face points down AND part near ground
   - PhotoSensor: how much axis points toward light (up)

3. Evaluate LOCAL NEURONS for each part:
   - Gather weighted inputs from all sources
   - Apply neuron function (with state for temporal neurons)
   - Store outputs

4. Apply EFFECTORS to joints:
   - Get input value (weighted)
   - Scale: motor_velocity = value * 5.0, clamped to [-50, 50]
   - Set joint motor with max_force as damping
```

### Neuron Evaluation

Each neuron function computes its output:

| Function | Computation |
|----------|-------------|
| Sum | ∑inputs |
| Product | ∏inputs |
| SumThreshold | sum > 0 ? 1 : -1 |
| GreaterThan | inputs[0] > inputs[1] ? 1 : -1 |
| SignOf | sign(input) |
| Min/Max | min/max of inputs |
| Abs | abs(input) |
| If | inputs[0] > 0 ? inputs[1] : inputs[2] |
| Interpolate | lerp(inputs[1], inputs[2], inputs[0]) |
| Sin/Cos | Trigonometric |
| Sigmoid | 1 / (1 + e^(-x)) |
| Integrate | state += input * dt |
| Smooth | state * 0.9 + input * 0.1 |
| OscillateWave | sin(time * freq * 2π) |
| OscillateSaw | sawtooth wave |

### Motor Control

Effectors drive joint motors:

```rust
joint.set_motor(
    axis,                    // JointAxis::AngX/Y/Z
    target_pos: 0.0,         // Not used for velocity control
    motor_velocity,          // From neural output
    stiffness: 0.0,          // No spring
    damping: max_force       // Force limit
);
```

---

## Evolution System

The genetic algorithm is implemented in `src/evolution/mod.rs`.

### Configuration

```rust
pub struct EvolutionConfig {
    pub population_size: usize,   // Default: 20
    pub survival_ratio: f32,      // Default: 0.2 (top 20%)
    pub asexual_prob: f32,        // Default: 0.4
    pub crossover_prob: f32,      // Default: 0.3
    pub grafting_prob: f32,       // Default: 0.3
    pub mutation_rate: f32,       // Default: 0.3
    pub test_duration: f32,       // Default: 10.0 seconds
}
```

### Evolution Cycle

```
1. Initialize population with random_genotype()

2. For each individual:
   - Spawn phenotype
   - Simulate for test_duration seconds
   - Calculate fitness (walking speed in +X direction)
   - Despawn

3. After all tested:
   - select_survivors() - keep top 20%
   - Save best to archive (creatures.json)
   - reproduce() to fill population:
     - 40% asexual (copy + mutate)
     - 30% crossover (combine two parents)
     - 30% grafting (splice subtree from donor)
   - Increment generation

4. Repeat from step 2
```

### Fitness Function

```rust
pub fn calculate_fitness(start_pos: Vec3, end_pos: Vec3, duration: f32) -> f32 {
    // Horizontal distance in any direction (ignore Y to avoid rewarding falling)
    let horizontal_dist = Vec2::new(
        end_pos.x - start_pos.x,
        end_pos.z - start_pos.z
    ).length();

    // Normalize by time to get speed
    horizontal_dist / duration.max(1.0)
}
```

**Fitness metric:** Horizontal speed in any direction.

### Mutation

Mutations scale inversely with complexity:

```rust
let complexity = node_count + connection_count;
let adjusted_rate = mutation_rate / complexity.sqrt();
```

Per-node mutations:
- Dimensions: scale each axis by [0.8, 1.25]
- Joint type: 10% chance to change
- Recursive limit: 20% chance to change
- Neuron weights: adjust by [-0.2, 0.2]

Per-connection mutations:
- Position: shift by [-0.2, 0.2] per axis
- Orientation: rotate up to ±0.2 radians
- Scale: multiply by [0.9, 1.1]

Topology mutations:
- 20% chance to add new part (if < 10 nodes)

### Crossover

```rust
pub fn crossover(parent1: &CreatureGenotype, parent2: &CreatureGenotype) -> CreatureGenotype {
    // Copy parent1's morphology
    // Select random non-root node from parent2
    // Attach it to random node in child
}
```

### Grafting

Similar to crossover but explicitly "splices" a subtree from a donor onto a base genotype.

---

## Execution Modes

### Graphics Mode (default)

```bash
cargo run
```

- Full 3D rendering with Bevy
- Camera follows creature
- Real-time simulation

### Headless Mode

```bash
cargo run -- --headless
cargo run -- --headless --speed 10  # 10x faster
```

- No rendering
- Faster evolution
- Minimal output with `--quiet`

### Replay Mode

```bash
cargo run -- --replay creatures.json
```

- Load saved creatures
- Press SPACE to cycle through them
- Shows fitness, generation, part count

---

## Data Flow Summary

```
START
  │
  ├─ Initialize population (random genotypes)
  │
  └─ Main Loop:
     │
     ├─ Spawn current individual's phenotype
     │  └─ PhenotypeBuilder::spawn() → physics entities
     │
     ├─ Simulate for test_duration:
     │  ├─ Physics tick (Rapier)
     │  └─ run_brains() → sensors → neurons → effectors → motors
     │
     ├─ Calculate fitness from displacement
     │
     ├─ Despawn creature
     │
     ├─ Next individual (or evolve if generation complete)
     │  └─ select_survivors() → reproduce() → mutate()
     │
     └─ Repeat
```

---

## Key Files Reference

| File | Key Exports |
|------|-------------|
| `genotype/graph.rs` | `DirectedGraph<N,C>`, `NodeId`, `Connection` |
| `genotype/morphology.rs` | `CreatureGenotype`, `MorphologyNode`, `MorphologyConnection`, `JointType` |
| `genotype/neural.rs` | `NeuronFunc`, `SensorType`, `PartRef`, `NeuralInput`, `Neuron`, `Effector`, `NeuralGraph` |
| `genotype/io.rs` | `SavedCreature`, `CreatureArchive` |
| `phenotype/builder.rs` | `PhenotypeBuilder`, `SpawnedCreature`, `CreaturePart` |
| `brain/mod.rs` | `Brain`, `run_brains` system |
| `evolution/mod.rs` | `EvolutionConfig`, `EvolutionState`, `random_genotype`, `mutate`, `crossover` |
| `main.rs` | App setup, `evolution_system`, CLI parsing |
