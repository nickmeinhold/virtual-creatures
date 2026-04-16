# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Virtual Creatures is a Rust implementation of Karl Sims' 1994 SIGGRAPH paper "Evolving Virtual Creatures". It evolves creature morphologies and neural control systems in a 3D physics environment using genetic algorithms.

**Tech Stack:** Rust, Bevy 0.15 (ECS game engine), Rapier 0.28 (physics)

## Build & Run Commands

```bash
# Run with 3D visualization
cargo run

# Headless mode (no rendering, faster evolution)
cargo run -- --headless

# Headless with speed multiplier
cargo run -- --headless --speed 10

# Replay saved creatures
cargo run -- --replay
cargo run -- --replay creatures.json
```

Dev builds have `opt-level=3` for dependencies to speed up physics. Use `cargo build --release` for extended evolutionary runs.

## Architecture

The system follows a Genotype → Phenotype → Brain → Fitness → Evolution pipeline:

```
Genotype (directed graph)     →  Phenotype (Bevy/Rapier entities)
     ↓                                    ↓
Neural circuits per body part  →  Brain evaluation each frame
     ↓                                    ↓
Effectors → motor torque       →  Fitness = horizontal displacement
     ↓
Evolution (select top 20%, mutate, crossover, graft)
```

### Core Modules

| Module | Purpose |
|--------|---------|
| `main.rs` | CLI, Bevy app setup, evolution loop, creature spawning/despawning |
| `genotype/` | Genetic representation: morphology graph, neural circuits, save/load |
| `phenotype/` | Converts genotype to physics entities (rigid bodies, joints, motors) |
| `brain/` | Neural evaluation system: sensors → neurons → effectors → motors |
| `evolution/` | GA operators: selection, mutation, crossover, grafting, population management |

### Key Data Structures

- **DirectedGraph<N, C>**: Arena-based graph (avoids Rc<RefCell>)
- **MorphologyNode**: Dimensions, JointType (7 variants), JointLimits, recursive_limit, NeuralGraph
- **NeuralGraph**: Sensors, Neurons (17 types), Effectors per body part
- **Brain**: Runtime neural state, manages temporal neurons and sensor values

### Physics Integration

- 7 joint types: Rigid, Revolute, Twist, Universal, BendTwist, TwistBend, Spherical
- Collision groups: Creatures (GROUP_1) only collide with ground (GROUP_2), not each other
- Motors use velocity targets, not position targets

## Common Modification Points

- **Fitness function**: `src/evolution/mod.rs` around line 459
- **Neuron types**: `src/genotype/neural.rs` enum `NeuronFunc`
- **Mutation rates**: `src/evolution/mod.rs` `EvolutionConfig`
- **Joint types**: `src/genotype/morphology.rs` enum `JointType`
- **Motor/effector scaling**: `src/brain/mod.rs` in `apply_effectors()`

## Configuration

Edit `EvolutionConfig` in `src/evolution/mod.rs`:
```rust
EvolutionConfig {
    population_size: 20,
    asexual_prob: 0.4,
    crossover_prob: 0.3,
    mutation_rate: 0.3,
    test_duration: 10.0,      // Seconds per creature
}
```

Speciation is configured separately via `SpeciationConfig` (compatibility threshold, stagnation limit, etc.).

## Common Issues

- **NaN/Inf in physics**: Check `SafeTransform` in `brain/mod.rs`
- **Creatures exploding**: Collision groups in `phenotype/builder.rs`
- **Motors not working**: Check motor velocity scaling in `brain/mod.rs`

## Save Files

`creatures.json` is written after each generation (graphics mode) and can be replayed with `--replay`.
