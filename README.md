# Virtual Creatures

Evolving virtual creatures in a physically simulated 3D environment, extending Karl Sims' seminal 1994 SIGGRAPH paper ["Evolving Virtual Creatures"](https://www.karlsims.com/papers/siggraph94.pdf).

Creatures' bodies (morphology) and brains (neural control systems) are both encoded as directed graphs and evolved together using genetic algorithms. No hand-design required - interesting locomotion strategies emerge from random mutations and selection pressure.

## Demo

```bash
cargo run
```

Watch random creatures spawn, wiggle for 10 seconds each, get scored on forward movement, then evolve over generations.

## Architecture

### Genotype (DNA)

Creatures are encoded as **directed graphs** that can be recursive/cyclic:

- **Nodes:** body part dimensions, joint type, recursive limit, local neural circuit
- **Connections:** attachment position, orientation, scale, reflection (for symmetry)

### Phenotype (Body)

Genotypes are instantiated into physics entities:

- Hierarchy of 3D rigid boxes (Rapier physics)
- 7 joint types: rigid, revolute, twist, universal, bend-twist, twist-bend, spherical
- Recursive structures (e.g., multi-segment limbs) via graph cycles

### Neural Control

Each body part has a local neural graph:

- **Sensors:** joint angles, contact, photosensors
- **Neurons (23 types):** arithmetic, comparison, trigonometric, temporal (oscillators, memory, integrators)
- **Effectors:** joint motor torques

Neural circuits are duplicated with body parts, enabling distributed control.

### Genetic Algorithm

- Population of 20 (configurable)
- Top 20% survive each generation
- Reproduction: 40% asexual mutation, 30% crossover, 30% grafting
- Mutations: dimensions, joint types, neural weights, topology

### Fitness

Currently: **forward distance / time** (walking speed in +X direction)

## Project Structure

```sh
src/
├── main.rs              # Evolution loop, Bevy app
├── genotype/
│   ├── graph.rs         # Directed graph with arena indexing
│   ├── morphology.rs    # Body parts, joints, connections
│   └── neural.rs        # Sensors, neurons, effectors
├── phenotype/
│   └── builder.rs       # Genotype → Bevy/Rapier entities
├── brain/
│   └── mod.rs           # Neural simulation, motor control
└── evolution/
    └── mod.rs           # Random generation, mutation, crossover, selection
```

## Tech Stack

- **Rust** - systems language
- **Bevy** - ECS game engine for rendering
- **Rapier** - physics simulation (via bevy_rapier3d)
- **nalgebra** - linear algebra

## Configuration

Edit `EvolutionConfig` in `src/evolution/mod.rs`:

```rust
EvolutionConfig {
    population_size: 20,
    survival_ratio: 0.2,      // top 20% survive
    asexual_prob: 0.4,
    crossover_prob: 0.3,
    grafting_prob: 0.3,
    mutation_rate: 0.3,
    test_duration: 10.0,      // seconds per creature
}
```

## Karl Sims' Original Architecture

### Morphology (Original)

- Hierarchy of 3D rigid rectangular solids
- 7 joint types with limits and restoring springs
- Recursive/fractal structures via graph cycles

### Neural Control (Original)

- Nested inside morphology nodes (duplicated with parts)
- 23 neuron function types
- Effector strength proportional to cross-sectional area

### Physics (Original)

- Featherstone's O(N) articulated body dynamics
- Runge-Kutta-Fehlberg adaptive integration
- Collision detection with bounding boxes
- Hybrid impulse/spring collision response

### Fitness Functions (Original)

- **Walking:** horizontal speed after settling
- **Jumping:** max height of lowest part
- **Following:** speed toward light source

## Future Work

### Near-term

- [ ] Wire up full neural network evaluation (currently using simple oscillators)
- [ ] Implement proper sensor feedback (joint angles, contact)
- [ ] Add jumping fitness function
- [ ] Add light-following behavior with photosensors
- [ ] Parallel fitness evaluation (test multiple creatures simultaneously)
- [ ] Save/load best genotypes to disk

### Medium-term

- [ ] More morphology shapes (cylinders, capsules)
- [ ] Muscle-like effectors (opposing pairs)
- [ ] Energy efficiency in fitness (distance / energy consumed)
- [ ] Terrain variations (slopes, stairs, obstacles)
- [ ] Interactive evolution mode (user picks favorites)

### Long-term

- [ ] Coevolution (predator/prey competition)
- [ ] Soft body parts / deformable creatures
- [ ] GPU-accelerated parallel physics (Brax/JAX or CUDA)
- [ ] Transfer to real robots
- [ ] More complex behaviors (object manipulation, navigation)

### Research Extensions

- Modern neural architectures (transformers, attention)
- Lifetime learning (evolution + learning during evaluation)
- Open-ended evolution (novelty search, quality-diversity)
- Multi-objective optimization

## References

- Sims, K. (1994). ["Evolving Virtual Creatures."](https://www.karlsims.com/papers/siggraph94.pdf) SIGGRAPH '94
- Featherstone, R. "Robot Dynamics Algorithms" - articulated body dynamics
- [Rapier Physics](https://rapier.rs/) - Rust physics engine
- [Bevy Engine](https://bevyengine.org/) - Rust game engine

## License

MIT
