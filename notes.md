# Virtual Creatures Project

Extending Karl Sims' 1994 "Evolving Virtual Creatures" work.

## Scope

**Focus:** Land-based creatures (walking, jumping, following)
**Skipping:** Water/swimming behaviors

## Karl Sims' Architecture Summary

### Genotype (Directed Graphs)
- **Nodes:** part dimensions, joint type/limits, recursive limit, local neurons
- **Connections:** position, orientation, scale, reflection, terminal-only flag
- Graphs can be recursive/cyclic for fractal-like structures

### Morphology (Phenotype)
- Hierarchy of 3D rigid rectangular solids
- 7 joint types: rigid, revolute, twist, universal, bend-twist, twist-bend, spherical
- Joint limits with restoring spring forces

### Neural Control System
- Nested inside morphology nodes (duplicated with parts)
- **Sensors:** joint angles, contact sensors, photosensors
- **Neurons (23 types):** sum, product, divide, sum-threshold, greater-than, sign-of, min, max, abs, if, interpolate, sin, cos, atan, log, expt, sigmoid, integrate, differentiate, smooth, memory, oscillate-wave, oscillate-saw
- **Effectors:** joint torques, max strength proportional to cross-sectional area

### Physics Simulation
- Featherstone's O(N) articulated body dynamics
- Runge-Kutta-Fehlberg integration (adaptive step)
- Collision detection with bounding box hierarchies
- Hybrid collision response (impulses + penalty springs)
- Friction model

### Genetic Algorithm
- Population ~300, survival ratio 1/5
- 50-100 generations typical
- **Reproduction ratios:** 40% asexual, 30% crossover, 30% grafting
- Mutation: parameter tweaks, new nodes, connection pointer changes
- Crossover: align nodes, switch copy source at crossover points
- Grafting: connect node from one parent to node in another

### Fitness Functions
- **Walking:** horizontal speed (after settling to prevent falling tricks)
- **Jumping:** max/avg height of lowest part
- **Following:** speed toward light source (multiple trials, averaged)

## Implementation Requirements

### Physics Engine (need to build or use existing)
- Rigid body dynamics with mass, inertia tensors
- Articulated body constraints (joints)
- Collision detection (broad + narrow phase)
- Contact resolution with friction
- Numerical integration

### Rendering (for visualization)
- Could be simple (OpenGL/WebGL)
- Or sophisticated (PBR ray tracing)

## Open Questions
- Language choice?
- Custom physics vs existing engine (MuJoCo, PyBullet, Brax)?
- What existing code/thesis work can be leveraged?

## References
- Sims, K. (1994). "Evolving Virtual Creatures." SIGGRAPH '94
- Featherstone, R. "Robot Dynamics Algorithms" (for articulated body dynamics)
