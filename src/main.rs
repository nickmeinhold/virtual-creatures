use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::env;

mod brain;
mod evolution;
mod genotype;
mod phenotype;

use brain::{Brain, BrainPlugin};
use evolution::*;
use phenotype::*;

/// Command-line options for the simulation
#[derive(Resource, Clone)]
struct SimulationOptions {
    /// Run without graphics (headless mode)
    headless: bool,
    /// Simulation speed multiplier (1.0 = realtime)
    speed: f32,
    /// Verbose output
    verbose: bool,
    /// Replay mode: load and watch saved creatures
    replay: Option<String>,
}

impl Default for SimulationOptions {
    fn default() -> Self {
        Self {
            headless: false,
            speed: 1.0,
            verbose: true,
            replay: None,
        }
    }
}

fn parse_args() -> SimulationOptions {
    let args: Vec<String> = env::args().collect();
    let mut opts = SimulationOptions::default();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--headless" => opts.headless = true,
            "--speed" | "-s" => {
                i += 1;
                if i < args.len() {
                    opts.speed = args[i].parse().unwrap_or(1.0);
                }
            }
            "--quiet" | "-q" => opts.verbose = false,
            "--replay" | "-r" => {
                i += 1;
                if i < args.len() {
                    opts.replay = Some(args[i].clone());
                } else {
                    opts.replay = Some("creatures.json".to_string());
                }
            }
            "--help" | "-h" => {
                println!("Virtual Creatures Evolution Simulator");
                println!();
                println!("Options:");
                println!("  --headless        Run without graphics (faster evolution)");
                println!("  --speed, -s N     Simulation speed multiplier (default: 1.0)");
                println!("  --quiet, -q       Reduce output verbosity");
                println!("  --replay, -r FILE Load and watch saved creatures (default: creatures.json)");
                println!("  --help, -h        Show this help message");
                println!();
                println!("Examples:");
                println!("  cargo run                          # Run with graphics");
                println!("  cargo run -- --headless --speed 10 # Fast evolution");
                println!("  cargo run -- --replay              # Watch saved creatures");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    opts
}

fn main() {
    let opts = parse_args();

    if let Some(ref path) = opts.replay {
        run_replay(opts.clone(), path.clone());
    } else if opts.headless {
        run_headless(opts);
    } else {
        run_with_graphics(opts);
    }
}

fn run_with_graphics(opts: SimulationOptions) {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(BrainPlugin)
        .insert_resource(opts)
        .insert_resource(EvolutionConfig::default())
        .insert_resource(EvolutionState::default())
        .insert_resource(CreatureTracker { center: Vec3::new(0.0, 1.0, 0.0) })
        .add_systems(Startup, setup_with_graphics)
        .add_systems(Update, (evolution_system, camera_follow))
        .run();
}

fn run_headless(opts: SimulationOptions) {
    let speed = opts.speed;

    let mut app = App::new();

    // Minimal plugins for headless - just enough for ECS and time
    app.add_plugins(MinimalPlugins);

    // AssetPlugin and scene resources required by RapierPhysicsPlugin
    app.add_plugins(bevy::asset::AssetPlugin::default());
    app.add_plugins(bevy::scene::ScenePlugin);
    app.init_resource::<Assets<Mesh>>();

    // Add physics without rendering
    app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default());

    app.add_plugins(BrainPlugin);
    app.insert_resource(opts);
    app.insert_resource(EvolutionConfig::default());
    app.insert_resource(EvolutionState::default());
    app.insert_resource(SimulationSpeed(speed));
    app.add_systems(Startup, setup_headless);
    app.add_systems(Update, (advance_simulation_time, evolution_system_headless));

    println!("Running headless at {}x speed...", speed);
    println!("Press Ctrl+C to stop\n");

    app.run();
}

/// Resource to track simulation speed
#[derive(Resource)]
struct SimulationSpeed(f32);

/// State for replay mode
#[derive(Resource)]
struct ReplayState {
    archive: genotype::CreatureArchive,
    current_index: usize,
    creature_spawned: bool,
    display_time: f32,
    /// Frames to wait before spawning first creature (let physics initialize)
    frames_before_spawn: u32,
}

fn run_replay(opts: SimulationOptions, path: String) {
    // Load the archive
    let archive = match genotype::CreatureArchive::load(&path) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error loading creatures from '{}': {}", path, e);
            eprintln!("Run evolution first to generate creatures.");
            std::process::exit(1);
        }
    };

    if archive.creatures.is_empty() {
        eprintln!("No creatures found in '{}'", path);
        std::process::exit(1);
    }

    println!("Loaded {} creatures from '{}'", archive.creatures.len(), path);
    println!("Press SPACE to cycle through creatures\n");

    let replay_state = ReplayState {
        archive,
        current_index: 0,
        creature_spawned: false,
        display_time: 0.0,
        frames_before_spawn: 2,
    };

    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(BrainPlugin)
        .insert_resource(opts)
        .insert_resource(replay_state)
        .insert_resource(CreatureTracker { center: Vec3::new(0.0, 2.0, 0.0) })
        .add_systems(Startup, setup_replay)
        .add_systems(Update, (replay_system, camera_follow))
        .run();
}

fn setup_replay(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(5.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ground plane - belongs to GROUP_2, collides with GROUP_1 (creature parts)
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10000.0, 10000.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Collider::halfspace(Vec3::Y).unwrap(),
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

#[allow(clippy::too_many_arguments)]
fn replay_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<ReplayState>,
    mut tracker: ResMut<CreatureTracker>,
    creatures: Query<Entity, With<TestCreature>>,
    creature_parts: Query<(&CreaturePart, &Transform)>,
) {
    // Wait for physics to initialize
    if state.frames_before_spawn > 0 {
        state.frames_before_spawn -= 1;
        return;
    }

    // Check for space to cycle creatures
    if keyboard.just_pressed(KeyCode::Space) {
        // Despawn current creature
        for entity in creatures.iter() {
            commands.entity(entity).despawn_recursive();
        }
        state.current_index = (state.current_index + 1) % state.archive.creatures.len();
        state.creature_spawned = false;
        state.display_time = 0.0;
    }

    // Spawn creature if needed
    if !state.creature_spawned && state.current_index < state.archive.creatures.len() {
        // Clone data to avoid borrow issues
        let saved = state.archive.creatures[state.current_index].clone();
        let current_index = state.current_index;
        let total_creatures = state.archive.creatures.len();
        let spawn_pos = Vec3::new(0.0, 2.0, 0.0);

        let spawned = PhenotypeBuilder::spawn(
            &mut commands,
            &mut meshes,
            &mut materials,
            &saved.genotype,
            spawn_pos,
        );

        // Mark as test creature
        for entity in &spawned.parts {
            commands.entity(*entity).insert(TestCreature);
        }
        commands.entity(spawned.root).insert(TestCreature);

        // Add brain
        commands.entity(spawned.creature_entity).insert(Brain::new(saved.genotype.clone()));

        tracker.center = spawn_pos;
        state.creature_spawned = true;

        println!(
            "Creature {}/{}: fitness={:.3}, gen={}, parts={}, spawned {} entities at {:?}",
            current_index + 1,
            total_creatures,
            saved.fitness,
            saved.generation,
            saved.part_count,
            spawned.parts.len(),
            spawn_pos,
        );
    }

    // Update display time and tracker
    state.display_time += time.delta_secs();

    let mut total_pos = Vec3::ZERO;
    let mut count = 0;
    for (_, transform) in creature_parts.iter() {
        total_pos += transform.translation;
        count += 1;
    }
    if count > 0 {
        let new_center = total_pos / count as f32;
        if new_center.is_finite() {
            tracker.center = new_center;
        }
    }
}

/// Marker for the current test creature
#[derive(Component)]
struct TestCreature;

/// Resource to track creature's center of mass
#[derive(Resource, Default)]
struct CreatureTracker {
    center: Vec3,
}

/// Simulated elapsed time for headless mode
#[derive(Resource, Default)]
struct SimulatedTime {
    elapsed: f32,
}

fn setup_with_graphics(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    config: Res<EvolutionConfig>,
    mut state: ResMut<EvolutionState>,
) {
    // Camera
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(5.0, 5.0, 10.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Light
    commands.spawn((
        DirectionalLight {
            illuminance: 10000.0,
            shadows_enabled: true,
            ..default()
        },
        Transform::from_xyz(4.0, 8.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));

    // Ground plane with visual mesh - belongs to GROUP_2, collides with GROUP_1 (creature parts)
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10000.0, 10000.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Collider::halfspace(Vec3::Y).unwrap(),
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Initialize population
    state.population = init_population(&config);
    println!("Initialized population with {} individuals", state.population.len());
}

fn setup_headless(
    mut commands: Commands,
    config: Res<EvolutionConfig>,
    mut state: ResMut<EvolutionState>,
) {
    // Ground plane - just collider, no mesh - belongs to GROUP_2, collides with GROUP_1 (creature parts)
    commands.spawn((
        Collider::halfspace(Vec3::Y).unwrap(),
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Initialize population
    state.population = init_population(&config);
    println!("Initialized population with {} individuals", state.population.len());

    // Tracker resource
    commands.insert_resource(CreatureTracker::default());
    commands.insert_resource(SimulatedTime::default());
}

/// Advance simulation time faster in headless mode
fn advance_simulation_time(
    mut sim_time: ResMut<SimulatedTime>,
    time: Res<Time>,
    speed: Res<SimulationSpeed>,
) {
    sim_time.elapsed += time.delta_secs() * speed.0;
}

/// Main evolution system (with graphics)
#[allow(clippy::too_many_arguments)]
fn evolution_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
    config: Res<EvolutionConfig>,
    mut state: ResMut<EvolutionState>,
    mut tracker: ResMut<CreatureTracker>,
    opts: Res<SimulationOptions>,
    creatures: Query<Entity, With<TestCreature>>,
    creature_parts: Query<(&CreaturePart, &Transform)>,
) {
    // Wait for physics to initialize
    if state.frames_before_spawn > 0 {
        state.frames_before_spawn -= 1;
        return;
    }

    let current_time = time.elapsed_secs();

    // Check if we need to spawn a new creature
    let has_creature = !creatures.is_empty();

    if !has_creature {
        // Spawn the current individual
        if state.current_individual < state.population.len() {
            let individual = &state.population[state.current_individual];
            let spawn_pos = Vec3::new(0.0, 2.0, 0.0);

            // Get root node info for logging before we use spawn
            let root = individual.genotype.root_node();
            let root_dims = root.dimensions;

            let spawned = PhenotypeBuilder::spawn(
                &mut commands,
                &mut meshes,
                &mut materials,
                &individual.genotype,
                spawn_pos,
            );

            // Mark all parts as test creature, with special handling for root
            for entity in &spawned.parts {
                commands.entity(*entity).insert(TestCreature);
            }
            // Also mark the root part entity for easier identification
            commands.entity(spawned.root).insert(TestCreature);

            // Add brain to the creature entity (which has CreatureBody)
            // The brain needs the genotype to evaluate neural networks
            commands.entity(spawned.creature_entity).insert(Brain::new(individual.genotype.clone()));

            let num_parts = spawned.parts.len();
            let current = state.current_individual + 1;
            let pop_size = state.population.len();
            let gen = state.generation;

            state.test_start_time = current_time;
            state.test_start_position = spawn_pos;
            tracker.center = spawn_pos;

            if opts.verbose {
                println!(
                    "Testing individual {}/{} (gen {}) - root size: {:.2}x{:.2}x{:.2}, {} parts",
                    current,
                    pop_size,
                    gen,
                    root_dims.x,
                    root_dims.y,
                    root_dims.z,
                    num_parts
                );
            }
        }
    } else {
        // Update center of mass tracking
        let mut total_pos = Vec3::ZERO;
        let mut count = 0;
        for (_, transform) in creature_parts.iter() {
            total_pos += transform.translation;
            count += 1;
        }
        if count > 0 {
            let new_center = total_pos / count as f32;
            if new_center.is_finite() {
                tracker.center = new_center;
            }
        }

        // Check if test duration elapsed
        let elapsed = current_time - state.test_start_time;
        if elapsed >= config.test_duration {
            // Calculate fitness
            let fitness = calculate_fitness(
                state.test_start_position,
                tracker.center,
                config.test_duration,
            );

            let idx = state.current_individual;
            state.population[idx].fitness = fitness;
            if opts.verbose {
                println!(
                    "  Individual {} fitness: {:.3} (moved to x={:.2})",
                    state.current_individual + 1,
                    fitness,
                    tracker.center.x
                );
            }

            // Despawn current creature
            for entity in creatures.iter() {
                commands.entity(entity).despawn_recursive();
            }

            // Move to next individual
            state.current_individual += 1;

            // Check if generation complete
            if state.current_individual >= state.population.len() {
                evolve_generation(&mut state, &config);
            }
        }
    }
}

/// Headless evolution system - uses simulated time and no meshes
#[allow(clippy::too_many_arguments)]
fn evolution_system_headless(
    mut commands: Commands,
    sim_time: Res<SimulatedTime>,
    config: Res<EvolutionConfig>,
    mut state: ResMut<EvolutionState>,
    mut tracker: ResMut<CreatureTracker>,
    opts: Res<SimulationOptions>,
    creatures: Query<Entity, With<TestCreature>>,
    creature_parts: Query<(&CreaturePart, &Transform)>,
) {
    // Wait for physics to initialize
    if state.frames_before_spawn > 0 {
        state.frames_before_spawn -= 1;
        return;
    }

    let current_time = sim_time.elapsed;

    // Check if we need to spawn a new creature
    let has_creature = !creatures.is_empty();

    if !has_creature {
        // Spawn the current individual
        if state.current_individual < state.population.len() {
            let individual = &state.population[state.current_individual];
            let spawn_pos = Vec3::new(0.0, 2.0, 0.0);

            // Get root node info for logging
            let root = individual.genotype.root_node();
            let root_dims = root.dimensions;

            // Spawn without meshes for headless mode
            let spawned = spawn_creature_headless(
                &mut commands,
                &individual.genotype,
                spawn_pos,
            );

            // Mark all parts as test creature
            for entity in &spawned.parts {
                commands.entity(*entity).insert(TestCreature);
            }
            commands.entity(spawned.root).insert(TestCreature);

            // Add brain
            commands.entity(spawned.creature_entity).insert(Brain::new(individual.genotype.clone()));

            let num_parts = spawned.parts.len();
            let current = state.current_individual + 1;
            let pop_size = state.population.len();
            let gen = state.generation;

            state.test_start_time = current_time;
            state.test_start_position = spawn_pos;
            tracker.center = spawn_pos;

            if opts.verbose {
                println!(
                    "Testing individual {}/{} (gen {}) - root: {:.2}x{:.2}x{:.2}, {} parts",
                    current, pop_size, gen,
                    root_dims.x, root_dims.y, root_dims.z, num_parts
                );
            }
        }
    } else {
        // Update center of mass tracking
        let mut total_pos = Vec3::ZERO;
        let mut count = 0;
        for (_, transform) in creature_parts.iter() {
            total_pos += transform.translation;
            count += 1;
        }
        if count > 0 {
            let new_center = total_pos / count as f32;
            if new_center.is_finite() {
                tracker.center = new_center;
            }
        }

        // Check if test duration elapsed
        let elapsed = current_time - state.test_start_time;
        if elapsed >= config.test_duration {
            // Calculate fitness
            let fitness = calculate_fitness(
                state.test_start_position,
                tracker.center,
                config.test_duration,
            );

            let idx = state.current_individual;
            state.population[idx].fitness = fitness;
            if opts.verbose {
                println!(
                    "  Individual {} fitness: {:.3} (x={:.2})",
                    state.current_individual + 1, fitness, tracker.center.x
                );
            }

            // Despawn current creature
            for entity in creatures.iter() {
                commands.entity(entity).despawn_recursive();
            }

            // Move to next individual
            state.current_individual += 1;

            // Check if generation complete
            if state.current_individual >= state.population.len() {
                evolve_generation(&mut state, &config);
            }
        }
    }
}

/// Spawn a creature without meshes (for headless mode)
fn spawn_creature_headless(
    commands: &mut Commands,
    genotype: &genotype::CreatureGenotype,
    position: Vec3,
) -> SpawnedCreature {
    use std::collections::HashMap;

    let mut state = HeadlessSpawnState {
        node_instances: HashMap::new(),
        spawned_parts: Vec::new(),
    };

    let creature_id = commands.spawn_empty().id();
    let root_node = &genotype.morphology[genotype.root];
    let root_transform = Transform::from_translation(position);

    // Spawn root part (no mesh)
    let root_entity = spawn_part_headless(
        commands,
        creature_id,
        genotype.root,
        0,
        root_node,
        root_transform,
        None,
    );
    state.spawned_parts.push(root_entity);
    state.node_instances.insert(genotype.root, 1);

    // Recursively spawn children
    spawn_children_headless(
        commands,
        genotype,
        creature_id,
        genotype.root,
        root_entity,
        root_node,
        root_transform,
        Vec3::ONE,
        &mut state,
        0,
    );

    // Add creature components
    commands.entity(creature_id).insert((
        CreatureRoot,
        CreatureBody {
            parts: state.spawned_parts.clone(),
        },
    ));
    commands.entity(root_entity).insert(CreatureRoot);

    SpawnedCreature {
        creature_entity: creature_id,
        root: root_entity,
        parts: state.spawned_parts,
    }
}

struct HeadlessSpawnState {
    node_instances: std::collections::HashMap<genotype::NodeId, usize>,
    spawned_parts: Vec<Entity>,
}

#[allow(clippy::too_many_arguments)]
fn spawn_children_headless(
    commands: &mut Commands,
    genotype: &genotype::CreatureGenotype,
    creature_id: Entity,
    parent_node_id: genotype::NodeId,
    parent_entity: Entity,
    parent_node: &genotype::MorphologyNode,
    parent_transform: Transform,
    parent_reflection: Vec3,
    state: &mut HeadlessSpawnState,
    depth: usize,
) {
    const MAX_DEPTH: usize = 10;
    if depth >= MAX_DEPTH {
        return;
    }

    for conn in genotype.morphology.connections_from(parent_node_id) {
        let child_node = &genotype.morphology[conn.to];

        let instance_count = state.node_instances.get(&conn.to).copied().unwrap_or(0);
        if instance_count >= child_node.recursive_limit as usize {
            continue;
        }

        let at_terminal = instance_count + 1 >= child_node.recursive_limit as usize;
        if conn.data.terminal_only && !at_terminal {
            continue;
        }

        let combined_reflection = parent_reflection * conn.data.reflection;
        let child_transform = compute_child_transform(
            &parent_transform,
            parent_node,
            &conn.data,
            combined_reflection,
        );

        let child_entity = spawn_part_headless(
            commands,
            creature_id,
            conn.to,
            instance_count,
            child_node,
            child_transform,
            Some((parent_entity, parent_node, &conn.data)),
        );
        state.spawned_parts.push(child_entity);
        *state.node_instances.entry(conn.to).or_insert(0) += 1;

        spawn_children_headless(
            commands,
            genotype,
            creature_id,
            conn.to,
            child_entity,
            child_node,
            child_transform,
            combined_reflection,
            state,
            depth + 1,
        );
    }
}

fn spawn_part_headless(
    commands: &mut Commands,
    creature_id: Entity,
    node_id: genotype::NodeId,
    instance: usize,
    node: &genotype::MorphologyNode,
    transform: Transform,
    parent_info: Option<(Entity, &genotype::MorphologyNode, &genotype::MorphologyConnection)>,
) -> Entity {
    use bevy_rapier3d::dynamics::TypedJoint;

    let dims = node.dimensions;

    // Collision groups: creature parts only collide with ground, not each other
    let creature_group = Group::GROUP_1;
    let ground_group = Group::GROUP_2;

    // Spawn without mesh - just physics
    let mut entity_commands = commands.spawn((
        transform,
        RigidBody::Dynamic,
        Collider::cuboid(dims.x / 2.0, dims.y / 2.0, dims.z / 2.0),
        ColliderMassProperties::Mass(node.volume()),
        CollisionGroups::new(creature_group, ground_group),
        CreaturePart {
            creature_id,
            node_id,
            instance,
        },
    ));

    let entity = entity_commands.id();

    if let Some((parent_entity, parent_node, connection)) = parent_info {
        let joint = create_joint_headless(node, parent_node, connection);
        entity_commands.insert(ImpulseJoint {
            parent: parent_entity,
            data: TypedJoint::GenericJoint(joint),
        });
    }

    entity
}

fn compute_child_transform(
    parent_transform: &Transform,
    parent_node: &genotype::MorphologyNode,
    connection: &genotype::MorphologyConnection,
    reflection: Vec3,
) -> Transform {
    let parent_half = parent_node.dimensions / 2.0;
    let conn_pos = connection.position * reflection;

    // Clamp to surface
    let scaled = conn_pos * parent_half;
    let abs_scaled = scaled.abs();
    let attach_local = if abs_scaled.x >= abs_scaled.y && abs_scaled.x >= abs_scaled.z {
        Vec3::new(parent_half.x * conn_pos.x.signum(), scaled.y, scaled.z)
    } else if abs_scaled.y >= abs_scaled.z {
        Vec3::new(scaled.x, parent_half.y * conn_pos.y.signum(), scaled.z)
    } else {
        Vec3::new(scaled.x, scaled.y, parent_half.z * conn_pos.z.signum())
    };

    let attach_world = parent_transform.transform_point(attach_local);

    let orientation = if reflection.x * reflection.y * reflection.z < 0.0 {
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

fn create_joint_headless(
    child_node: &genotype::MorphologyNode,
    parent_node: &genotype::MorphologyNode,
    connection: &genotype::MorphologyConnection,
) -> GenericJoint {
    use genotype::JointType;

    let parent_half = parent_node.dimensions / 2.0;
    let child_half = child_node.dimensions / 2.0;

    // Clamp to surface for anchor
    let conn_pos = connection.position;
    let scaled = conn_pos * parent_half;
    let abs_scaled = scaled.abs();
    let parent_anchor = if abs_scaled.x >= abs_scaled.y && abs_scaled.x >= abs_scaled.z {
        Vec3::new(parent_half.x * conn_pos.x.signum(), scaled.y, scaled.z)
    } else if abs_scaled.y >= abs_scaled.z {
        Vec3::new(scaled.x, parent_half.y * conn_pos.y.signum(), scaled.z)
    } else {
        Vec3::new(scaled.x, scaled.y, parent_half.z * conn_pos.z.signum())
    };

    let conn_dir = connection.position.normalize_or_zero();
    let child_anchor = -conn_dir * child_half;

    let mut joint = match child_node.joint_type {
        JointType::Rigid => {
            GenericJointBuilder::new(JointAxesMask::LOCKED_FIXED_AXES)
                .local_anchor1(parent_anchor)
                .local_anchor2(child_anchor)
                .build()
        }
        JointType::Revolute => {
            GenericJointBuilder::new(JointAxesMask::LOCKED_REVOLUTE_AXES)
                .local_anchor1(parent_anchor)
                .local_anchor2(child_anchor)
                .build()
        }
        JointType::Twist => {
            let mut axes = JointAxesMask::LOCKED_FIXED_AXES;
            axes.set(JointAxesMask::ANG_Z, false);
            GenericJointBuilder::new(axes)
                .local_anchor1(parent_anchor)
                .local_anchor2(child_anchor)
                .build()
        }
        JointType::Universal | JointType::BendTwist | JointType::TwistBend => {
            let mut axes = JointAxesMask::LOCKED_FIXED_AXES;
            axes.set(JointAxesMask::ANG_X, false);
            axes.set(JointAxesMask::ANG_Y, false);
            GenericJointBuilder::new(axes)
                .local_anchor1(parent_anchor)
                .local_anchor2(child_anchor)
                .build()
        }
        JointType::Spherical => {
            GenericJointBuilder::new(JointAxesMask::LOCKED_SPHERICAL_AXES)
                .local_anchor1(parent_anchor)
                .local_anchor2(child_anchor)
                .build()
        }
    };

    // Apply joint limits
    for (dof_idx, &(min, max)) in child_node.joint_limits.limits.iter().enumerate() {
        let axis = match dof_idx {
            0 => JointAxis::AngX,
            1 => JointAxis::AngY,
            _ => JointAxis::AngZ,
        };
        joint.set_limits(axis, [min, max]);
        let stiffness = child_node.joint_limits.stiffness;
        joint.set_motor(axis, 0.0, 0.0, 0.0, stiffness);
    }

    joint
}

/// Camera follows the creature (only used in graphics mode)
fn camera_follow(
    tracker: Res<CreatureTracker>,
    mut camera: Query<&mut Transform, With<Camera3d>>,
) {
    for mut transform in camera.iter_mut() {
        let target = tracker.center + Vec3::new(-5.0, 3.0, 8.0);
        transform.translation = transform.translation.lerp(target, 0.02);
        transform.look_at(tracker.center, Vec3::Y);
    }
}
