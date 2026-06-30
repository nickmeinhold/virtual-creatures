use bevy::prelude::*;
use bevy_rapier3d::prelude::*;
use std::env;

mod brain;
mod evolution;
mod export;
mod genotype;
mod phenotype;

use brain::{Brain, BrainPlugin};
use evolution::*;
use phenotype::*;

/// Fixed simulation timestep (seconds) shared by physics, the brain, and trial
/// duration accounting. Decoupling the sim from wall-clock time is what lets
/// headless evolution run as fast as the CPU allows *without* distorting
/// evaluation: every tick advances physics, brain, and the duration clock by
/// exactly this much, so a 10 "second" trial is always 600 faithful steps
/// regardless of how fast the loop spins.
pub const SIM_DT: f32 = 1.0 / 60.0;

/// Rapier's `TimestepMode` pinned to a fixed per-tick step. Without this the
/// plugin defaults to `Variable` (steps by real `Time::delta`), which made the
/// `--speed` multiplier silently judge creatures on a fraction of their
/// intended physics. Inserted as a resource before the app runs.
const FIXED_TIMESTEP: TimestepMode = TimestepMode::Fixed { dt: SIM_DT, substeps: 1 };

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
    /// Export mode: bake saved creatures to a web-playable JSON at this path
    export: Option<String>,
}

impl Default for SimulationOptions {
    fn default() -> Self {
        Self {
            headless: false,
            speed: 1.0,
            verbose: true,
            replay: None,
            export: None,
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
            "--export" | "-e" => {
                // Optional output path; default creatures-web.json.
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    opts.export = Some(args[i].clone());
                } else {
                    opts.export = Some("creatures-web.json".to_string());
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
                println!("  --export, -e FILE Bake creatures.json to web-playable JSON (default: creatures-web.json)");
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

    if let Some(ref out) = opts.export {
        run_export(opts.clone(), out.clone());
    } else if let Some(ref path) = opts.replay {
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
        // Unified fixed timestep across ALL modes: the brain steps SIM_DT per
        // tick, so physics must too, or the two desync at non-60Hz framerates.
        .insert_resource(FIXED_TIMESTEP)
        .insert_resource(EvolutionConfig::default())
        .insert_resource(EvolutionState::default())
        .insert_resource(CreatureTracker { center: Vec3::new(0.0, 1.0, 0.0), ..default() })
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
    app.insert_resource(FIXED_TIMESTEP);
    app.add_systems(Startup, setup_headless);
    app.add_systems(Update, (advance_simulation_time, evolution_system_headless));

    println!("Running headless at {}x speed...", speed);
    println!("Press Ctrl+C to stop\n");

    app.run();
}

/// State for export mode: walk the archive, simulate each creature, and bake
/// its per-frame poses into a web-playable gallery.
#[derive(Resource)]
struct ExportState {
    archive: genotype::CreatureArchive,
    out_path: String,
    fps: u32,
    /// Seconds of motion to record per creature.
    duration: f32,
    current_index: usize,
    gallery: export::WebGallery,
    /// Part entities of the creature being recorded, in stable spawn order.
    part_entities: Vec<Entity>,
    /// Box dimensions per part (same order as `part_entities`).
    part_dims: Vec<[f32; 3]>,
    /// Accumulated poses: frames[f][p].
    frames: Vec<Vec<[f32; 7]>>,
    /// Sim time at which recording for the current creature began.
    start_time: Option<f32>,
}

fn run_export(opts: SimulationOptions, out_path: String) {
    let mut archive = match genotype::CreatureArchive::load("creatures.json") {
        Ok(a) => a,
        Err(e) => {
            eprintln!("Error loading creatures.json: {}", e);
            std::process::exit(1);
        }
    };
    if archive.creatures.is_empty() {
        eprintln!("No creatures in creatures.json to export.");
        std::process::exit(1);
    }

    // Export only the strongest distinct lineages — the gallery showcases the
    // top handful, and this keeps the web payload light.
    archive.creatures.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
    archive.creatures.truncate(16);

    let fps = 24;
    let n = archive.creatures.len();
    println!("Exporting {} creatures to '{}' ({} fps)...", n, out_path, fps);

    let mut app = App::new();
    app.add_plugins(MinimalPlugins);
    app.add_plugins(bevy::asset::AssetPlugin::default());
    app.add_plugins(bevy::scene::ScenePlugin);
    app.init_resource::<Assets<Mesh>>();
    app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default());
    app.add_plugins(BrainPlugin);
    // Record at realtime so physics integrates with full fidelity — a high
    // speed multiplier would take fewer physics steps per recorded second and
    // make the baked motion choppy.
    app.insert_resource(SimulationSpeed(1.0));
    app.insert_resource(opts);
    app.insert_resource(SimulatedTime::default());
    app.insert_resource(ExportState {
        archive,
        out_path,
        fps,
        duration: 8.0,
        current_index: 0,
        gallery: export::WebGallery::new(fps),
        part_entities: Vec::new(),
        part_dims: Vec::new(),
        frames: Vec::new(),
        start_time: None,
    });
    app.insert_resource(FIXED_TIMESTEP);
    app.add_systems(Startup, setup_export);
    app.add_systems(Update, (advance_simulation_time, export_system));
    app.run();
}

fn setup_export(mut commands: Commands) {
    // Ground plane (collider only; no mesh needed headless).
    commands.spawn((
        Collider::halfspace(Vec3::Y).unwrap(),
        CollisionGroups::new(Group::GROUP_2, Group::GROUP_1),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));
}

#[allow(clippy::too_many_arguments)]
fn export_system(
    mut commands: Commands,
    sim_time: Res<SimulatedTime>,
    mut state: ResMut<ExportState>,
    creatures: Query<Entity, With<TestCreature>>,
    parts_q: Query<(&CreaturePart, &Transform)>,
) {
    let t = sim_time.elapsed;
    let total_frames = (state.duration * state.fps as f32).round() as usize;

    // No creature in the world: either finish, or spawn the next one.
    if creatures.is_empty() {
        if state.current_index >= state.archive.creatures.len() {
            // Done — write the gallery and exit.
            if let Err(e) = state.gallery.save(&state.out_path) {
                eprintln!("Failed to write {}: {}", state.out_path, e);
                std::process::exit(1);
            }
            println!(
                "Wrote {} creatures to '{}'.",
                state.gallery.creatures.len(),
                state.out_path
            );
            std::process::exit(0);
        }

        let genotype = state.archive.creatures[state.current_index].genotype.clone();
        let spawned = spawn_creature_headless(&mut commands, &genotype, Vec3::new(0.0, 2.0, 0.0));
        for entity in &spawned.parts {
            commands.entity(*entity).insert(TestCreature);
        }
        commands.entity(spawned.root).insert(TestCreature);
        commands.entity(spawned.creature_entity).insert(Brain::new(genotype.clone()));

        // Record part dimensions in spawn order so they line up with poses.
        state.part_dims = spawned.parts.iter().map(|_| [0.0; 3]).collect();
        state.part_entities = spawned.parts.clone();
        state.frames.clear();
        state.start_time = None;
        return;
    }

    // Lazily fill dims the first tick the parts are queryable, and anchor the
    // recording clock to that moment.
    if state.start_time.is_none() {
        let genotype = state.archive.creatures[state.current_index].genotype.clone();
        let entities = state.part_entities.clone();
        let mut dims = vec![[0.0f32; 3]; entities.len()];
        for (i, entity) in entities.iter().enumerate() {
            if let Ok((cp, _)) = parts_q.get(*entity) {
                if let Some(node) = genotype.morphology.get_node(cp.node_id) {
                    let d = node.dimensions;
                    dims[i] = [d.x, d.y, d.z];
                }
            }
        }
        state.part_dims = dims;
        state.start_time = Some(t);
    }

    let local = t - state.start_time.unwrap();

    // Sample one pose whenever the playback grid advances past a new frame.
    if local >= state.frames.len() as f32 / state.fps as f32 && state.frames.len() < total_frames {
        let entities = state.part_entities.clone();
        let mut pose = Vec::with_capacity(entities.len());
        // Quantize to keep the web JSON small: positions to mm, quaternions to
        // 1e-4. Visually lossless for a card-sized canvas, ~2x smaller on disk.
        let qpos = |v: f32| (v * 1000.0).round() / 1000.0;
        let qrot = |v: f32| (v * 10000.0).round() / 10000.0;
        for entity in &entities {
            if let Ok((_, tf)) = parts_q.get(*entity) {
                let p = tf.translation;
                let q = tf.rotation;
                pose.push([qpos(p.x), qpos(p.y), qpos(p.z), qrot(q.x), qrot(q.y), qrot(q.z), qrot(q.w)]);
            } else {
                pose.push([0.0; 7]);
            }
        }
        state.frames.push(pose);
    }

    // Finished recording this creature: stash it and tear down for the next.
    if state.frames.len() >= total_frames {
        let id = state.current_index;
        let (fitness, generation, species_id) = {
            let saved = &state.archive.creatures[id];
            (saved.fitness, saved.generation, saved.species_id)
        };
        let frames = std::mem::take(&mut state.frames);
        let parts = state.part_dims.iter().map(|d| export::WebPart { dims: *d }).collect();
        let web = export::WebCreature { id, fitness, generation, species_id, parts, frames };
        println!("  baked creature {} (fitness {:.3})", id, fitness);
        state.gallery.creatures.push(web);
        state.current_index += 1;

        for entity in creatures.iter() {
            commands.entity(entity).despawn_recursive();
        }
    }
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
        // Same unified fixed timestep as every other mode (see run_with_graphics).
        .insert_resource(FIXED_TIMESTEP)
        .insert_resource(replay_state)
        .insert_resource(CreatureTracker { center: Vec3::new(0.0, 2.0, 0.0), ..default() })
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

/// Resource to track creature's center of mass and per-objective telemetry.
/// Height/spin fields are reset once at `settled` so they measure behaviour,
/// not the initial spawn drop.
#[derive(Resource, Default)]
struct CreatureTracker {
    center: Vec3,
    /// Highest center-of-mass Y reached (post-settle) — distance launch guard,
    /// jump reward.
    peak_height: f32,
    /// Highest Y any single part reaches (post-settle) — reach reward.
    peak_part_height: f32,
    /// Highest the LOWEST part ever rose (post-settle) — grounded check for reach.
    min_part_floor: f32,
    /// Centre-of-mass height captured at the settle point — jump's resting baseline.
    settle_com: f32,
    /// Fastest single part speed (linear, m/s) over the run — universal cheat guard.
    max_part_speed: f32,
    /// Fastest single part angular speed (rad/s) over the run — spin cheat guard.
    max_angspeed: f32,
    /// Accumulated mean angular motion (radians, post-settle) — spin reward.
    total_spin: f32,
    /// Whether the post-spawn settle reset has happened yet this run.
    settled: bool,
}

impl CreatureTracker {
    /// Clear all telemetry for a fresh creature spawning at `spawn_pos`.
    fn reset(&mut self, spawn_pos: Vec3) {
        self.center = spawn_pos;
        self.peak_height = spawn_pos.y;
        self.peak_part_height = spawn_pos.y;
        self.min_part_floor = spawn_pos.y;
        self.settle_com = spawn_pos.y;
        self.max_part_speed = 0.0;
        self.max_angspeed = 0.0;
        self.total_spin = 0.0;
        self.settled = false;
    }

    /// Snapshot the telemetry into the objective-agnostic fitness inputs.
    fn fitness_inputs(&self, start_pos: Vec3, duration: f32) -> FitnessInputs {
        FitnessInputs {
            start_pos,
            end_pos: self.center,
            duration,
            peak_com_height: self.peak_height,
            settle_com_height: self.settle_com,
            peak_part_height: self.peak_part_height,
            min_part_floor: self.min_part_floor,
            max_part_speed: self.max_part_speed,
            max_angspeed: self.max_angspeed,
            total_spin: self.total_spin,
        }
    }
}

/// Seconds to let a creature drop and settle before height/spin telemetry
/// starts — otherwise the spawn fall pollutes every height-based objective.
const SETTLE_SECS: f32 = 1.0;

/// Fold one frame of part state into the tracker: centre of mass, the linear
/// and angular cheat-guard peaks (tracked from spawn), and — once settled — the
/// per-objective behaviour peaks. `elapsed` is seconds since this run's start.
fn accumulate_telemetry(
    tracker: &mut CreatureTracker,
    parts: &Query<(&CreaturePart, &Transform, &Velocity)>,
    elapsed: f32,
) {
    let mut total_pos = Vec3::ZERO;
    let mut count = 0u32;
    let mut frame_min_y = f32::INFINITY;
    let mut frame_max_y = f32::NEG_INFINITY;
    let mut sum_angspeed = 0.0;
    for (_, tf, vel) in parts.iter() {
        total_pos += tf.translation;
        count += 1;
        frame_min_y = frame_min_y.min(tf.translation.y);
        frame_max_y = frame_max_y.max(tf.translation.y);
        let lin = vel.linvel.length();
        if lin.is_finite() && lin > tracker.max_part_speed {
            tracker.max_part_speed = lin;
        }
        let ang = vel.angvel.length();
        if ang.is_finite() && ang > tracker.max_angspeed {
            tracker.max_angspeed = ang;
        }
        sum_angspeed += ang;
    }
    if count == 0 {
        return;
    }
    let com = total_pos / count as f32;
    if com.is_finite() {
        tracker.center = com;
    }
    let mean_angspeed = sum_angspeed / count as f32;

    if !tracker.settled {
        // Re-baseline the behaviour peaks the moment the creature has settled,
        // then start accumulating against that baseline.
        if elapsed >= SETTLE_SECS {
            tracker.peak_height = com.y;
            tracker.settle_com = com.y;
            tracker.peak_part_height = frame_max_y;
            tracker.min_part_floor = frame_min_y;
            tracker.total_spin = 0.0;
            tracker.settled = true;
        }
        return;
    }
    tracker.peak_height = tracker.peak_height.max(com.y);
    tracker.peak_part_height = tracker.peak_part_height.max(frame_max_y);
    tracker.min_part_floor = tracker.min_part_floor.max(frame_min_y);
    if mean_angspeed.is_finite() {
        tracker.total_spin += mean_angspeed * SIM_DT;
    }
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
    state.population = init_population(&config, &mut state.innovation_counter);
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
    state.population = init_population(&config, &mut state.innovation_counter);
    println!("Initialized population with {} individuals", state.population.len());

    // Tracker resource
    commands.insert_resource(CreatureTracker::default());
    commands.insert_resource(SimulatedTime::default());
}

/// Advance the simulation clock by one fixed step per tick, matching the fixed
/// physics timestep. The loop runs uncapped headless, so wall-clock speed comes
/// from how fast ticks fire — not from scaling time, which would desync the
/// duration clock from the physics the creature actually experiences.
fn advance_simulation_time(mut sim_time: ResMut<SimulatedTime>) {
    sim_time.elapsed += SIM_DT;
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
    keyboard: Res<ButtonInput<KeyCode>>,
    creatures: Query<Entity, With<TestCreature>>,
    creature_parts: Query<(&CreaturePart, &Transform, &Velocity)>,
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
            tracker.reset(spawn_pos);

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
        let elapsed = current_time - state.test_start_time;
        accumulate_telemetry(&mut tracker, &creature_parts, elapsed);

        // Check if test duration elapsed or space pressed to skip
        let skip_pressed = keyboard.just_pressed(KeyCode::Space);
        if elapsed >= config.test_duration || skip_pressed {
            let fitness = calculate_fitness(
                &tracker.fitness_inputs(state.test_start_position, config.test_duration),
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
    creature_parts: Query<(&CreaturePart, &Transform, &Velocity)>,
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
            tracker.reset(spawn_pos);

            if opts.verbose {
                println!(
                    "Testing individual {}/{} (gen {}) - root: {:.2}x{:.2}x{:.2}, {} parts",
                    current, pop_size, gen,
                    root_dims.x, root_dims.y, root_dims.z, num_parts
                );
            }
        }
    } else {
        let elapsed = current_time - state.test_start_time;
        accumulate_telemetry(&mut tracker, &creature_parts, elapsed);

        // Check if test duration elapsed
        if elapsed >= config.test_duration {
            let fitness = calculate_fitness(
                &tracker.fitness_inputs(state.test_start_position, config.test_duration),
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
    const MAX_PARTS: usize = 8;
    if depth >= MAX_DEPTH || state.spawned_parts.len() >= MAX_PARTS {
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
        // Velocity is written back by Rapier each step; we read it to detect
        // physics-cheat creatures (solver-energy-leak vibration, launches).
        Velocity::default(),
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

    // Apply joint limits - let the brain control motors
    for (dof_idx, &(min, max)) in child_node.joint_limits.limits.iter().enumerate() {
        let axis = match dof_idx {
            0 => JointAxis::AngX,
            1 => JointAxis::AngY,
            _ => JointAxis::AngZ,
        };
        joint.set_limits(axis, [min, max]);
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
