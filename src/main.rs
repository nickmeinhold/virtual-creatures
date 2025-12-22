use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

mod brain;
mod evolution;
mod genotype;
mod phenotype;

use brain::*;
use evolution::*;
use genotype::*;
use phenotype::*;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(RapierPhysicsPlugin::<NoUserData>::default())
        .add_plugins(RapierDebugRenderPlugin::default())
        .add_plugins(BrainPlugin)
        .insert_resource(EvolutionConfig::default())
        .insert_resource(EvolutionState::default())
        .add_systems(Startup, setup)
        .add_systems(Update, (evolution_system, camera_follow))
        .run();
}

/// Marker for the current test creature
#[derive(Component)]
struct TestCreature;

/// Resource to track creature's center of mass
#[derive(Resource, Default)]
struct CreatureTracker {
    center: Vec3,
}

fn setup(
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

    // Ground plane - infinite for all practical purposes
    commands.spawn((
        Mesh3d(meshes.add(Plane3d::default().mesh().size(10000.0, 10000.0))),
        MeshMaterial3d(materials.add(Color::srgb(0.3, 0.5, 0.3))),
        Collider::halfspace(Vec3::Y).unwrap(),
        Transform::from_xyz(0.0, 0.0, 0.0),
    ));

    // Initialize population
    state.population = init_population(&config);
    println!("Initialized population with {} individuals", state.population.len());

    // Tracker resource
    commands.insert_resource(CreatureTracker::default());
}

/// Main evolution system
fn evolution_system(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    time: Res<Time>,
    config: Res<EvolutionConfig>,
    mut state: ResMut<EvolutionState>,
    mut tracker: ResMut<CreatureTracker>,
    creatures: Query<Entity, With<TestCreature>>,
    creature_parts: Query<(&CreaturePart, &Transform)>,
) {
    let current_time = time.elapsed_secs();

    // Check if we need to spawn a new creature
    let has_creature = !creatures.is_empty();

    if !has_creature {
        // Spawn the current individual
        if state.current_individual < state.population.len() {
            let individual = &state.population[state.current_individual];
            let spawn_pos = Vec3::new(0.0, 2.0, 0.0);

            let spawned = PhenotypeBuilder::spawn(
                &mut commands,
                &mut meshes,
                &mut materials,
                &individual.genotype,
                spawn_pos,
            );

            // Mark all parts as test creature and add brain
            for entity in &spawned.parts {
                commands.entity(*entity).insert(TestCreature);
            }

            // Add brain to root
            commands.entity(spawned.root).insert(Brain::new());

            state.test_start_time = current_time;
            state.test_start_position = spawn_pos;
            tracker.center = spawn_pos;

            println!(
                "Testing individual {}/{} (gen {})",
                state.current_individual + 1,
                state.population.len(),
                state.generation
            );
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
            tracker.center = total_pos / count as f32;
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
            println!(
                "  Individual {} fitness: {:.3} (moved to x={:.2})",
                state.current_individual + 1,
                fitness,
                tracker.center.x
            );

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

/// Camera follows the creature
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
