use std::ops::{Deref, DerefMut};

use bevy::{
    app::{App, FixedPostUpdate, FixedUpdate, Plugin},
    ecs::{
        component::Component,
        resource::Resource,
        schedule::IntoScheduleConfigs,
        system::{Query, Res},
    },
    math::{Vec2, Vec3},
    time::Time,
    transform::components::Transform,
};

#[derive(Resource)]
pub struct SimulatedDeltaTime(f32);

#[derive(Default)]
pub struct PhysicsPlugin {
    pub simulated_delta_time: Option<f32>,
}
impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(
            FixedPostUpdate,
            (
                (Acceleration2::system, Velocity2::system).chain(),
                (AngularAcceleration2::system, AngularVelocity2::system).chain(),
            ),
        );

        if let Some(simulated_delta_time) = self.simulated_delta_time {
            app.insert_resource(SimulatedDeltaTime(simulated_delta_time));
        }
    }
}

#[derive(Component)]
pub struct Area(pub f32);
#[derive(Component)]
pub struct Mass(pub f32);
#[derive(Component)]
#[require(Mass, Area(10.))]
pub struct Drag {
    pub linear_coefficient: f32,
    pub quadratic_coefficient: f32,
    pub angular_lin_coefficient: f32,
    pub angular_quad_coefficient: f32,
}
#[derive(Component, Default, Clone)]
#[require(Velocity2)]
pub struct Acceleration2(pub Vec2);

impl Deref for Acceleration2 {
    type Target = Vec2;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Acceleration2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Component, Default, Clone, Debug)]
#[require(Transform)]
pub struct Velocity2(pub Vec2);
impl Deref for Velocity2 {
    type Target = Vec2;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl DerefMut for Velocity2 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Component, Default)]
#[require(AngularVelocity2)]
pub struct AngularAcceleration2(pub f32);

#[derive(Component, Default)]
#[require(Transform)]
pub struct AngularVelocity2(pub f32);

impl Default for Mass {
    fn default() -> Self {
        Self(1.)
    }
}
impl Default for Drag {
    fn default() -> Self {
        Self {
            linear_coefficient: 0.2,
            quadratic_coefficient: 0.0,
            angular_lin_coefficient: 2.,
            angular_quad_coefficient: 0.5,
        }
    }
}
impl Drag {
    fn apply_to_velocity(&self, v: Vec2, mass: f32, dt: f32) -> Vec2 {
        let (c1, c2) = (self.linear_coefficient, self.quadratic_coefficient);

        // --- Strang splitting for accuracy: half linear -> quadratic -> half linear ---
        let mut v_new = v;

        // half-step linear (exact)
        if c1 > 0.0 {
            let decay = (-0.5 * c1 * dt / mass).exp();
            v_new *= decay;
        }

        // full-step quadratic (exact)
        if c2 > 0.0 {
            let speed = v_new.length();
            if speed > 0.0 {
                let factor = ((c2 / mass) * speed).mul_add(dt, 1.0);
                v_new /= factor;
            }
        }

        // second half-step linear (exact)
        if c1 > 0.0 {
            let decay = (-0.5 * c1 * dt / mass).exp();
            v_new *= decay;
        }

        // Optional micro-threshold
        if v_new.length_squared() < 1e-8 {
            Vec2::ZERO
        } else {
            v_new
        }
    }
}
impl Acceleration2 {
    #[allow(clippy::needless_pass_by_value)]
    pub fn system(
        time: Res<Time>,
        simulated_dt: Option<Res<SimulatedDeltaTime>>,
        query: Query<(&mut Self, &mut Velocity2)>,
    ) {
        let dt = simulated_dt.map_or(time.delta_secs(), |simulated_dt| simulated_dt.0);
        for (mut acceleration, mut velocity) in query {
            velocity.0 += dt * acceleration.0;
            acceleration.0 = Vec2::ZERO;
        }
    }
}
impl Velocity2 {
    pub fn new(x: f32, y: f32) -> Self {
        Self(Vec2::new(x, y))
    }
    #[allow(clippy::needless_pass_by_value)]
    pub fn system(
        time: Res<Time>,
        simulated_dt: Option<Res<SimulatedDeltaTime>>,
        query: Query<(&mut Transform, &mut Self, Option<&Mass>, Option<&Drag>)>,
    ) {
        let dt = simulated_dt.map_or(time.delta_secs(), |simulated_dt| simulated_dt.0);
        for (mut transform, mut velocity, mass, drag) in query {
            // Apply drag (if present)
            if let Some(drag) = drag
                && let Some(mass) = mass
            {
                velocity.0 = drag.apply_to_velocity(velocity.0, mass.0, dt);
            }

            let local_velocity = velocity.0;
            let forward = transform.rotation.mul_vec3(Vec3::X).truncate();
            let right = transform.rotation.mul_vec3(Vec3::Y).truncate();

            let world_velocity = local_velocity.x * forward + local_velocity.y * right;
            transform.translation += dt * Vec3::from((world_velocity, 0.0));
        }
    }
}
impl AngularAcceleration2 {
    #[allow(clippy::needless_pass_by_value)]
    pub fn system(
        time: Res<Time>,
        simulated_dt: Option<Res<SimulatedDeltaTime>>,
        query: Query<(&mut Self, &mut AngularVelocity2)>,
    ) {
        let dt = simulated_dt.map_or(time.delta_secs(), |simulated_dt| simulated_dt.0);
        for (mut acceleration, mut velocity) in query {
            velocity.0 += dt * acceleration.0;
            acceleration.0 = 0.;
        }
    }
}
impl AngularVelocity2 {
    #[allow(clippy::needless_pass_by_value)]
    pub fn system(
        time: Res<Time>,
        simulated_dt: Option<Res<SimulatedDeltaTime>>,
        mut q: Query<(&mut Transform, &mut Self, &Mass, &Area, Option<&Drag>)>,
    ) {
        let dt = simulated_dt.map_or(time.delta_secs(), |simulated_dt| simulated_dt.0);
        for (mut transform, mut omega, mass, area, drag) in &mut q {
            let inertia = (1.0 / 6.0) * mass.0 * area.0;

            if let Some(drag) = drag {
                let w = omega.0;

                // --- Linear drag (stable exact update) ---
                if drag.angular_lin_coefficient > 0.0 {
                    let decay = (-drag.angular_lin_coefficient * dt / inertia).exp();
                    omega.0 *= decay;
                }

                // --- Quadratic drag (explicit) ---
                if drag.angular_quad_coefficient > 0.0 && w != 0.0 {
                    let tau = -drag.angular_quad_coefficient * w.abs() * w; // τ = -c |ω| ω
                    let domega = (tau / inertia) * dt; // Δω = α dt
                    omega.0 += domega;
                }
            }

            // Advance orientation about Z (2D)
            transform.rotate_local_z(omega.0 * dt);

            // Optional: deadzone to kill micro-oscillations
            if omega.0.abs() < 1e-4 {
                omega.0 = 0.0;
            }
        }
    }
}
