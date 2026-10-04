"""Collect training data for a Drosophila navigation classifier.

Runs a NeuroMechFly simulation with a heuristic chemotaxis policy:
the fly navigates toward a food source by turning toward it.
Each decision step records features and the expert's action label.
"""

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from scipy.spatial.transform import Rotation as R

from flygym import Simulation
from flygym.anatomy import ContactBodiesPreset, BodySegment
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    HybridTurningController, HybridControllerObservation,
    LocomotionAction, PreprogrammedSteps,
    apply_locomotion_action, make_locomotion_fly,
)
from flygym.utils.math import Rotation3D

STEPS_PER_DECISION = 1000     # physics steps between decisions (0.1s)
ARENA_RADIUS = 8.0             # mm, how far food can spawn
FOOD_REACH_DIST = 1.5          # mm, close enough = reached
TURN_THRESHOLD_DEG = 25.0      # within this angle = walk forward

ACTION_NAMES = ["walk", "turn_left", "turn_right"]
DESCENDING_SIGNALS = {
    "walk":       np.array([1.2, 1.2]),
    "turn_left":  np.array([0.4, 1.4]),   # less left, more right → turn left
    "turn_right": np.array([1.4, 0.4]),   # more left, less right → turn right
}

FEATURE_NAMES = [
    "food_distance", "food_angle", "food_angle_sin", "food_angle_cos",
    "forward_velocity", "lateral_velocity", "angular_velocity",
    "heading_sin", "heading_cos",
    "pos_x", "pos_y", "height",
    "action_t1", "action_t2", "action_t3", "action_t4",
]
N_FEATURES = len(FEATURE_NAMES)


def get_fly_state(sim, fly, thorax_idx):
    positions = sim.get_body_positions(fly.name)
    rotations = sim.get_body_rotations(fly.name)
    pos = positions[thorax_idx]
    quat = rotations[thorax_idx]  # (w, x, y, z)
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    forward = rot.apply([1, 0, 0])
    heading = np.arctan2(forward[1], forward[0])
    return pos, heading, forward


def extract_features(fly_pos, fly_heading, food_pos, velocity, angular_vel, action_history):
    features = np.zeros(N_FEATURES, dtype=np.float64)

    dx = food_pos[0] - fly_pos[0]
    dy = food_pos[1] - fly_pos[1]
    dist = np.sqrt(dx * dx + dy * dy)
    angle_to_food = np.arctan2(dy, dx)
    relative_angle = angle_to_food - fly_heading
    relative_angle = (relative_angle + np.pi) % (2 * np.pi) - np.pi

    features[0] = dist / ARENA_RADIUS
    features[1] = relative_angle / np.pi
    features[2] = np.sin(relative_angle)
    features[3] = np.cos(relative_angle)
    features[4] = velocity[0] / 20.0  # normalize by typical walking speed
    features[5] = velocity[1] / 20.0
    features[6] = angular_vel / 10.0  # normalize angular velocity
    features[7] = np.sin(fly_heading)
    features[8] = np.cos(fly_heading)
    features[9] = fly_pos[0] / ARENA_RADIUS
    features[10] = fly_pos[1] / ARENA_RADIUS
    features[11] = fly_pos[2]
    for i, a in enumerate(action_history[:4]):
        features[12 + i] = a

    return features


def heuristic_action(features):
    relative_angle = features[1] * np.pi
    if abs(relative_angle) < np.radians(TURN_THRESHOLD_DEG):
        return 0  # walk
    elif relative_angle > 0:
        return 1  # turn_left
    else:
        return 2  # turn_right


def collect_episodes(n_episodes=30, max_decisions=200):
    fly = make_locomotion_fly(name="dfly", add_adhesion=True, colorize=True)
    world = FlatGroundWorld()
    world.add_fly(
        fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
        bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
        add_ground_contact_sensors=False,
    )
    sim = Simulation(world)

    preprogrammed_steps = PreprogrammedSteps()
    dof_order = fly.get_actuated_jointdofs_order("position")
    controller = HybridTurningController(
        timestep=sim.timestep, preprogrammed_steps=preprogrammed_steps,
        output_dof_order=dof_order,
    )

    bodysegs = fly.get_bodysegs_order()
    thorax_idx = bodysegs.index(BodySegment("c_thorax"))

    all_features = []
    all_labels = []
    rng = np.random.default_rng(42)

    for ep in range(n_episodes):
        sim.reset()
        controller.reset(seed=ep)

        initial_action = LocomotionAction(
            joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        apply_locomotion_action(sim, fly.name, initial_action)
        sim.warmup()

        # Warmup walking for stability
        for _ in range(500):
            obs = HybridControllerObservation.from_sim(sim, fly.name)
            action = controller.step(np.array([1.0, 1.0]), obs)
            apply_locomotion_action(sim, fly.name, action)
            sim.step()

        action_history = [0.0, 0.0, 0.0, 0.0]
        prev_pos = None
        prev_heading = None
        decisions = 0
        foods_reached = 0

        # Place first food relative to starting position
        fly_start, _, _ = get_fly_state(sim, fly, thorax_idx)
        food_angle = rng.uniform(0, 2 * np.pi)
        food_dist = rng.uniform(3.0, ARENA_RADIUS)
        food_pos = fly_start[:2] + np.array([food_dist * np.cos(food_angle),
                                              food_dist * np.sin(food_angle)])

        for dec in range(max_decisions):
            fly_pos, fly_heading, _ = get_fly_state(sim, fly, thorax_idx)
            fly_pos_2d = fly_pos[:2]

            if prev_pos is not None:
                dt = STEPS_PER_DECISION * sim.timestep
                world_vel = (fly_pos_2d - prev_pos) / dt
                cos_h = np.cos(fly_heading)
                sin_h = np.sin(fly_heading)
                forward_vel = world_vel[0] * cos_h + world_vel[1] * sin_h
                lateral_vel = -world_vel[0] * sin_h + world_vel[1] * cos_h
                vel = np.array([forward_vel, lateral_vel])
                angular_vel = (fly_heading - prev_heading)
                angular_vel = ((angular_vel + np.pi) % (2 * np.pi) - np.pi) / dt
            else:
                vel = np.array([0.0, 0.0])
                angular_vel = 0.0

            prev_pos = fly_pos_2d.copy()
            prev_heading = fly_heading

            feats = extract_features(fly_pos, fly_heading, food_pos, vel, angular_vel, action_history)
            action_idx = heuristic_action(feats)
            label = ACTION_NAMES[action_idx]

            all_features.append(feats)
            all_labels.append(label)

            action_history = [float(action_idx)] + action_history[:3]

            ds = DESCENDING_SIGNALS[label]
            for _ in range(STEPS_PER_DECISION):
                obs = HybridControllerObservation.from_sim(sim, fly.name)
                ctrl_action = controller.step(ds, obs)
                apply_locomotion_action(sim, fly.name, ctrl_action)
                sim.step()

            decisions += 1

            d = np.linalg.norm(fly_pos_2d - food_pos)
            if d < FOOD_REACH_DIST:
                foods_reached += 1
                food_angle = rng.uniform(0, 2 * np.pi)
                food_dist = rng.uniform(3.0, ARENA_RADIUS)
                food_pos = fly_pos_2d + np.array([food_dist * np.cos(food_angle),
                                                   food_dist * np.sin(food_angle)])

        print(f"  Episode {ep+1}/{n_episodes}: {decisions} decisions, "
              f"{foods_reached} foods reached")

    sim.close()
    return np.array(all_features), all_labels


if __name__ == "__main__":
    print(f"Collecting fly navigation data ({N_FEATURES} features)...")
    features, labels = collect_episodes(n_episodes=30, max_decisions=200)
    print(f"\nCollected {len(features)} samples")
    for name in ACTION_NAMES:
        print(f"  {name}: {labels.count(name)}")

    np.savez_compressed(
        "examples/fly/training_data.npz",
        features=features,
        labels=np.array(labels),
        feature_names=np.array(FEATURE_NAMES),
    )
    print("Saved to examples/fly/training_data.npz")
