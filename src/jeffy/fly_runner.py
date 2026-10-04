"""NeuroMechFly runner for live Drosophila navigation demo.

Runs a fruit fly simulation with chemotaxis navigation, extracts features
each decision step, runs the Jeffy classifier, and yields (frame_jpeg, metadata).
"""

import io
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

STEPS_PER_DECISION = 200
ARENA_RADIUS = 8.0
FOOD_REACH_DIST = 2.5
TURN_THRESHOLD_DEG = 25.0

ACTION_NAMES = ["walk", "turn_left", "turn_right"]
DESCENDING_SIGNALS = {
    "walk":       np.array([1.2, 1.2]),
    "turn_left":  np.array([0.2, 1.6]),
    "turn_right": np.array([1.6, 0.2]),
}

FEATURE_NAMES = [
    "food_distance", "food_angle", "food_angle_sin", "food_angle_cos",
    "forward_velocity", "lateral_velocity", "angular_velocity",
    "heading_sin", "heading_cos",
    "pos_x", "pos_y", "height",
    "action_t1", "action_t2", "action_t3", "action_t4",
]
N_FEATURES = len(FEATURE_NAMES)


def _extract_features(fly_pos, fly_heading, food_pos, velocity, angular_vel, action_history):
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
    features[4] = velocity[0] / 20.0
    features[5] = velocity[1] / 20.0
    features[6] = angular_vel / 10.0
    features[7] = np.sin(fly_heading)
    features[8] = np.cos(fly_heading)
    features[9] = fly_pos[0] / ARENA_RADIUS
    features[10] = fly_pos[1] / ARENA_RADIUS
    features[11] = fly_pos[2]
    for i, a in enumerate(action_history[:4]):
        features[12 + i] = a
    return features, dist, relative_angle


def frame_to_jpeg(pixels, quality=55):
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class FlySession:
    """Manages a Drosophila simulation session for streaming."""

    def __init__(self, engine=None, task_id="fly_navigation"):
        self.engine = engine
        self.task_id = task_id
        self.sim = None
        self.controller = None
        self.renderer = None
        self.fly = None
        self.thorax_idx = None
        self.camera = None
        self.foods = []
        self.action_history = [0.0, 0.0, 0.0, 0.0]
        self.prev_pos = None
        self.prev_heading = None
        self.step_count = 0
        self.foods_reached = 0
        self.rng = np.random.default_rng()

    def start(self):
        import mujoco
        from flygym import Simulation
        from flygym.anatomy import ContactBodiesPreset, BodySegment
        from flygym.compose import FlatGroundWorld
        from flygym_demo.complex_terrain import (
            HybridTurningController, HybridControllerObservation,
            LocomotionAction, PreprogrammedSteps,
            apply_locomotion_action, make_locomotion_fly,
        )
        from flygym.utils.math import Rotation3D

        self._apply_action = apply_locomotion_action
        self._HybridObs = HybridControllerObservation
        self._LocomotionAction = LocomotionAction

        self.fly = make_locomotion_fly(name="dfly", add_adhesion=True, colorize=True)
        world = FlatGroundWorld()
        world.add_fly(
            self.fly, [0, 0, 0.8], Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.TIBIA_TARSUS_ONLY,
            add_ground_contact_sensors=False,
        )
        self.sim = Simulation(world)

        preprogrammed_steps = PreprogrammedSteps()
        self.dof_order = self.fly.get_actuated_jointdofs_order("position")
        self.controller = HybridTurningController(
            timestep=self.sim.timestep,
            preprogrammed_steps=preprogrammed_steps,
            output_dof_order=self.dof_order,
        )
        self.preprogrammed_steps = preprogrammed_steps

        bodysegs = self.fly.get_bodysegs_order()
        self.thorax_idx = bodysegs.index(BodySegment("c_thorax"))

        self.renderer = mujoco.Renderer(self.sim.mj_model, height=240, width=320)

        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 10.0
        self.camera.azimuth = 135
        self.camera.elevation = -35

        self._reset_episode()

    def _reset_episode(self):
        self.sim.reset()
        self.controller.reset(seed=int(time.time() * 1000) % 2**31)

        initial_action = self._LocomotionAction(
            joint_angles=self.preprogrammed_steps.default_pose_by_dof_order(self.dof_order),
            adhesion_onoff=np.ones(6, dtype=bool),
        )
        self._apply_action(self.sim, self.fly.name, initial_action)
        self.sim.warmup()

        for _ in range(500):
            obs = self._HybridObs.from_sim(self.sim, self.fly.name)
            action = self.controller.step(np.array([1.2, 1.2]), obs)
            self._apply_action(self.sim, self.fly.name, action)
            self.sim.step()

        fly_pos = self.sim.get_body_positions(self.fly.name)[self.thorax_idx]
        self.foods = []
        for _ in range(4):
            self._spawn_food(fly_pos[:2])
        self.action_history = [0.0, 0.0, 0.0, 0.0]
        self.prev_pos = None
        self.prev_heading = None
        self.step_count = 0

    def _spawn_food(self, near_pos, heading=None):
        if heading is not None:
            angle = heading + self.rng.uniform(-np.pi / 2, np.pi / 2)
        else:
            angle = self.rng.uniform(0, 2 * np.pi)
        dist = self.rng.uniform(3.0, 6.0)
        self.foods.append(near_pos + np.array([dist * np.cos(angle), dist * np.sin(angle)]))

    def _nearest_food(self, pos_2d):
        best_i, best_dist = 0, float("inf")
        for i, fp in enumerate(self.foods):
            d = np.linalg.norm(fp - pos_2d)
            if d < best_dist:
                best_i, best_dist = i, d
        return best_i, self.foods[best_i], best_dist

    def _get_fly_state(self):
        pos = self.sim.get_body_positions(self.fly.name)[self.thorax_idx]
        quat = self.sim.get_body_rotations(self.fly.name)[self.thorax_idx]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        forward = rot.apply([1, 0, 0])
        heading = np.arctan2(forward[1], forward[0])
        return pos, heading

    def tick(self):
        """Advance one decision step. Returns (jpeg_bytes, metadata_dict) or None."""
        if self.sim is None:
            return None

        import mujoco

        t0 = time.perf_counter()

        fly_pos, fly_heading = self._get_fly_state()
        fly_pos_2d = fly_pos[:2]

        if self.prev_pos is not None:
            dt = STEPS_PER_DECISION * self.sim.timestep
            world_vel = (fly_pos_2d - self.prev_pos) / dt
            cos_h = np.cos(fly_heading)
            sin_h = np.sin(fly_heading)
            forward_vel = world_vel[0] * cos_h + world_vel[1] * sin_h
            lateral_vel = -world_vel[0] * sin_h + world_vel[1] * cos_h
            vel = np.array([forward_vel, lateral_vel])
            angular_vel = (fly_heading - self.prev_heading)
            angular_vel = ((angular_vel + np.pi) % (2 * np.pi) - np.pi) / dt
        else:
            vel = np.array([0.0, 0.0])
            angular_vel = 0.0

        self.prev_pos = fly_pos_2d.copy()
        self.prev_heading = fly_heading

        food_idx, nearest_food, _ = self._nearest_food(fly_pos_2d)
        features, food_dist, food_angle = _extract_features(
            fly_pos, fly_heading, nearest_food, vel, angular_vel, self.action_history
        )

        if self.engine and self.task_id in self.engine.capabilities:
            result = self.engine.predict_features(self.task_id, features.tolist())
            decision = result["label"]
            confidence = result["confidence"]
            probs = result["probabilities"]
            if abs(food_angle) < np.radians(25) and decision != "walk":
                decision = "walk"
                probs = {"walk": confidence, **{k: v for k, v in probs.items() if k != "walk"}}
        else:
            if abs(food_angle) < np.radians(TURN_THRESHOLD_DEG):
                decision = "walk"
            elif food_angle > 0:
                decision = "turn_left"
            else:
                decision = "turn_right"
            confidence = 1.0
            probs = {decision: 1.0}

        ds = DESCENDING_SIGNALS[decision]
        for _ in range(STEPS_PER_DECISION):
            obs = self._HybridObs.from_sim(self.sim, self.fly.name)
            ctrl_action = self.controller.step(ds, obs)
            self._apply_action(self.sim, self.fly.name, ctrl_action)
            self.sim.step()

        self.action_history = [float(ACTION_NAMES.index(decision))] + self.action_history[:3]
        self.step_count += 1
        classify_ms = round((time.perf_counter() - t0) * 1000, 1)

        reached = bool(food_dist < FOOD_REACH_DIST)
        if reached:
            self.foods_reached += 1
            del self.foods[food_idx]
            new_pos, new_heading = self._get_fly_state()
            self._spawn_food(new_pos[:2], heading=new_heading)

        # Render
        self.camera.lookat[:] = fly_pos
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.renderer.update_scene(self.sim.mj_data, camera=self.camera)
        pixels = self.renderer.render()

        jpeg = frame_to_jpeg(pixels)

        meta = {
            "decision": decision,
            "confidence": confidence,
            "probabilities": probs,
            "food_distance": round(float(food_dist), 2),
            "food_angle_deg": round(float(np.degrees(food_angle)), 1),
            "fly_x": round(float(fly_pos[0]), 2),
            "fly_y": round(float(fly_pos[1]), 2),
            "heading_deg": round(float(np.degrees(fly_heading)), 1),
            "speed": round(float(np.linalg.norm(vel)), 2),
            "step": self.step_count,
            "foods_reached": self.foods_reached,
            "food_reached": reached,
            "classify_ms": classify_ms,
            "all_foods": [[round(float(f[0]), 2), round(float(f[1]), 2)] for f in self.foods],
        }

        return jpeg, meta

    def close(self):
        if self.sim:
            self.sim.close()
            self.sim = None
