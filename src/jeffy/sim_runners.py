"""Simulated runners for demos when game engines aren't installed.

Generate synthetic features, run them through the real Jeffy classifiers,
and produce metadata compatible with the live demo WebSocket protocol.
When a recorded GIF is available, plays back real game footage; otherwise
falls back to a simple placeholder.
No VizDoom, flygym, or MuJoCo required.
"""

import io
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)

# ── GIF frame loading ──────────────────────────────────────────────


def _find_gif(name: str) -> Path | None:
    demo_dir = os.environ.get("JEFFY_DEMO_DIR")
    if demo_dir:
        p = Path(demo_dir) / name
        if p.exists():
            return p
    pkg_dir = Path(__file__).parent
    for candidate in [
        pkg_dir / "examples" / name,
        pkg_dir.parent.parent / "examples" / name,
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_gif_frames(gif_path: Path, quality: int = 75) -> list[bytes]:
    gif = Image.open(gif_path)
    frames = []
    try:
        while True:
            frame = gif.copy().convert("RGB")
            buf = io.BytesIO()
            frame.save(buf, format="JPEG", quality=quality)
            frames.append(buf.getvalue())
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass
    logger.info("Loaded %d frames from %s", len(frames), gif_path.name)
    return frames


# ── Doom simulated ──────────────────────────────────────────────────

DOOM_ACTIONS = ["turn_left", "turn_right", "fire"]
DOOM_N_FEATURES = 24
DOOM_W, DOOM_H = 320, 240


def _doom_placeholder(step, decision, health, ammo, enemies):
    img = Image.new("RGB", (DOOM_W, DOOM_H), (20, 20, 30))
    draw = ImageDraw.Draw(img)
    cx, cy = DOOM_W // 2, DOOM_H // 2
    for i, e in enumerate(enemies):
        ex = int(cx + e["offset"] * DOOM_W * 0.4)
        ey = int(cy + 20 - e["size"] * 30)
        sz = int(10 + e["size"] * 25)
        c = (180, 40, 40) if decision == "fire" and i == 0 else (120, 30, 30)
        draw.ellipse([ex - sz, ey - sz, ex + sz, ey + sz], fill=c)
    if decision == "fire":
        draw.line([cx, DOOM_H - 20, cx, cy + 40], fill=(255, 255, 50), width=2)
    draw.rectangle([0, 0, DOOM_W - 1, DOOM_H - 1], outline=(60, 60, 80))
    draw.text((10, 10), "SIMULATED", fill=(255, 200, 50))
    draw.text((10, DOOM_H - 25), f"HP:{health}  AMMO:{ammo}", fill=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=60)
    return buf.getvalue()


class SimDoomSession:
    def __init__(self, engine=None, task_id="doom_fire"):
        self.engine = engine
        self.task_id = task_id
        self.rng = np.random.default_rng()
        self.step = 0
        self.episode = 1
        self.total_kills = 0
        self.health = 100
        self.ammo = 52
        self.kills = 0
        self.fire_history = [0.0] * 5
        self.reward_history = [0.0] * 5
        self.enemies = []
        self._spawn_wave()
        self._gif_frames: list[bytes] = []
        self._frame_idx = 0
        gif_path = _find_gif("doom_battle.gif")
        if gif_path:
            self._gif_frames = _load_gif_frames(gif_path)

    def _spawn_wave(self):
        n = self.rng.integers(1, 4)
        self.enemies = []
        for _ in range(n):
            self.enemies.append({
                "offset": self.rng.uniform(-0.8, 0.8),
                "size": self.rng.uniform(0.3, 1.0),
                "speed": self.rng.uniform(0.005, 0.02),
            })

    def start(self):
        pass

    def tick(self):
        t0 = time.perf_counter()
        self.step += 1

        for e in self.enemies:
            e["offset"] += e.get("drift", 0) + self.rng.uniform(-0.02, 0.02)
            if abs(e["offset"]) > 0.7 or self.rng.random() < 0.08:
                e["drift"] = -0.04 * np.sign(e["offset"]) if abs(e["offset"]) > 0.3 else self.rng.uniform(-0.03, 0.03)
            e["size"] = min(1.2, e["size"] + e["speed"])
            e["offset"] = max(-1.0, min(1.0, e["offset"]))

        enemies = sorted(self.enemies, key=lambda e: e["size"], reverse=True)
        features = np.zeros(DOOM_N_FEATURES, dtype=np.float64)
        features[0] = self.health
        features[1] = self.ammo
        features[2] = self.kills
        features[3] = len(enemies)
        if enemies:
            e = enemies[0]
            cx_norm = 0.5 + e["offset"] * 0.4
            features[4] = cx_norm
            features[5] = 0.5
            features[6] = e["size"] * 0.15
            features[7] = e["size"] * 0.25
            features[8] = features[6] * features[7]
            features[9] = e["offset"] * 0.8
            features[10] = 1.0 if abs(e["offset"]) < 0.15 else 0.0
        if len(enemies) > 1:
            e2 = enemies[1]
            features[11] = 0.5 + e2["offset"] * 0.4
            features[12] = e2["size"] * 0.15
            features[13] = e2["offset"] * 0.8
        for i in range(5):
            features[14 + i] = self.fire_history[i]
            features[19 + i] = self.reward_history[i]

        if self.engine and self.task_id in self.engine.capabilities:
            result = self.engine.predict_features(self.task_id, features.tolist())
            decision = result["label"]
            confidence = result["confidence"]
            probs = result["probabilities"]
        else:
            nearest = enemies[0] if enemies else None
            decision = "fire" if nearest and abs(nearest["offset"]) < 0.2 else "turn_left"
            confidence = 0.8
            probs = {"fire": 0.5, "turn_left": 0.3, "turn_right": 0.2}

        fired = 1.0 if decision == "fire" else 0.0
        reward = 0.0
        nearest = min(self.enemies, key=lambda e: abs(e["offset"])) if self.enemies else None
        if decision == "fire" and nearest and abs(nearest["offset"]) < 0.2 and nearest["size"] > 0.7:
            reward = 1.0
            self.kills += 1
            self.total_kills += 1
            self.enemies.remove(nearest)
            self.ammo = max(0, self.ammo - 1)
        if decision == "fire":
            self.ammo = max(0, self.ammo - 1)

        self.fire_history = [fired] + self.fire_history[:4]
        self.reward_history = [reward] + self.reward_history[:4]

        for e in list(self.enemies):
            if e["size"] > 1.1:
                self.health = max(0, self.health - self.rng.integers(5, 15))
                e["size"] = 0.4

        if not self.enemies or self.rng.random() < 0.05:
            self._spawn_wave()

        if self.health <= 0:
            self.episode += 1
            self.health = 100
            self.ammo = 52
            self.kills = 0
            self.fire_history = [0.0] * 5
            self.reward_history = [0.0] * 5
            self._spawn_wave()

        classify_ms = round((time.perf_counter() - t0) * 1000, 1)

        if self._gif_frames:
            jpeg = self._gif_frames[self._frame_idx % len(self._gif_frames)]
            self._frame_idx += 1
        else:
            jpeg = _doom_placeholder(self.step, decision, self.health, self.ammo, self.enemies)
        action_name = decision.upper()

        meta = {
            "decision": decision,
            "confidence": confidence,
            "action": action_name,
            "probabilities": probs,
            "health": self.health,
            "ammo": self.ammo,
            "kills": self.kills,
            "enemies_visible": len(self.enemies),
            "episode": self.episode,
            "step": self.step,
            "classify_ms": classify_ms,
            "total_kills": self.total_kills,
            "simulated": True,
        }
        return jpeg, meta

    def close(self):
        pass


# ── Fly simulated ───────────────────────────────────────────────────

FLY_ACTIONS = ["walk", "turn_left", "turn_right"]
FLY_N_FEATURES = 16
FLY_ARENA = 8.0
FLY_FOOD_REACH = 2.5
FLY_STEP_DT = 0.2
FLY_WALK_SPEED = 3.0
FLY_TURN_RATE = 1.5
FLY_IMG_W, FLY_IMG_H = 320, 240


def _fly_placeholder(fly_pos, heading_deg, decision, foods_reached):
    img = Image.new("RGB", (FLY_IMG_W, FLY_IMG_H), (25, 35, 25))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), "SIMULATED", fill=(100, 200, 120))
    draw.text((10, FLY_IMG_H - 25), f"Foods: {foods_reached}  Heading: {heading_deg:.0f}°",
              fill=(150, 200, 150))
    cx, cy = FLY_IMG_W // 2, FLY_IMG_H // 2
    draw.ellipse([cx - 6, cy - 3, cx + 6, cy + 3], fill=(80, 180, 100))
    hrad = math.radians(heading_deg)
    hx = cx + int(12 * math.cos(hrad))
    hy = cy - int(12 * math.sin(hrad))
    draw.line([cx, cy, hx, hy], fill=(120, 220, 140), width=2)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=55)
    return buf.getvalue()


class SimFlySession:
    def __init__(self, engine=None, task_id="fly_navigation"):
        self.engine = engine
        self.task_id = task_id
        self.rng = np.random.default_rng()
        self.pos = np.array([0.0, 0.0, 0.6])
        self.heading = self.rng.uniform(-math.pi, math.pi)
        self.vel = np.array([0.0, 0.0])
        self.angular_vel = 0.0
        self.action_history = [0.0, 0.0, 0.0, 0.0]
        self.step_count = 0
        self.foods_reached = 0
        self.foods = []
        for _ in range(4):
            self._spawn_food()
        self._gif_frames: list[bytes] = []
        self._frame_idx = 0
        gif_path = _find_gif("fly_demo.gif")
        if gif_path:
            self._gif_frames = _load_gif_frames(gif_path)

    def _spawn_food(self):
        angle = self.rng.uniform(0, 2 * math.pi)
        dist = self.rng.uniform(3.0, 6.0)
        self.foods.append(np.array([
            self.pos[0] + dist * math.cos(angle),
            self.pos[1] + dist * math.sin(angle),
        ]))

    def start(self):
        pass

    def tick(self):
        if not self.foods:
            return None

        t0 = time.perf_counter()

        pos_2d = self.pos[:2]
        dists = [np.linalg.norm(f - pos_2d) for f in self.foods]
        food_idx = int(np.argmin(dists))
        nearest_food = self.foods[food_idx]
        food_dist = dists[food_idx]

        dx = nearest_food[0] - pos_2d[0]
        dy = nearest_food[1] - pos_2d[1]
        angle_to_food = math.atan2(dy, dx)
        relative_angle = angle_to_food - self.heading
        relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

        features = np.zeros(FLY_N_FEATURES, dtype=np.float64)
        features[0] = food_dist / FLY_ARENA
        features[1] = relative_angle / math.pi
        features[2] = math.sin(relative_angle)
        features[3] = math.cos(relative_angle)
        features[4] = self.vel[0] / 20.0
        features[5] = self.vel[1] / 20.0
        features[6] = self.angular_vel / 10.0
        features[7] = math.sin(self.heading)
        features[8] = math.cos(self.heading)
        features[9] = self.pos[0] / FLY_ARENA
        features[10] = self.pos[1] / FLY_ARENA
        features[11] = self.pos[2]
        for i in range(4):
            features[12 + i] = self.action_history[i]

        if self.engine and self.task_id in self.engine.capabilities:
            result = self.engine.predict_features(self.task_id, features.tolist())
            decision = result["label"]
            confidence = result["confidence"]
            probs = result["probabilities"]
            if abs(relative_angle) < math.radians(25) and decision != "walk":
                decision = "walk"
                probs = {"walk": confidence, **{k: v for k, v in probs.items() if k != "walk"}}
        else:
            if abs(relative_angle) < math.radians(25):
                decision = "walk"
            elif relative_angle > 0:
                decision = "turn_left"
            else:
                decision = "turn_right"
            confidence = 0.9
            probs = {decision: 0.9}

        noise = self.rng.normal(0, 0.05)
        if decision == "walk":
            speed = FLY_WALK_SPEED + self.rng.normal(0, 0.3)
            self.pos[0] += speed * math.cos(self.heading) * FLY_STEP_DT
            self.pos[1] += speed * math.sin(self.heading) * FLY_STEP_DT
            self.heading += noise * FLY_STEP_DT
        elif decision == "turn_left":
            self.heading += FLY_TURN_RATE * FLY_STEP_DT
            self.pos[0] += 0.5 * math.cos(self.heading) * FLY_STEP_DT
            self.pos[1] += 0.5 * math.sin(self.heading) * FLY_STEP_DT
        elif decision == "turn_right":
            self.heading -= FLY_TURN_RATE * FLY_STEP_DT
            self.pos[0] += 0.5 * math.cos(self.heading) * FLY_STEP_DT
            self.pos[1] += 0.5 * math.sin(self.heading) * FLY_STEP_DT

        self.heading = (self.heading + math.pi) % (2 * math.pi) - math.pi
        old_vel = self.vel.copy()
        self.vel = np.array([
            (self.pos[0] - (self.pos[0] - self.vel[0] * FLY_STEP_DT)) / FLY_STEP_DT,
            (self.pos[1] - (self.pos[1] - self.vel[1] * FLY_STEP_DT)) / FLY_STEP_DT,
        ])

        self.action_history = [float(FLY_ACTIONS.index(decision))] + self.action_history[:3]
        self.step_count += 1
        classify_ms = round((time.perf_counter() - t0) * 1000, 1)

        reached = bool(food_dist < FLY_FOOD_REACH)
        if reached:
            self.foods_reached += 1
            del self.foods[food_idx]
            self._spawn_food()

        heading_deg = round(float(math.degrees(self.heading)), 1)
        if self._gif_frames:
            jpeg = self._gif_frames[self._frame_idx % len(self._gif_frames)]
            self._frame_idx += 1
        else:
            jpeg = _fly_placeholder(self.pos, heading_deg, decision, self.foods_reached)

        meta = {
            "decision": decision,
            "confidence": confidence,
            "probabilities": probs,
            "food_distance": round(float(food_dist), 2),
            "food_angle_deg": round(float(math.degrees(relative_angle)), 1),
            "fly_x": round(float(self.pos[0]), 2),
            "fly_y": round(float(self.pos[1]), 2),
            "heading_deg": heading_deg,
            "speed": round(float(np.linalg.norm(self.vel)), 2),
            "step": self.step_count,
            "foods_reached": self.foods_reached,
            "food_reached": reached,
            "classify_ms": classify_ms,
            "all_foods": [[round(float(f[0]), 2), round(float(f[1]), 2)] for f in self.foods],
            "simulated": True,
        }
        return jpeg, meta

    def close(self):
        pass
