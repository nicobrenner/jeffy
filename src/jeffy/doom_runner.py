"""VizDoom runner for live classifier demo.

Runs defend_the_center headless, extracts features each frame,
runs the Jeffy classifier, and yields (frame_jpeg, decision_metadata).
"""

import io
import time

import numpy as np
from PIL import Image

SCREEN_W = 320
SCREEN_H = 240
CENTER_X = SCREEN_W / 2
CENTER_THRESHOLD = 0.15 * SCREEN_W
HISTORY_LEN = 5
IGNORE_NAMES = {"DoomPlayer", "BulletPuff", "Blood", "Clip", "RocketAmmo",
                "CellPack", "Medikit", "Stimpack", "GreenArmor", "BlueArmor"}
FEATURE_NAMES = [
    "health", "ammo", "killcount",
    "n_enemies",
    "primary_cx", "primary_cy", "primary_width", "primary_height",
    "primary_area", "primary_offset", "primary_centered",
    "secondary_cx", "secondary_width", "secondary_offset",
    "fired_t1", "fired_t2", "fired_t3", "fired_t4", "fired_t5",
    "reward_t1", "reward_t2", "reward_t3", "reward_t4", "reward_t5",
]
N_FEATURES = len(FEATURE_NAMES)

ACTION_NAMES = ["TURN_LEFT", "TURN_RIGHT", "ATTACK"]


def _is_enemy(label):
    return label.object_name not in IGNORE_NAMES


def extract_features(state, fire_history, reward_history):
    gv = state.game_variables if state.game_variables is not None else [0, 0, 0]
    features = np.zeros(N_FEATURES, dtype=np.float64)

    features[0] = gv[1] if len(gv) > 1 else 0
    features[1] = gv[0] if len(gv) > 0 else 0
    features[2] = gv[2] if len(gv) > 2 else 0

    enemies = []
    if state.labels is not None:
        for label in state.labels:
            if _is_enemy(label):
                cx = label.x + label.width / 2
                cy = label.y + label.height / 2
                area = label.width * label.height
                offset = cx - CENTER_X
                enemies.append({
                    "cx": cx, "cy": cy,
                    "width": label.width, "height": label.height,
                    "area": area, "offset": offset,
                    "centered": 1.0 if abs(offset) < CENTER_THRESHOLD else 0.0,
                })

    enemies.sort(key=lambda e: e["area"], reverse=True)
    features[3] = len(enemies)

    if len(enemies) >= 1:
        e = enemies[0]
        features[4] = e["cx"] / SCREEN_W
        features[5] = e["cy"] / SCREEN_H
        features[6] = e["width"] / SCREEN_W
        features[7] = e["height"] / SCREEN_H
        features[8] = e["area"] / (SCREEN_W * SCREEN_H)
        features[9] = e["offset"] / CENTER_X
        features[10] = e["centered"]

    if len(enemies) >= 2:
        e = enemies[1]
        features[11] = e["cx"] / SCREEN_W
        features[12] = e["width"] / SCREEN_W
        features[13] = e["offset"] / CENTER_X

    for i in range(HISTORY_LEN):
        features[14 + i] = fire_history[i]
        features[19 + i] = reward_history[i]

    return features, enemies


def frame_to_jpeg(screen_buffer, quality=55, decision=None):
    img = Image.fromarray(screen_buffer)
    if decision == "fire":
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        w, h = img.size
        t = 3
        for i in range(t):
            draw.rectangle([i, i, w - 1 - i, h - 1 - i], outline=(255, 50, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


class DoomSession:
    """Manages a single VizDoom game session for streaming."""

    def __init__(self, engine=None, task_id="doom_fire"):
        self.engine = engine
        self.task_id = task_id
        self.game = None
        self.fire_history = [0.0] * HISTORY_LEN
        self.reward_history = [0.0] * HISTORY_LEN
        self.episode = 0
        self.step = 0
        self.total_kills = 0

    def start(self):
        import vizdoom as vzd
        self.game = vzd.DoomGame()
        self.game.load_config(vzd.scenarios_path + '/defend_the_center.cfg')
        self.game.set_window_visible(False)
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_labels_buffer_enabled(True)
        self.game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
        self.game.init()
        self.game.new_episode()
        self.episode = 1
        self.step = 0

    def tick(self):
        """Advance one frame. Returns (jpeg_bytes, metadata_dict) or None if done."""
        if self.game is None:
            return None

        if self.game.is_episode_finished():
            self.game.new_episode()
            self.episode += 1
            self.step = 0
            self.fire_history = [0.0] * HISTORY_LEN
            self.reward_history = [0.0] * HISTORY_LEN

        state = self.game.get_state()
        if state is None:
            return None

        t0 = time.perf_counter()
        features, enemies = extract_features(state, self.fire_history, self.reward_history)

        if self.engine and self.task_id in self.engine.capabilities:
            result = self.engine.predict_features(self.task_id, features.tolist())
            decision = result["label"]
            confidence = result["confidence"]
            probs = result["probabilities"]
        else:
            decision = "turn_left"
            confidence = 0.0
            probs = {}

        action_map = {"turn_left": [1, 0, 0], "turn_right": [0, 1, 0], "fire": [0, 0, 1]}
        action = action_map.get(decision, [0, 1, 0])
        action_name = ACTION_NAMES[action.index(1)]
        fired = 1.0 if action[0] == 1 else 0.0

        reward = self.game.make_action(action, 4)
        classify_ms = round((time.perf_counter() - t0) * 1000, 1)

        self.fire_history = [fired] + self.fire_history[:-1]
        self.reward_history = [reward] + self.reward_history[:-1]
        self.step += 1

        gv = state.game_variables if state.game_variables is not None else [0, 0, 0]
        kills = int(gv[2]) if len(gv) > 2 else 0
        self.total_kills = max(self.total_kills, kills)

        jpeg = frame_to_jpeg(state.screen_buffer, decision=decision)

        meta = {
            "decision": decision,
            "confidence": confidence,
            "action": action_name,
            "probabilities": probs,
            "health": int(gv[1]) if len(gv) > 1 else 0,
            "ammo": int(gv[0]) if len(gv) > 0 else 0,
            "kills": kills,
            "enemies_visible": len(enemies),
            "episode": self.episode,
            "step": self.step,
            "classify_ms": classify_ms,
            "total_kills": self.total_kills,
        }

        return jpeg, meta

    def close(self):
        if self.game:
            self.game.close()
            self.game = None
