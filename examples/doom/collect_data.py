"""Collect training data for a fire/hold classifier from VizDoom defend_the_center."""

import numpy as np
import vizdoom as vzd

SCREEN_W = 320
SCREEN_H = 240
CENTER_X = SCREEN_W / 2
CENTER_THRESHOLD = 0.15 * SCREEN_W  # 15% of screen width
HISTORY_LEN = 5
ENEMY_NAMES = {"Cacodemon", "MarineChainsawVzd", "HellKnight", "Demon", "Imp",
               "ZombieMan", "ShotgunGuy", "ChaingunGuy", "Revenant", "Arachnotron"}
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


def is_enemy(label):
    name = label.object_name
    if name in IGNORE_NAMES:
        return False
    if name in ENEMY_NAMES:
        return True
    # Treat any unknown non-ignored label as potential enemy
    if name not in IGNORE_NAMES:
        return True
    return False


def extract_features(state, fire_history, reward_history):
    gv = state.game_variables if state.game_variables is not None else [0, 0, 0]
    features = np.zeros(N_FEATURES, dtype=np.float64)

    features[0] = gv[1] if len(gv) > 1 else 0  # health
    features[1] = gv[0] if len(gv) > 0 else 0  # ammo
    features[2] = gv[2] if len(gv) > 2 else 0  # killcount

    # Extract enemies from labels
    enemies = []
    if state.labels is not None:
        for label in state.labels:
            if is_enemy(label):
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

    # History
    for i in range(HISTORY_LEN):
        features[14 + i] = fire_history[i]
        features[19 + i] = reward_history[i]

    return features


def heuristic_action(features):
    """Shoot if primary enemy is roughly centered, else turn toward it.
    Button order from config: 0=TURN_LEFT, 1=TURN_RIGHT, 2=ATTACK
    """
    n_enemies = features[3]
    if n_enemies == 0:
        return 0  # TURN_LEFT to scan

    primary_offset = features[9]
    primary_centered = features[10]

    if primary_centered > 0.5:
        return 2  # ATTACK
    elif primary_offset > 0:
        return 1  # TURN_RIGHT
    else:
        return 0  # TURN_LEFT


def collect_episodes(n_episodes=50, max_steps=2000):
    game = vzd.DoomGame()
    game.load_config(vzd.scenarios_path + '/defend_the_center.cfg')
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_labels_buffer_enabled(True)
    game.add_available_game_variable(vzd.GameVariable.KILLCOUNT)
    game.init()

    all_features = []
    all_labels = []

    for ep in range(n_episodes):
        game.new_episode()
        fire_history = [0.0] * HISTORY_LEN
        reward_history = [0.0] * HISTORY_LEN
        steps = 0

        while not game.is_episode_finished() and steps < max_steps:
            state = game.get_state()
            if state is None:
                break

            feats = extract_features(state, fire_history, reward_history)
            action_idx = heuristic_action(feats)

            action = [0, 0, 0]
            action[action_idx] = 1
            fired = 1.0 if action_idx == 2 else 0.0
            label = "fire" if action_idx == 2 else "hold"

            reward = game.make_action(action)

            fire_history = [fired] + fire_history[:-1]
            reward_history = [reward] + reward_history[:-1]

            all_features.append(feats)
            all_labels.append(label)
            steps += 1

        total_reward = game.get_total_reward()
        kills = game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        print(f"  Episode {ep+1}/{n_episodes}: {steps} steps, {int(kills)} kills, reward={total_reward:.0f}")

    game.close()
    return np.array(all_features), all_labels


if __name__ == "__main__":
    print(f"Collecting data ({N_FEATURES} features)...")
    features, labels = collect_episodes(n_episodes=80)
    print(f"\nCollected {len(features)} frames")
    print(f"  fire: {labels.count('fire')}, hold: {labels.count('hold')}")

    np.savez_compressed(
        "examples/doom/training_data.npz",
        features=features,
        labels=np.array(labels),
        feature_names=np.array(FEATURE_NAMES),
    )
    print("Saved to examples/doom/training_data.npz")
