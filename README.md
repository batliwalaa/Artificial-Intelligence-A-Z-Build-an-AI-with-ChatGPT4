# 🌕 Deep Q-Learning Agent for LunarLander-v3

### \*SuperDataScience: Course Project for: **Artificial Intelligence A-Z: Build an AI with ChatGPT4\***

> 🚀 Train a Deep Q-Network (DQN) from scratch using PyTorch to autonomously land a spacecraft in OpenAI’s classic `LunarLander-v3` environment — now maintained under **Gymnasium (Farama Foundation)**. Includes full training, evaluation, and video recording/playback for Google Colab.

---

## 📖 Overview

This repository implements a **Deep Q-Learning agent** to solve the `LunarLander-v3` environment, which simulates the challenge of landing a lunar module safely on the moon’s surface using discrete thruster controls. The agent must learn when to fire its main engine or left/right orientation thrusters to control descent velocity, angle, and lateral drift — all while minimizing fuel consumption.

According to the [official Gymnasium documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/), an episode is **consigdered solved** when the agent achieves an **average reward of 200 or more over 100 consecutive episodes**.

This project is a complete, self-contained implementation of deep reinforcement learning in action — training a Deep Q-Network from scratch to master the LunarLander-v3 environment using PyTorch, with full video recording and performance visualization.

---

## 🎯 Environment Specification (LunarLander-v3)

### 🎮 Action Space (Discrete)

The agent can choose from **4 discrete actions**:

- `0`: Do nothing
- `1`: Fire left orientation engine (rotates lander counter-clockwise)
- `2`: Fire main engine (slows descent / ascends)
- `3`: Fire right orientation engine (rotates lander clockwise)

> 💡 _Pontryagin’s Maximum Principle_: Optimal control dictates engines should be either fully on or off — hence discrete actions.

---

### 📊 Observation Space (8-Dimensional State Vector)

The agent receives an 8-element vector at each timestep:

| Index | Component          | Description                                      | Range (v3) |
| ----- | ------------------ | ------------------------------------------------ | ---------- |
| 0     | `x`                | Horizontal position (relative to pad at x=0)     | ±2.5       |
| 1     | `y`                | Vertical position (relative to pad at y=0)       | ±2.5       |
| 2     | `vx`               | Horizontal velocity                              | ±10        |
| 3     | `vy`               | Vertical velocity                                | ±10        |
| 4     | `angle`            | Lander’s tilt angle (radians)                    | ±2π        |
| 5     | `angular_velocity` | Rate of angular change (radians/sec, ×2.5 scale) | ±(≈8)      |
| 6     | `leg1_contact`     | Boolean: Left leg touching ground?               | 0 or 1     |
| 7     | `leg2_contact`     | Boolean: Right leg touching ground?              | 0 or 1     |

> ⚠️ **Note**: Angular velocity is reported in units of 0.4 rad/sec — multiply by 2.5 to get true rad/sec.

---

## 🏆 Reward Structure (Shaped for Learning)

At each timestep, the agent receives a shaped reward composed of:

- **Position Bonus**: Higher reward the closer the lander is to the landing pad (x=0, y=0).
- **Velocity Bonus**: Higher reward for slower descent and lateral movement.
- **Angle Penalty**: Penalized for excessive tilt (angle far from horizontal).
- **Leg Contact Bonus**: **+10 points** for each leg touching the ground.
- **Fuel Cost**:
  - **-0.3 points** per frame when main engine fires.
  - **-0.03 points** per frame when left/right engine fires.
- **Terminal Rewards**:
  - **+100 points** for safe landing (comes to rest on pad with low velocity).
  - **-100 points** for crash (lander body touches terrain).

> ✅ **Solved Criterion**: Average score ≥ 200 over 100 episodes.

---

## 🔄 Episode Termination Conditions

An episode ends if any of the following occur:

1. **Crash**: The lander’s main body collides with the terrain.
2. **Out of Bounds**: The lander flies outside the viewport (|x| > 1.0).
3. **Asleep**: The lander “falls asleep” — meaning it stops moving and no longer interacts with the physics world (Box2D sleep state).

---

## 🧠 Agent Architecture & Algorithm

### Deep Q-Network (DQN) Components

- **Neural Network**: 3 fully connected layers (input → 128 → 128 → 4 actions)
- **Experience Replay**: Stores transitions (S, A, R, S', done) in a `deque` buffer; samples mini-batches for stable learning.
- **Target Network**: Separate Q-network updated periodically to stabilize training.
- **Epsilon-Greedy Exploration**: Starts at ε=1.0 (fully random), decays to ε=0.01 (mostly greedy) via ε = ε × 0.995 per episode.
- **Loss Function**: Mean Squared Error (MSE) between predicted Q-values and target Q-values (r + γ·max Q_target).
- **Optimizer**: Adam with learning rate = 0.001.

---

## 🎥 Video Recording & Playback

The trained agent’s performance is recorded and displayed inline in Colab:

1. Environment is re-initialized with `render_mode='rgb_array'`.
2. At each step, `env.render()` returns a frame (H×W×3 numpy array).
3. Frames are collected into a list and saved as `video.mp4` using `imageio.mimsave`.
4. Video is embedded in notebook using `base64` encoding and HTML5 `<video>` tag.

---

## ▶️ How to Run (Google Colab Recommended)

```bash
# Run in first cell
!apt-get install -y swig
!pip install "gymnasium[box2d]" torch matplotlib tqdm imageio
```
