# Whack-A-ML: Reinforcement Learning Robotic Arm (WIP)

Whack-A-ML is a Unity ML-Agents project demonstrating reinforcement learning for a physics-based 3-DOF robotic arm. The agent learns to strike a dynamic target (“mole”) using continuous control, reward shaping, and contact-based feedback, emphasizing impact velocity, approach geometry, and recovery behavior.

This project focuses on reactive manipulation and physics-driven control rather than traditional gameplay, serving as a compact robotics-style ML simulation.

---

## Overview

Whack-A-ML showcases applied reinforcement learning in a real-time physics environment, covering:

- Continuous joint control of a multi-DOF robot
- Reward engineering for velocity-based contact tasks
- Observation design for stable learning
- Curriculum-style iteration and hyperparameter tuning
- Debugging and validation of learned behaviors

The goal is to demonstrate practical ML system design in Unity, combining robotics concepts (kinematics, contact dynamics) with PPO-based reinforcement learning.

---

## Demo

<p align="center" width="100%">
  <video src="https://raw.githubusercontent.com/HasNate618/Whack-a-ML/master/Media/Demo1.mp4" width="80%" controls></video>
</p>

---

## ML Design

### Observations

The agent receives a compact vector-based state:

- Joint angles and angular velocities (normalized)
- Mallet world position and linear velocity
- Relative target position
- Scalar distance to active target

This balances expressiveness with training stability while avoiding vision-based input for faster iteration.

---

### Actions

- 3 continuous actions controlling target angular velocities for each joint
- Outputs are smoothed via acceleration limits to reduce jitter and improve physical realism

---

### Rewards

- **Hit Reward**: Base reward for valid target contact
- **Impact Bonus**: Scaled by mallet collision velocity to encourage decisive strikes
- **Time Penalty**: Small per-step cost to promote fast reactions
- **Progress Shaping**: Delta-distance reward for moving closer to the active target
- **Top-Hit Bonus**: Additional reward for downward-aligned strikes, validated using collision normals and velocity direction

Soft or lateral contacts do not receive credit.

---

### Training Practices

- Randomized target spawn positions for generalization
- Per-agent target instancing for multi-agent scalability
- Bounded reward bonuses to avoid policy instability
- Iterative curriculum (reach → strike → dynamic targets)

Training uses PPO via Unity ML-Agents with TensorBoard monitoring.