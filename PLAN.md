# WhackRL — Physics-Based Robotic Arm for Reactive Striking

## Overview

WhackRL is a Unity + ML-Agents simulation where a 3-DOF physics-based robotic arm equipped with a mallet learns to play Whack-a-Mole.  
The agent must strike appearing moles with sufficient force while avoiding bombs, learning reactive manipulation, velocity control, and risk-aware decision making via reinforcement learning.

Core goals:
- Continuous control of a multi-joint robot in Unity physics
- Learn high-speed downward strikes (not slow pushes)
- Discriminate targets (mole vs bomb)
- Demonstrate reward shaping, curriculum learning, and contact dynamics

---

## System Architecture

Unity Environment:
- 3 DOF robotic arm (Rigidbody + ConfigurableJoint)
- Fixed vertical mallet
- Whack-a-mole board with multiple holes
- Random mole/bomb spawner

Learning:
- Unity ML-Agents (PPO)
- Continuous action space
- Vector observations (optionally camera later)
- TensorBoard for training metrics

Control Loop:
State → Policy → Joint Targets → Physics → Collision → Reward

---

## Robot

Degrees of Freedom:
- Base yaw (left/right)
- Shoulder pitch (forward/back)
- Elbow pitch (reach/height)

Control:
- Agent outputs Δ target joint angles per step
- ConfigurableJoint angular drives (high spring, moderate damping)
- Gravity enabled

End Effector:
- Mallet rigidbody
- Used to measure impact velocity

---

## Observations (State Space)

Robot:
- Joint angles (normalized)
- Joint angular velocities
- End effector position
- End effector velocity

Environment (per hole):
- Active (0/1)
- IsBomb (0/1)
- Mole height (0–1)
- Time since spawn

Optional later:
- Camera observation for vision-based policy

---

## Actions

Continuous vector:
- Δ target rotation for each joint (mapped from [-1, 1])

---

## Strike Validation (Prevent Side Hits)

A hit only counts if ALL are true:

1. Collision occurs with top collider of mole
2. Collision normal is mostly upward:
   dot(normal, Vector3.up) > threshold
3. Mallet velocity is mostly downward:
   dot(velocity, Vector3.down) > threshold
4. Impact velocity exceeds minimum strike speed

This enforces true hammer-style downward strikes.

---

## Reward Function (Core)

- +1.0 for valid mole hit
- +0.2 * impactVelocity bonus
- −2.0 for bomb hit
- −0.001 per timestep (time pressure)
- −distanceToActiveMole * small factor (shaping)

Soft hits below velocity threshold give zero reward.

Optional:
- Penalty for staying in low posture (encourage recovery)

---

## Training Curriculum

### Phase 1 — Reach
- Static target sphere
- Learn basic kinematics

Reward: −distance

---

### Phase 2 — Single Mole Strike
- One mole, fixed position
- Learn downward striking + velocity

Reward: contact + impact velocity

---

### Phase 3 — Random Mole
- Mole appears in random holes
- Learn targeting and reaction

---

### Phase 4 — Bombs
- Bombs randomly replace moles
- Penalize bomb hits
- Learn discrimination

---

### Phase 5 — Force Control (Optional)
- Require minimum impact
- Penalize excessive force

---

## Episode Reset

End episode if:
- Bomb hit
- Time limit reached

Reset:
- Arm pose
- Mole layout
- Randomized physics parameters (optional domain randomization)

---

## Metrics to Track

- Mean episode reward
- Mole hit rate
- Bomb hit rate
- Average reaction time
- Average impact velocity

Visualize via TensorBoard.

---

## Expected Learned Behavior

- Arm retracts upward before striking
- Fast downward mallet swings
- Recovery posture between hits
- Hesitation/avoidance near bombs

---

## Optional Extensions

- Vision-based policy (CNN + camera)
- Domain randomization (mass, gravity, friction)
- EMG or keyboard imitation learning
- IK baseline comparison
- Training playback + reward graphs in Unity

---

## Portfolio Focus

Demonstrates:
- Physics-based robotic control
- Reinforcement learning (PPO)
- Contact dynamics + velocity optimization
- Risk-aware decision making
- Curriculum learning
- Robotics-style manipulation pipeline
