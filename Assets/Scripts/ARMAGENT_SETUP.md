# ArmAgent Setup Guide

## Scene Hierarchy Requirements

Your SampleScene must have this GameObject structure:

```
RobotArm (root, contains ArmAgent script)
├── ShoulderYaw (Rigidbody + HingeJoint)
├── ShoulderPitch (Rigidbody + HingeJoint)
├── ElbowPitch (Rigidbody + HingeJoint)
├── Mallet (Rigidbody, child of ElbowPitch chain)
└── Target (Transform, physics Rigidbody for collision)
```

---

## Joint Configuration (Critical for Physics Control)

Each joint (ShoulderYaw, ShoulderPitch, ElbowPitch) must be a **Rigidbody** with a **HingeJoint** component.

### Per-Joint Setup:

1. **Rigidbody Component:**
  - Mass: 1.0 (adjust for realism)
  - Drag: 0.1
  - Angular Drag: 0.1
  - Use Gravity: ✓ enabled
  - Constraints: Freeze rotation Y and Z (leave X free for hinge)

2. **HingeJoint Component:**
  - Connected Body: parent segment (or leave empty for base)
  - Axis: (1, 0, 0) [X-axis rotation]
  - Use Limits: ✓ enable and set Min/Max to joint angle ranges
  - Motor:
    - Target Velocity: controlled by the agent (deg/sec)
    - Force: set a high enough value (e.g., 100–1000) to let the motor move the link
    - Free Spin: usually false
  - Use Motor: enabled at runtime by `ArmAgent` when applying actions

### Suggested Joint Angle Limits:

- **Shoulder Yaw:** Min -90°, Max +90° (left-right sweep)
- **Shoulder Pitch:** Min -30°, Max +90° (forward-back reach)
- **Elbow Pitch:** Min -120°, Max +0° (full extension to vertical)

---

## Mallet Configuration

The **Mallet** should be:
- Child of ElbowPitch (at the end of the kinematic chain)
- Rigidbody (Mass: 0.5, Drag: 0.1, Angular Drag: 0.1)
- **Collider:** Capsule or Box Collider (NOT trigger)
- Material: High friction (0.8+)
- Position: Offset downward from ElbowPitch (e.g., 0, -0.3, 0)

---

## Target Configuration

The **Target** sphere should be:
- Rigidbody (Kinematic or static, Mass doesn't matter)
- **Collider:** Sphere Collider (NOT trigger initially, or use trigger with OnTriggerEnter)
- Starting Position: e.g., (0, 1.0, 1.0)
- Physics Material: Default (or low friction for clean strikes)

**For v1:** The target is stationary. In later phases, it can spawn in different holes.

---

## ArmAgent Script Assignment

1. Add the **ArmAgent** script to the **RobotArm** root GameObject
2. In the Inspector, assign:
   - **Shoulder Yaw Rigidbody:** drag ShoulderYaw GameObject
   - **Shoulder Pitch Rigidbody:** drag ShoulderPitch GameObject
   - **Elbow Pitch Rigidbody:** drag ElbowPitch GameObject
   - **Mallet Rigidbody:** drag Mallet GameObject
   - **Target Transform:** drag Target GameObject

---

## Behavior Component

Ensure the **RobotArm** GameObject has:
- **ArmAgent** script (already attached)
- **Decision Requester** (from ML-Agents)
  - Decision Period: 5 (agent decides every 5 frames)
  - Take Actions Between Decisions: ✓ enabled
- **Model** (drag trained model once available, or leave empty for training)

---

## Training Configuration (WhackRL.yaml)

Create/modify `Training/WhackRL.yaml`:

```yaml
behaviors:
  WhackRL:
    trainer_type: ppo
    hyperparameters:
      batch_size: 1024
      buffer_size: 10240
      learning_rate: 3.0e-4
      beta: 5.0e-4
      epsilon: 0.2
      lambd: 0.99
      num_epoch: 3
      learning_rate_schedule: linear
    network_settings:
      normalize_input: true
      hidden_units: 128
      num_layers: 2
      activation: relu
      memory: null
    reward_signals:
      extrinsic:
        strength: 1.0
        gamma: 0.99
    max_steps: 500000
    time_horizon: 64
    summary_freq: 5000
    keep_checkpoints: 5
    checkpoint_interval: 50000
    threaded: true
```

---

## ML-Agents Config

1. **Update .yaml behavior name** from "MoveToTarget" to "WhackRL" to match your agent
2. Run training:
   ```
   mlagents-learn Training/WhackRL.yaml --run-id=WhackRL_v1 --time-scale=10
   ```

---

## Testing Checklist

- [ ] Scene hierarchy is correct
- [ ] All joint Rigidbodies have HingeJoint components
- [ ] ArmAgent references are all assigned in Inspector
- [ ] Mallet has a non-trigger Collider
- [ ] Target has a Rigidbody and Collider
- [ ] DecisionRequester is on RobotArm
- [ ] Training YAML exists and behavior name matches agent script
- [ ] Play scene: arm moves in response to actions (expect random chaos initially)
- [ ] Use "Print Current Observations" context menu to verify state is sensible
- [ ] Use "Trigger Manual Strike Test" to verify strike detection

---

## Expected Training Curve (v1)

- **First 10k steps:** Random thrashing, occasional lucky strikes
- **10k–50k steps:** Agent learns to move arm toward target
- **50k–150k steps:** Agent discovers how to strike (positive reward jumps)
- **150k–300k steps:** Agent refines velocity control and positioning
- **300k+ steps:** Convergence on efficient strike strategy

Monitor via TensorBoard:
```
tensorboard --logdir=results
```

---

## Debug Output

The ArmAgent prints:
- "Episode started" at each reset
- "VALID STRIKE!" when a successful hit is detected (with velocity stats)
- "Invalid strike" when a soft/bad-angle hit occurs
- Warnings if joint references are missing

Watch the Console tab in Unity while training to verify physics and rewards are firing correctly.

---

## Next Steps (Post-v1)

1. Add second target position randomization
2. Add bomb target (negative reward)
3. Implement curriculum learning progression
4. Add vision observations (camera)
5. Domain randomization (joint mass, friction)
