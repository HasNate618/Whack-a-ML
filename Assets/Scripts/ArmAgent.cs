using UnityEngine;
using System.Collections;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// ArmAgent controls a 3-DOF robotic arm using custom kinematic rotation physics.
/// Each joint has configurable angle limits, max speed, and acceleration.
/// The agent outputs target velocities [-1, 1] which are smoothed via acceleration.
/// 
/// Observations (17 total):
///   - Joint angles (3) normalized by limits
///   - Joint angular velocities (3) normalized by max speed
///   - Mallet world position (3)
///   - Mallet world velocity (3)
///   - Target relative position (3)
///   - Distance to target (1)
///
/// Actions (3 continuous):
///   - Target angular velocity for each joint [-1, 1] mapped to [-maxSpeed, +maxSpeed]
///
/// Rewards:
///   - +1.0 for valid mole strike
///   - +0.2 × impact velocity bonus
///   - -0.001 per step (time pressure)
///   - -0.01 × normalized distance (shaping)
/// </summary>
public class ArmAgent : Agent
{
    // ============================================================================
    // JOINT CONFIGURATION (per-joint settings)
    // ============================================================================

    [System.Serializable]
    public class JointConfig
    {
        [Tooltip("The Transform to rotate (pivot point of this joint)")]
        public Transform jointTransform;

        [Tooltip("Local rotation axis for this joint (e.g., Vector3.up for yaw)")]
        public Vector3 rotationAxis = Vector3.right;

        [Tooltip("Minimum angle in degrees (local space)")]
        public float minAngle = -90f;

        [Tooltip("Maximum angle in degrees (local space)")]
        public float maxAngle = 90f;

        [Tooltip("Maximum angular speed in degrees/sec")]
        public float maxSpeed = 180f;

        [Tooltip("Angular acceleration in degrees/sec²")]
        public float acceleration = 720f;

        [Tooltip("Starting angle in degrees")]
        public float startAngle = 0f;

        // Runtime state
        [HideInInspector] public float currentAngle;
        [HideInInspector] public float currentVelocity;
    }

    [Header("Joint Configs (Shoulder Yaw, Shoulder Pitch, Elbow Pitch)")]
    [SerializeField] private JointConfig[] joints = new JointConfig[3];

    [Header("Mallet & Target")]
    [SerializeField] private Transform malletTransform;
    [SerializeField] private Rigidbody malletRigidbody;
    [SerializeField] private Transform targetTransform;

    // Public accessors for helper components (used by TargetHitProxy)
    public Rigidbody GetMalletRigidbody() { return malletRigidbody; }
    public Transform GetMalletTransform() { return malletTransform; }

    // ============================================================================
    // STRIKE VALIDATION
    // ============================================================================

    [Header("Strike Validation")]
    [SerializeField] private float minImpactVelocity = 2f;
    [SerializeField] private float strikeNormalThreshold = 0.5f;
    [SerializeField] private float strikeVelocityDownwardThreshold = 0.5f;
    [Tooltip("Minimum allowed floor Y for the mallet (prevents going below floor)")]
    [SerializeField] private float floorY = 0f;

    // ============================================================================
    // REWARDS
    // ============================================================================

    [Header("Rewards")]
    [SerializeField] private float rewardValidStrike = 1.5f;
    [SerializeField] private float rewardVelocityMultiplier = 0.5f;
    [SerializeField] private float rewardMaxVelocityBonus = 5f;
    [SerializeField] private float rewardPenaltyPerStep = 0.001f;
    [SerializeField] private float rewardShapingDistance = 0.01f;
    [SerializeField] private float penaltyBombHit = 2f;

    // ============================================================================
    // EPISODE SETTINGS
    // ============================================================================

    [Header("Episode Settings")]
    [SerializeField] private int maxStepsPerEpisode = 500;
    [SerializeField] private Vector3 targetStartPosition = new Vector3(0, 0.5f, 1f);
    [SerializeField] private float targetResetHeight = 0.5f;
    [SerializeField] private float targetRandomRadius = 1.0f;

    // ============================================================================
    // RUNTIME STATE
    // ============================================================================

    private int stepsSinceLastReset = 0;
    private bool strikeValidatedThisFrame = false;
    private float lastImpactVelocity = 0f;

    private bool pendingRandomTarget = false;
    // Pending randomized target local offset (local to the agent's transform)
    private Vector3 pendingRandomLocalOffset = Vector3.zero;

    // Mallet velocity tracking (since we control joints kinematically)
    private Vector3 lastMalletPosition;
    private Vector3 computedMalletVelocity;
    // Mallet local offset relative to the end joint (elbow)
    private Vector3 malletLocalPosition;
    private Quaternion malletLocalRotation;

    // ============================================================================
    // HELPERS
    // ============================================================================

    private static float SafeFloat(float v)
    {
        if (float.IsNaN(v) || float.IsInfinity(v)) return 0f;
        return v;
    }

    private static Vector3 SafeVector3(Vector3 v)
    {
        v.x = SafeFloat(v.x);
        v.y = SafeFloat(v.y);
        v.z = SafeFloat(v.z);
        return v;
    }

    /// <summary>
    /// Normalize an angle to the range [-180, 180].
    /// </summary>
    private static float NormalizeAngle(float angle)
    {
        while (angle > 180f) angle -= 360f;
        while (angle < -180f) angle += 360f;
        return angle;
    }

    // ============================================================================
    // INITIALIZATION
    // ============================================================================

    public override void Initialize()
    {
        if (joints == null || joints.Length != 3)
        {
            Debug.LogError("ArmAgent: Exactly 3 JointConfigs required.");
            enabled = false;
            return;
        }

        foreach (var j in joints)
        {
            if (j.jointTransform == null)
            {
                Debug.LogError("ArmAgent: A joint Transform is not assigned.");
                enabled = false;
                return;
            }
        }

        if (malletTransform == null || targetTransform == null)
        {
            Debug.LogError("ArmAgent: Mallet or Target Transform not assigned.");
            enabled = false;
            return;
        }

        // Initialize joint states
        for (int i = 0; i < joints.Length; i++)
        {
            joints[i].currentAngle = joints[i].startAngle;
            joints[i].currentVelocity = 0f;
        }

        lastMalletPosition = malletTransform.position;
        computedMalletVelocity = Vector3.zero;

        // Record mallet's initial local offset relative to the last joint so we can
        // restore that exact pose each physics tick (prevents snapping issues).
        if (joints != null && joints.Length > 0 && malletTransform != null)
        {
            var endJoint = joints[joints.Length - 1];
            malletLocalPosition = endJoint.jointTransform.InverseTransformPoint(malletTransform.position);
            malletLocalRotation = Quaternion.Inverse(endJoint.jointTransform.rotation) * malletTransform.rotation;
        }

        // Ensure the target is unique per agent. If the assigned Target is shared
        // (not a child), instantiate a clone as this agent's child so agents don't
        // overwrite a global target when randomized.
        if (targetTransform != null && !targetTransform.IsChildOf(this.transform))
        {
            GameObject newT = GameObject.Instantiate(targetTransform.gameObject, this.transform);
            newT.name = targetTransform.gameObject.name + "_agent";
            targetTransform = newT.transform;
            Debug.Log($"[ArmAgent] Cloned target for per-agent use: {targetTransform.name}");
        }

        // Schedule an initial randomized local target offset so the first episode
        // begins with a randomized target local to this agent.
        {
            // Place on the outer radius (on circumference)
            float r = targetRandomRadius;
            float theta = Random.Range(0f, Mathf.PI * 2f);
            // Randomize around configured `targetStartPosition` (local coordinates)
            pendingRandomLocalOffset = targetStartPosition + new Vector3(Mathf.Cos(theta) * r, 0f, Mathf.Sin(theta) * r);
            pendingRandomTarget = true;
            // Apply immediately so the editor/play starts with a random target position
            ResetTarget();
        }

        Debug.Log("ArmAgent initialized with custom rotation physics.");
    }

    // ============================================================================
    // FIXED UPDATE — CUSTOM PHYSICS TICK
    // ============================================================================

    private void FixedUpdate()
    {
        // Keep mallet rigidbody fixed to last joint's transform (elbow)
        if (joints != null && joints.Length > 0 && malletTransform != null)
        {
            var endJoint = joints[joints.Length - 1];
            // Compute mallet world pose from stored local offset
            Vector3 malletWorldPos = endJoint.jointTransform.TransformPoint(malletLocalPosition);
            Quaternion worldRot = endJoint.jointTransform.rotation * malletLocalRotation;

            // If mallet would be below the floor, try to raise it by adjusting
            // the pitch joints (shoulder pitch and elbow) so the arm bends upward
            // instead of translating the entire agent.
            if (malletWorldPos.y < floorY)
            {
                TryRaiseMalletByAdjustingPitchJoints(floorY);
                malletWorldPos = endJoint.jointTransform.TransformPoint(malletLocalPosition);
                worldRot = endJoint.jointTransform.rotation * malletLocalRotation;
            }

            malletTransform.position = malletWorldPos;
            malletTransform.rotation = worldRot;
            if (malletRigidbody != null)
            {
                malletRigidbody.MovePosition(malletWorldPos);
                malletRigidbody.MoveRotation(worldRot);
            }
        }

        // Compute mallet velocity from position delta (for observations/reward)
        Vector3 currentPos = malletTransform.position;
        computedMalletVelocity = (currentPos - lastMalletPosition) / Time.fixedDeltaTime;
        lastMalletPosition = currentPos;
    }

    // ============================================================================
    // OBSERVATIONS
    // ============================================================================

    public override void CollectObservations(VectorSensor sensor)
    {
        // Joint angles normalized to [-1, 1] based on limits
        for (int i = 0; i < 3; i++)
        {
            var j = joints[i];
            float range = j.maxAngle - j.minAngle;
            float mid = (j.maxAngle + j.minAngle) * 0.5f;
            float normalized = (range > 0f) ? (j.currentAngle - mid) / (range * 0.5f) : 0f;
            sensor.AddObservation(SafeFloat(Mathf.Clamp(normalized, -1f, 1f)));
        }

        // Joint velocities normalized by max speed
        for (int i = 0; i < 3; i++)
        {
            var j = joints[i];
            float normalized = (j.maxSpeed > 0f) ? j.currentVelocity / j.maxSpeed : 0f;
            sensor.AddObservation(SafeFloat(Mathf.Clamp(normalized, -1f, 1f)));
        }

        // Mallet world position (normalized loosely)
        Vector3 malletPos = SafeVector3(malletTransform.position);
        sensor.AddObservation(SafeFloat(malletPos.x) / 2f);
        sensor.AddObservation(SafeFloat(malletPos.y) / 2f);
        sensor.AddObservation(SafeFloat(malletPos.z) / 2f);

        // Mallet velocity
        Vector3 vel = SafeVector3(computedMalletVelocity);
        sensor.AddObservation(Mathf.Clamp(SafeFloat(vel.x) / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(vel.y) / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(vel.z) / 10f, -1f, 1f));

        // Target relative position + distance
        Vector3 relPos = SafeVector3(targetTransform.position - malletPos);
        sensor.AddObservation(Mathf.Clamp(SafeFloat(relPos.x) / 2f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(relPos.y) / 2f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(relPos.z) / 2f, -1f, 1f));
        float dist = SafeFloat(relPos.magnitude);
        sensor.AddObservation(Mathf.Clamp(dist / 3f, 0f, 1f));
    }

    // ============================================================================
    // ACTIONS
    // ============================================================================

    public override void OnActionReceived(ActionBuffers actions)
    {
        float dt = Time.fixedDeltaTime;

        for (int i = 0; i < 3; i++)
        {
            var j = joints[i];
            float action = actions.ContinuousActions[i]; // [-1, 1]

            // Target velocity from action
            float targetVel = action * j.maxSpeed;

            // Accelerate toward target velocity
            float velDiff = targetVel - j.currentVelocity;
            float maxDelta = j.acceleration * dt;
            float deltaVel = Mathf.Clamp(velDiff, -maxDelta, maxDelta);
            j.currentVelocity += deltaVel;

            // Clamp velocity to max speed
            j.currentVelocity = Mathf.Clamp(j.currentVelocity, -j.maxSpeed, j.maxSpeed);

            // Integrate angle
            float newAngle = j.currentAngle + j.currentVelocity * dt;

            // Clamp to limits and stop velocity if hitting limit
            if (newAngle < j.minAngle)
            {
                newAngle = j.minAngle;
                j.currentVelocity = 0f;
            }
            else if (newAngle > j.maxAngle)
            {
                newAngle = j.maxAngle;
                j.currentVelocity = 0f;
            }

            j.currentAngle = newAngle;

            // Apply rotation to transform
            ApplyJointRotation(j);
        }

        // Rewards
        AddReward(-rewardPenaltyPerStep);

        Vector3 relPos = targetTransform.position - malletTransform.position;
        float distToTarget = relPos.magnitude;
        AddReward(-rewardShapingDistance * Mathf.Clamp01(distToTarget / 3f));

        stepsSinceLastReset++;
        if (stepsSinceLastReset >= maxStepsPerEpisode)
        {
            EndEpisode();
        }
    }

    /// <summary>
    /// Iteratively nudges the pitch joints (indices 1 and 2) to raise the mallet
    /// until its world Y is >= floorY or no further improvement is possible.
    /// Uses small test steps and respects joint limits.
    /// </summary>
    private void TryRaiseMalletByAdjustingPitchJoints(float floorYTarget)
    {
        if (joints == null || joints.Length < 3) return;
        var endJoint = joints[joints.Length - 1];

        const int maxIters = 20;
        const float testStep = 1f; // degrees

        for (int iter = 0; iter < maxIters; iter++)
        {
            Vector3 worldPos = endJoint.jointTransform.TransformPoint(malletLocalPosition);
            if (worldPos.y >= floorYTarget) break;

            bool anyImproved = false;

            // Only adjust the pitch joints (shoulder pitch index 1 and elbow index 2)
            for (int jointIdx = 1; jointIdx <= 2; jointIdx++)
            {
                var j = joints[jointIdx];
                float original = j.currentAngle;

                float bestAngle = original;
                float bestY = worldPos.y;

                // Test both directions to see which increases the mallet Y
                for (int sign = -1; sign <= 1; sign += 2)
                {
                    float candidate = Mathf.Clamp(original + sign * testStep, j.minAngle, j.maxAngle);
                    j.currentAngle = candidate;
                    ApplyJointRotation(j);
                    Vector3 candidatePos = endJoint.jointTransform.TransformPoint(malletLocalPosition);
                    if (candidatePos.y > bestY + 1e-4f)
                    {
                        bestY = candidatePos.y;
                        bestAngle = candidate;
                    }
                }

                // Apply the best found angle for this joint
                if (Mathf.Abs(bestAngle - original) > 1e-4f)
                {
                    j.currentAngle = bestAngle;
                    ApplyJointRotation(j);
                    anyImproved = true;
                }
                else
                {
                    // restore original (no improvement)
                    j.currentAngle = original;
                    ApplyJointRotation(j);
                }

                // update worldPos for next joint's tests
                worldPos = endJoint.jointTransform.TransformPoint(malletLocalPosition);
                if (worldPos.y >= floorYTarget) break;
            }

            if (!anyImproved) break;
        }
    }

    private void ApplyJointRotation(JointConfig j)
    {
        // Rotate around local axis by currentAngle (absolute, not delta)
        // We reconstruct the local rotation based on axis and angle.
        Quaternion rot = Quaternion.AngleAxis(j.currentAngle, j.rotationAxis);
        j.jointTransform.localRotation = rot;
    }

    // ============================================================================
    // COLLISION (strike detection on mallet)
    // ============================================================================

    private void OnCollisionEnter(Collision collision)
    {
        // Only process if the collision involves the mallet
        if (collision.gameObject == targetTransform.gameObject)
        {
            ValidateAndRewardStrike(collision);
        }
    }

    public void ValidateAndRewardStrike(Collision collision)
    {
        if (strikeValidatedThisFrame) return;
        strikeValidatedThisFrame = true;

        Vector3 malletVel = computedMalletVelocity;
        ContactPoint contact = collision.contacts[0];
        Vector3 normal = contact.normal;

        float impactVel = malletVel.magnitude;
        // Approach velocity along the surface normal (positive when moving into the target)
        float approachVel = Vector3.Dot(malletVel, -normal);

        // Ensure motion is generally downward (y component negative), and approach is along the contact normal
        bool isMovingDownward = malletVel.y < -0.1f;
        float normalUp = Vector3.Dot(normal, Vector3.up);

        bool valid = (approachVel > minImpactVelocity) && isMovingDownward && (normalUp > strikeNormalThreshold);

        if (valid)
        {
            AddReward(rewardValidStrike);
            float velBonus = Mathf.Clamp(impactVel, 0f, rewardMaxVelocityBonus);
            AddReward(rewardVelocityMultiplier * velBonus / rewardMaxVelocityBonus);
            lastImpactVelocity = impactVel;

            Debug.Log($"[ArmAgent] VALID STRIKE! Velocity={impactVel:F2} m/s, Approach={approachVel:F2}, NormalUp={normalUp:F2}");

            // Schedule random target for next episode (local to agent)
            Transform root = this.transform;
            // Place on the outer radius (on circumference)
            float r = targetRandomRadius;
            float theta = Random.Range(0f, Mathf.PI * 2f);
            Vector3 localOffset = new Vector3(Mathf.Cos(theta) * r, 0f, Mathf.Sin(theta) * r);
            // Randomize around configured `targetStartPosition` (local coordinates)
            pendingRandomLocalOffset = targetStartPosition + new Vector3(localOffset.x, 0f, localOffset.z);
            pendingRandomTarget = true;
            Vector3 worldPos = root.TransformPoint(pendingRandomLocalOffset);
            Debug.Log($"[ArmAgent] Target hit — next target at {worldPos} (local to agent). Ending episode.");
            EndEpisode();
        }
        else
        {
            Debug.Log($"[ArmAgent] Invalid strike. Vel={impactVel:F2}, Approach={approachVel:F2}, NormalUp={normalUp:F2}");
        }
    }

    // ============================================================================
    // EPISODE MANAGEMENT
    // ============================================================================

    public override void OnEpisodeBegin()
    {
        ResetArm();

        // Ensure each episode starts with a randomized target local to this agent
        if (!pendingRandomTarget)
        {
            Transform root = this.transform;
            // Place on the outer radius (on circumference)
            float r = targetRandomRadius;
            float theta = Random.Range(0f, Mathf.PI * 2f);
            Vector3 localOffset = new Vector3(Mathf.Cos(theta) * r, 0f, Mathf.Sin(theta) * r);
            // Randomize around configured `targetStartPosition` (local coordinates)
            pendingRandomLocalOffset = targetStartPosition + localOffset;
            pendingRandomTarget = true;
            Vector3 worldPos = root.TransformPoint(pendingRandomLocalOffset);
            Debug.Log($"[ArmAgent] Scheduled randomized start target at {worldPos}");
        }

        ResetTarget();

        stepsSinceLastReset = 0;
        strikeValidatedThisFrame = false;
        lastImpactVelocity = 0f;

        Debug.Log("[ArmAgent] Episode started.");
    }

    private void ResetArm()
    {
        for (int i = 0; i < joints.Length; i++)
        {
            var j = joints[i];
            j.currentAngle = j.startAngle;
            j.currentVelocity = 0f;
            ApplyJointRotation(j);
        }

        // Reset mallet rigidbody if present (for collision detection)
        if (malletRigidbody != null)
        {
            // Place mallet at its correct relative pose to the end joint and clear velocities
            if (joints != null && joints.Length > 0)
            {
                var endJoint = joints[joints.Length - 1];
                Vector3 worldPos = endJoint.jointTransform.TransformPoint(malletLocalPosition);
                Quaternion worldRot = endJoint.jointTransform.rotation * malletLocalRotation;
                // If mallet would be below the floor, adjust pitch joints so the
                // arm bends upward and the mallet remains attached.
                if (worldPos.y < floorY)
                {
                    TryRaiseMalletByAdjustingPitchJoints(floorY);
                    worldPos = endJoint.jointTransform.TransformPoint(malletLocalPosition);
                    worldRot = endJoint.jointTransform.rotation * malletLocalRotation;
                }
                malletTransform.position = worldPos;
                malletTransform.rotation = worldRot;
                malletRigidbody.position = worldPos;
                malletRigidbody.rotation = worldRot;
            }
            malletRigidbody.linearVelocity = Vector3.zero;
            malletRigidbody.angularVelocity = Vector3.zero;
        }

        lastMalletPosition = malletTransform.position;
        computedMalletVelocity = Vector3.zero;

        Debug.Log("[ArmAgent] Arm reset to starting angles.");
    }

    private void ResetTarget()
    {
        if (pendingRandomTarget)
        {
            // Apply local offset if target is parented to this agent; otherwise compute world position.
            if (targetTransform.IsChildOf(this.transform))
            {
                targetTransform.localPosition = pendingRandomLocalOffset;
            }
            else
            {
                targetTransform.position = this.transform.TransformPoint(pendingRandomLocalOffset);
            }
            pendingRandomTarget = false;
            Debug.Log($"[ArmAgent] Target randomized to {targetTransform.position}");
        }
        else
        {
            // Place target relative to this agent's transform
            if (targetTransform.IsChildOf(this.transform))
            {
                targetTransform.localPosition = targetStartPosition;
            }
            else
            {
                targetTransform.position = this.transform.TransformPoint(new Vector3(targetStartPosition.x, targetStartPosition.y, targetStartPosition.z));
            }
            Debug.Log($"[ArmAgent] Target reset to {targetTransform.position} (local to agent)");
        }

        Rigidbody targetRb = targetTransform.GetComponent<Rigidbody>();
        if (targetRb != null)
        {
            targetRb.linearVelocity = Vector3.zero;
            targetRb.angularVelocity = Vector3.zero;
        }
    }

    // ============================================================================
    // UTILITY
    // ============================================================================

    public void BombHit()
    {
        AddReward(-penaltyBombHit);
        EndEpisode();
        Debug.Log("[ArmAgent] BOMB HIT! Episode ended.");
    }

    private void LateUpdate()
    {
        strikeValidatedThisFrame = false;
    }

    [ContextMenu("Print Observations")]
    public void PrintObservations()
    {
        Debug.Log("=== ArmAgent Observations ===");
        for (int i = 0; i < joints.Length; i++)
        {
            var j = joints[i];
            Debug.Log($"Joint {i}: Angle={j.currentAngle:F1}° Vel={j.currentVelocity:F1}°/s  Limits=[{j.minAngle},{j.maxAngle}]");
        }
        Debug.Log($"Mallet Pos: {malletTransform.position}, Vel: {computedMalletVelocity}");
        Debug.Log($"Target Pos: {targetTransform.position}, Dist: {(targetTransform.position - malletTransform.position).magnitude:F2}m");
        Debug.Log($"Steps: {stepsSinceLastReset}/{maxStepsPerEpisode}");
    }

    [ContextMenu("Test Strike")]
    public void TestStrike()
    {
        // Simulate a downward strike for testing collision
        computedMalletVelocity = new Vector3(0, -5f, 0);
        Debug.Log("[ArmAgent] Test strike velocity applied.");
    }
}
