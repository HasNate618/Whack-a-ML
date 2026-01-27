using UnityEngine;
using System.Collections;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

/// <summary>
/// ArmAgent controls a 3-DOF robotic arm (shoulder yaw, shoulder pitch, elbow pitch)
/// to strike a target "mole" using a downward-mounted mallet.
/// 
/// The agent learns via PPO to:
/// 1. Position and orient the arm toward the target
/// 2. Execute downward strikes with sufficient velocity
/// 3. Maximize impact velocity for bonus rewards
/// 4. Avoid self-collisions and stay within joint limits
/// 
/// Observations: Joint angles (3), angular velocities (3), mallet position (3), 
///               mallet velocity (3), target relative position (3), 
///               target distance (1), active flag (1) = 18 total
/// 
/// Actions: Continuous delta rotations for 3 joints (mapped from [-1, 1])
/// 
/// Rewards:
/// - +1.0 for valid mole strike (downward velocity, upward normal)
/// - +0.2 × impactVelocity bonus (clamped 0–5 m/s)
/// - -0.001 per timestep (time pressure)
/// - Optional: -0.01 × distance to target (shaping)
/// </summary>
public class ArmAgent : Agent
{
    // ============================================================================
    // JOINT REFERENCES
    // ============================================================================
    
    [Header("Joint References")]
    [SerializeField] private Rigidbody shoulderYawRigidbody;
    [SerializeField] private Rigidbody shoulderPitchRigidbody;
    [SerializeField] private Rigidbody elbowPitchRigidbody;
    
    [SerializeField] private Rigidbody malletRigidbody;
    [SerializeField] private Transform targetTransform;
    
    // ============================================================================
    // JOINT CONTROL PARAMETERS
    // ============================================================================
    
    [Header("Joint Control")]
    [SerializeField] private float maxAngularVelocity = 180f; // degrees/sec
    [SerializeField] private float actionScale = 30f; // scales action [-1,1] to delta velocity
    
    [Header("Joint Motor Settings")]
    [Tooltip("Maximum motor force applied by hinge motors")]
    [SerializeField] private float motorForce = 200f;
    [Tooltip("Whether hinge motors should free spin when target reached")]
    [SerializeField] private bool motorFreeSpin = false;
    
    // ============================================================================
    // STRIKE VALIDATION PARAMETERS
    // ============================================================================
    
    [Header("Strike Validation")]
    [SerializeField] private float minImpactVelocity = 2f; // m/s
    [SerializeField] private float strikeNormalThreshold = 0.5f; // dot(normal, up) > this
    [SerializeField] private float strikeVelocityDownwardThreshold = 0.5f; // dot(velocity, down) > this
    
    // ============================================================================
    // REWARD PARAMETERS
    // ============================================================================
    
    [Header("Rewards")]
    [SerializeField] private float rewardValidStrike = 1f;
    [SerializeField] private float rewardVelocityMultiplier = 0.2f;
    [SerializeField] private float rewardMaxVelocityBonus = 5f; // clamp velocity for bonus
    [SerializeField] private float rewardPenaltyPerStep = 0.001f;
    [SerializeField] private float rewardShapingDistance = 0.01f; // distance shaping strength
    [SerializeField] private float penaltyBombHit = 2f;
    
    // ============================================================================
    // EPISODE CONFIGURATION
    // ============================================================================
    
    [Header("Episode Settings")]
    [SerializeField] private int maxStepsPerEpisode = 500;
    [SerializeField] private Vector3 startingArmRotation = Vector3.zero;
    [SerializeField] private Vector3 targetStartPosition = new Vector3(0, 1f, 1f);
    [SerializeField] private float targetResetHeight = 0.5f;
    [SerializeField] private float targetRandomRadius = 1.0f;
    
    // ============================================================================
    // RUNTIME STATE
    // ============================================================================
    
    private Rigidbody[] jointRigidbodies;
    private float[] currentTargetAngularVelocities;
    
    private int stepsSinceLastReset = 0;
    private bool strikeValidatedThisFrame = false;
    private float lastImpactVelocity = 0f;
    // Pending randomized target position to apply on next episode begin
    private bool pendingRandomTarget = false;
    private Vector3 pendingRandomTargetPos = Vector3.zero;

    // Small helpers to sanitize observations to avoid NaN/Infinity being passed to ML-Agents
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
    
    // ============================================================================
    // INITIALIZATION
    // ============================================================================
    
    public override void Initialize()
    {
        // Cache rigidbody references
        jointRigidbodies = new Rigidbody[]
        {
            shoulderYawRigidbody,
            shoulderPitchRigidbody,
            elbowPitchRigidbody
        };
        
        currentTargetAngularVelocities = new float[3];
        
        // Validate scene setup
        if (shoulderYawRigidbody == null || shoulderPitchRigidbody == null || 
            elbowPitchRigidbody == null || malletRigidbody == null || 
            targetTransform == null)
        {
            Debug.LogError("ArmAgent: Missing rigidbody or target references! " +
                "Ensure shoulder yaw, shoulder pitch, elbow pitch, mallet, and target are assigned in inspector.");
            enabled = false;
            return;
        }
        
        // Verify HingeJoints are properly configured
        VerifyJointConfiguration();
        
        Debug.Log("ArmAgent initialized successfully. " +
            $"Joints: {jointRigidbodies.Length}, Mallet: {malletRigidbody.name}, Target: {targetTransform.name}");
    }
    
    private void VerifyJointConfiguration()
    {
        foreach (var joint in jointRigidbodies)
        {
            if (joint.GetComponent<HingeJoint>() == null)
            {
                Debug.LogWarning($"ArmAgent: Joint {joint.name} does not have a HingeJoint! " +
                    "Ensure all joint Rigidbodies have HingeJoint components.");
            }
        }
    }
    
    // ============================================================================
    // OBSERVATION COLLECTION (State)
    // ============================================================================
    
    public override void CollectObservations(VectorSensor sensor)
    {
        // Joint angles (3) - normalized to [-1, 1]
        for (int i = 0; i < 3; i++)
        {
            var hinge = jointRigidbodies[i].GetComponent<HingeJoint>();
            float angleDeg = 0f;
            if (hinge != null)
            {
                angleDeg = hinge.angle; // degrees
            }
            else
            {
                angleDeg = jointRigidbodies[i].transform.localEulerAngles.x;
                angleDeg = angleDeg > 180 ? angleDeg - 360 : angleDeg;
            }
            sensor.AddObservation(SafeFloat(angleDeg) / 180f);
        }

        // Joint angular velocities (3) - degrees/sec normalized by maxAngularVelocity
        for (int i = 0; i < 3; i++)
        {
            var hinge = jointRigidbodies[i].GetComponent<HingeJoint>();
            float velDeg = 0f;
            if (hinge != null)
            {
                // HingeJoint does not reliably expose an angular velocity property across
                // Unity versions. Use the joint's Rigidbody angular velocity as a proxy.
                velDeg = jointRigidbodies[i].angularVelocity.x * Mathf.Rad2Deg;
            }
            else
            {
                velDeg = jointRigidbodies[i].angularVelocity.x * Mathf.Rad2Deg;
            }
            sensor.AddObservation(SafeFloat(velDeg) / maxAngularVelocity);
        }

        // Mallet position (3)
        Vector3 malletPos = SafeVector3(malletRigidbody.position);
        sensor.AddObservation(SafeFloat(malletPos.x) / 2f);
        sensor.AddObservation(SafeFloat(malletPos.y) / 2f);
        sensor.AddObservation(SafeFloat(malletPos.z) / 2f);

        // Mallet velocity (3)
        Vector3 malletVel = SafeVector3(malletRigidbody.linearVelocity);
        sensor.AddObservation(Mathf.Clamp(SafeFloat(malletVel.x) / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(malletVel.y) / 10f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(malletVel.z) / 10f, -1f, 1f));

        // Target relative position (3) + distance (1)
        Vector3 targetRelativePos = SafeVector3(targetTransform.position - malletPos);
        sensor.AddObservation(Mathf.Clamp(SafeFloat(targetRelativePos.x) / 2f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(targetRelativePos.y) / 2f, -1f, 1f));
        sensor.AddObservation(Mathf.Clamp(SafeFloat(targetRelativePos.z) / 2f, -1f, 1f));
        float distanceToTarget = SafeFloat(targetRelativePos.magnitude);
        sensor.AddObservation(Mathf.Clamp(distanceToTarget / 3f, 0f, 1f));
    }
    
    // ============================================================================
    // ACTION HANDLING (Control)
    // ============================================================================
    
    public override void OnActionReceived(ActionBuffers actions)
    {
        float[] continuousActions = actions.ContinuousActions.Array;
        
        if (continuousActions.Length != 3)
        {
            Debug.LogError($"ArmAgent: Expected 3 continuous actions, got {continuousActions.Length}");
            return;
        }
        
        // Map continuous actions [-1, 1] to target angular velocities (deg/sec)
        for (int i = 0; i < 3; i++)
        {
            float action = continuousActions[i]; // [-1, 1]
            float deltaVelocity = action * actionScale; // degrees/sec

            var hinge = jointRigidbodies[i].GetComponent<HingeJoint>();
            float currentVelDeg = 0f;
            if (hinge != null)
            {
                // Use Rigidbody angular velocity (x) as a proxy for hinge angular speed
                currentVelDeg = jointRigidbodies[i].angularVelocity.x * Mathf.Rad2Deg; // deg/sec
            }
            else
            {
                currentVelDeg = jointRigidbodies[i].angularVelocity.x * Mathf.Rad2Deg;
            }

            float targetVelocityDeg = Mathf.Clamp(
                currentVelDeg + deltaVelocity,
                -maxAngularVelocity,
                maxAngularVelocity
            );

            currentTargetAngularVelocities[i] = targetVelocityDeg;
        }

        ApplyJointControl();
        
        // Per-step penalty (time pressure)
        AddReward(-rewardPenaltyPerStep);
        
        // Distance shaping (weak encouragement toward target)
        Vector3 targetRelativePos = targetTransform.position - malletRigidbody.position;
        float distanceToTarget = targetRelativePos.magnitude;
        AddReward(-rewardShapingDistance * (distanceToTarget / 3f)); // Normalize to [0, 1]
        
        stepsSinceLastReset++;
        
        // Check episode termination
        if (stepsSinceLastReset >= maxStepsPerEpisode)
        {
            EndEpisode();
        }
    }
    
    private void ApplyJointControl()
    {
        // Use HingeJoint motors when available; otherwise fallback to directly setting angular velocity.
        for (int i = 0; i < 3; i++)
        {
            var hinge = jointRigidbodies[i].GetComponent<HingeJoint>();
            float targetVelDeg = currentTargetAngularVelocities[i];

            if (hinge != null)
            {
                JointMotor motor = hinge.motor;
                motor.targetVelocity = targetVelDeg; // degrees/sec
                motor.force = motorForce;
                motor.freeSpin = motorFreeSpin;
                hinge.motor = motor;
                hinge.useMotor = true;
            }
            else
            {
                // Fallback: set rigidbody angular velocity (rad/s)
                float targetVelRad = targetVelDeg * Mathf.Deg2Rad;
                jointRigidbodies[i].angularVelocity = new Vector3(
                    targetVelRad,
                    jointRigidbodies[i].angularVelocity.y,
                    jointRigidbodies[i].angularVelocity.z
                );
            }
        }
    }
    
    // ============================================================================
    // COLLISION & STRIKE DETECTION
    // ============================================================================
    
    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject == targetTransform.gameObject)
        {
            ValidateAndRewardStrike(collision);
        }
    }
    
    private void ValidateAndRewardStrike(Collision collision)
    {
        // Ensure we only reward once per strike
        if (strikeValidatedThisFrame)
            return;
        strikeValidatedThisFrame = true;
        
        // Get collision details
        Vector3 malletVelocity = malletRigidbody.linearVelocity;
        ContactPoint contactPoint = collision.contacts[0];
        Vector3 collisionNormal = contactPoint.normal;
        
        // Strike validation checks:
        // 1. Mallet velocity is primarily downward
        float downwardComponent = Vector3.Dot(malletVelocity, Vector3.down);
        float downwardRatio = downwardComponent / (malletVelocity.magnitude + 0.01f); // avoid divide by zero
        
        // 2. Collision normal points upward (hit from above)
        float upwardComponent = Vector3.Dot(collisionNormal, Vector3.up);
        
        // 3. Impact velocity exceeds minimum threshold
        float impactVelocity = malletVelocity.magnitude;
        
        bool isValidStrike = (downwardRatio > strikeVelocityDownwardThreshold) &&
                             (upwardComponent > strikeNormalThreshold) &&
                             (impactVelocity > minImpactVelocity);
        
        if (isValidStrike)
        {
            // Award base strike reward
            AddReward(rewardValidStrike);
            
            // Award velocity bonus (clamped to avoid extreme rewards)
            float velocityBonus = Mathf.Clamp(impactVelocity, 0f, rewardMaxVelocityBonus);
            AddReward(rewardVelocityMultiplier * velocityBonus / rewardMaxVelocityBonus);
            
            lastImpactVelocity = impactVelocity;
            
            Debug.Log($"VALID STRIKE! Velocity: {impactVelocity:F2} m/s, Downward ratio: {downwardRatio:F2}, Normal up: {upwardComponent:F2}");
            // Schedule a randomized target relocation (XZ plane) relative to the arm base
            Vector3 center = shoulderYawRigidbody != null ? shoulderYawRigidbody.position : malletRigidbody.position;
            float r = Random.Range(0f, targetRandomRadius);
            float theta = Random.Range(0f, Mathf.PI * 2f);
            Vector3 offset = new Vector3(Mathf.Cos(theta) * r, 0f, Mathf.Sin(theta) * r);
            pendingRandomTargetPos = new Vector3(center.x + offset.x, targetResetHeight, center.z + offset.z);
            pendingRandomTarget = true;
            Debug.Log($"Target hit — scheduling random target at {pendingRandomTargetPos} and ending episode.");

            EndEpisode();
        }
        else
        {
            // Soft hit or incorrect angle - no reward
            Debug.Log($"Invalid strike. Velocity: {impactVelocity:F2} m/s (min: {minImpactVelocity}), " +
                $"Downward: {downwardRatio:F2} (min: {strikeVelocityDownwardThreshold}), " +
                $"Normal up: {upwardComponent:F2} (min: {strikeNormalThreshold})");
        }
    }
    
    // ============================================================================
    // EPISODE MANAGEMENT
    // ============================================================================
    
    public override void OnEpisodeBegin()
    {
        // Reset arm to starting pose
        ResetArmPose();
        
        // Reset target position
        ResetTargetPosition();
        
        // Reset episode state
        stepsSinceLastReset = 0;
        strikeValidatedThisFrame = false;
        lastImpactVelocity = 0f;
        
        Debug.Log("Episode started");
    }
    
    private void ResetArmPose()
    {
        // Reset each joint to starting rotation
        // In a physics-based system, we zero velocities and apply small resets
        // Make involved rigidbodies kinematic, snap to starting pose, zero velocities,
        // then re-enable physics on the next FixedUpdate to avoid solver impulses.
        foreach (var joint in jointRigidbodies)
        {
            // Make kinematic for a safe transform teleport
            joint.isKinematic = true;
            joint.linearVelocity = Vector3.zero;
            joint.angularVelocity = Vector3.zero;

            var hinge = joint.GetComponent<HingeJoint>();
            if (hinge != null)
            {
                hinge.useMotor = false;
            }

            // Reset rotation to configured starting rotation (local space)
            joint.transform.localEulerAngles = startingArmRotation;
        }

        // Reset mallet
        malletRigidbody.isKinematic = true;
        malletRigidbody.linearVelocity = Vector3.zero;
        malletRigidbody.angularVelocity = Vector3.zero;

        Debug.Log("Arm pose reset to starting position (physics disabled temporarily)");

        // Temporarily make target kinematic while resetting its transform
        Rigidbody targetRb = targetTransform.GetComponent<Rigidbody>();
        if (targetRb != null)
        {
            targetRb.isKinematic = true;
            targetRb.linearVelocity = Vector3.zero;
            targetRb.angularVelocity = Vector3.zero;
        }

        // Re-enable physics after a single FixedUpdate to avoid bounce/solver artifacts
        StartCoroutine(ReenablePhysicsNextFixedUpdate());
    }
    
    private void ResetTargetPosition()
    {
        // Reset target to either the pending randomized position (if scheduled)
        // or the configured start position.
        if (pendingRandomTarget)
        {
            targetTransform.position = pendingRandomTargetPos;
            pendingRandomTarget = false;
            Debug.Log($"Target randomized to {targetTransform.position}");
        }
        else
        {
            // Reset to configured start
            targetTransform.position = targetStartPosition;
            Debug.Log($"Target reset to {targetStartPosition}");
        }

        // Ensure target rigidbody is not moving
        Rigidbody targetRb = targetTransform.GetComponent<Rigidbody>();
        if (targetRb != null)
        {
            targetRb.linearVelocity = Vector3.zero;
            targetRb.angularVelocity = Vector3.zero;
        }
    }
    
    // ============================================================================
    // EPISODE TERMINATION
    // ============================================================================
    
    public void BombHit()
    {
        // Called when agent hits a bomb (for future multi-target support)
        AddReward(-penaltyBombHit);
        EndEpisode();
        Debug.Log("BOMB HIT! Episode ended.");
    }
    
    // ============================================================================
    // FRAME UPDATE LOGIC
    // ============================================================================
    
    private void LateUpdate()
    {
        // Reset strike validation flag each frame
        strikeValidatedThisFrame = false;
    }
    
    // ============================================================================
    // UTILITIES & DEBUGGING
    // ============================================================================
    
    [ContextMenu("Print Current Observations")]
    public void PrintObservations()
    {
        Debug.Log("=== Current Observations ===");
        for (int i = 0; i < 3; i++)
        {
            float angle = jointRigidbodies[i].transform.localEulerAngles.x;
            angle = angle > 180 ? angle - 360 : angle;
            Debug.Log($"Joint {i} - Angle: {angle:F1}°, AngVel: {jointRigidbodies[i].angularVelocity.magnitude:F2} rad/s");
        }
        
        Vector3 targetRelPos = targetTransform.position - malletRigidbody.position;
        Debug.Log($"Mallet Pos: {malletRigidbody.position}, Target Pos: {targetTransform.position}");
        Debug.Log($"Distance to target: {targetRelPos.magnitude:F2}m");
        Debug.Log($"Mallet Velocity: {malletRigidbody.linearVelocity.magnitude:F2} m/s (Last impact: {lastImpactVelocity:F2} m/s)");
        Debug.Log($"Steps this episode: {stepsSinceLastReset}/{maxStepsPerEpisode}");
    }
    
    [ContextMenu("Trigger Manual Strike Test")]
    public void TestStrike()
    {
        // Apply downward velocity to mallet for testing strike detection
        malletRigidbody.linearVelocity = new Vector3(0, -5f, 0);
        Debug.Log("Applied downward velocity to mallet for strike test");
    }

    private IEnumerator ReenablePhysicsNextFixedUpdate()
    {
        // Wait a single fixed update to allow physics to settle after teleport
        yield return new WaitForFixedUpdate();

        // Re-enable physics for joints and mallet
        foreach (var joint in jointRigidbodies)
        {
            joint.isKinematic = false;
            var hinge = joint.GetComponent<HingeJoint>();
            if (hinge != null)
            {
                hinge.useMotor = false;
            }
        }

        malletRigidbody.isKinematic = false;

        Rigidbody targetRb = targetTransform.GetComponent<Rigidbody>();
        if (targetRb != null)
        {
            targetRb.isKinematic = false;
        }
    }
}
