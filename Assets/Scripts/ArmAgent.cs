using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;

// Minimal Agent setup for Phase 1 (reach target). Fill in joints and board later.
public class ArmAgent : Agent
{
    [Header("Robot Joints")]
    [SerializeField] private ConfigurableJoint baseYaw;
    [SerializeField] private ConfigurableJoint shoulderPitch;
    [SerializeField] private ConfigurableJoint elbowPitch;

    [Header("End Effector")]
    [SerializeField] private Rigidbody malletRb;

    [Header("Training")]
    [SerializeField] private Transform target; // Phase 1 target sphere
    [SerializeField] private bool enableDebugLogs = true;

    [Header("Joint Drive Settings")]
    [SerializeField] private float driveSpring = 2000f;
    [SerializeField] private float driveDamper = 150f;
    [SerializeField] private float driveMaxForce = 10000f;

    // Called at the beginning of each episode
    public override void OnEpisodeBegin()
    {
        // TODO: Reset arm pose and target position for curriculum phases
        // For now, keep current scene setup
    }

    // Collect vector observations for the policy
    public override void CollectObservations(VectorSensor sensor)
    {
        // Joint angles (approx from joint target as placeholder)
        AddJointObservation(sensor, baseYaw);
        AddJointObservation(sensor, shoulderPitch);
        AddJointObservation(sensor, elbowPitch);

        // End effector position and velocity
        if (malletRb != null)
        {
            sensor.AddObservation(transform.InverseTransformPoint(malletRb.position));
            sensor.AddObservation(transform.InverseTransformDirection(malletRb.linearVelocity));
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
            sensor.AddObservation(Vector3.zero);
        }

        // Target relative position (Phase 1)
        if (target != null)
        {
            Vector3 rel = transform.InverseTransformPoint(target.position);
            sensor.AddObservation(rel);
        }
        else
        {
            sensor.AddObservation(Vector3.zero);
        }
    }

    // Map actions to joint target adjustments (placeholder)
    public override void OnActionReceived(ActionBuffers actions)
    {
        if (enableDebugLogs)
        {
            OnActionReceived_Debug(actions);
        }
        ActionSegment<float> a = actions.ContinuousActions;
        // Expect 3 actions: delta targets for base/shoulder/elbow
        ApplyDeltaToJoint(baseYaw, a.Length > 0 ? a[0] : 0f);
        ApplyDeltaToJoint(shoulderPitch, a.Length > 1 ? a[1] : 0f);
        ApplyDeltaToJoint(elbowPitch, a.Length > 2 ? a[2] : 0f);

        // Simple shaping for Phase 1: negative distance to target
        if (target != null && malletRb != null)
        {
            float dist = Vector3.Distance(malletRb.position, target.position);
            AddReward(-dist * 0.001f);
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        // For manual testing: map keys to small deltas
        var ca = actionsOut.ContinuousActions;
        ca[0] = Input.GetAxis("Horizontal");
        ca[1] = Input.GetAxis("Vertical");
        ca[2] = Input.GetKey(KeyCode.Q) ? 1f : (Input.GetKey(KeyCode.E) ? -1f : 0f);
    }

    private void AddJointObservation(VectorSensor sensor, ConfigurableJoint joint)
    {
        if (joint == null)
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
            return;
        }
        // Determine which angular axis is free on this joint and observe that angle
        Vector3 euler = joint.targetRotation.eulerAngles;
        float angleDeg = 0f;
        if (joint.angularXMotion != ConfigurableJointMotion.Locked)
        {
            angleDeg = euler.x;
        }
        else if (joint.angularYMotion != ConfigurableJointMotion.Locked)
        {
            angleDeg = euler.y;
        }
        else if (joint.angularZMotion != ConfigurableJointMotion.Locked)
        {
            angleDeg = euler.z;
        }
        sensor.AddObservation(NormalizeAngle(angleDeg));
        sensor.AddObservation(0f); // angular velocity placeholder
    }

    private void ApplyDeltaToJoint(ConfigurableJoint joint, float delta)
    {
        if (joint == null) return;
        // Map [-1,1] delta to small radians and adjust targetRotation
        const float scale = 5f; // degrees per step placeholder
        var tr = joint.targetRotation;
        var e = tr.eulerAngles;
        // Apply delta on the joint's free angular axis
        if (joint.angularXMotion != ConfigurableJointMotion.Locked)
        {
            e.x += delta * scale;
        }
        else if (joint.angularYMotion != ConfigurableJointMotion.Locked)
        {
            e.y += delta * scale;
        }
        else if (joint.angularZMotion != ConfigurableJointMotion.Locked)
        {
            e.z += delta * scale;
        }
        joint.targetRotation = Quaternion.Euler(e);
    }

    private float NormalizeAngle(float degrees)
    {
        float a = Mathf.DeltaAngle(0f, degrees) / 180f; // [-1,1]
        return a;
    }

    private void FixedUpdate()
    {
        // Ensure decisions are requested each physics step when training
        RequestDecision();
    }

    // Debug: log when training starts
    private void Start()
    {
        Debug.Log($"[ArmAgent] Started. Target={target != null}. MalletRb={malletRb != null}. " +
                  $"BaseYaw={baseYaw != null}. ShoulderPitch={shoulderPitch != null}. ElbowPitch={elbowPitch != null}");

        // Configure joints at runtime to ensure drives are non-zero for physics control
        ConfigureJoint(baseYaw);
        ConfigureJoint(shoulderPitch);
        ConfigureJoint(elbowPitch);
    }

    // Debug: log when actions are received
    public void OnActionReceived_Debug(ActionBuffers actions)
    {
        ActionSegment<float> a = actions.ContinuousActions;
        Debug.Log($"[ArmAgent] Action received: {a[0]:F2}, {a[1]:F2}, {a[2]:F2}");
    }

    private void ConfigureJoint(ConfigurableJoint joint)
    {
        if (joint == null)
        {
            if (enableDebugLogs) Debug.LogWarning("[ArmAgent] ConfigureJoint called with null joint");
            return;
        }

        // Ensure the joint will move towards targetRotation using slerp/angular drives
        try
        {
            joint.rotationDriveMode = RotationDriveMode.Slerp;
            JointDrive jd = new JointDrive
            {
                positionSpring = driveSpring,
                positionDamper = driveDamper,
                maximumForce = driveMaxForce
            };
            joint.slerpDrive = jd;
            joint.angularXDrive = jd;
            joint.angularYZDrive = jd;

            joint.projectionMode = JointProjectionMode.PositionAndRotation;
            joint.projectionDistance = 0.1f;
            joint.projectionAngle = 5f;

            if (enableDebugLogs)
            {
                Debug.Log($"[ArmAgent] Configured joint '{joint.gameObject.name}' drives: spring={driveSpring}, damper={driveDamper}, maxForce={driveMaxForce}");
                if (joint.connectedBody == null)
                {
                    Debug.LogWarning($"[ArmAgent] Joint '{joint.gameObject.name}' has no connectedBody set.");
                }
            }
        }
        catch (System.Exception ex)
        {
            Debug.LogWarning($"[ArmAgent] Failed to configure joint '{joint.gameObject.name}': {ex.Message}");
        }
    }
}
