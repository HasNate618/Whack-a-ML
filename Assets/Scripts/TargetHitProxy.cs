using UnityEngine;

[RequireComponent(typeof(Collider))]
public class TargetHitProxy : MonoBehaviour
{
    private void OnCollisionEnter(Collision collision)
    {
        // Prefer to forward the collision to the agent that owns the colliding object
        // (the striker). This avoids the target's owner receiving rewards when other
        // agents hit this target.
        var strikerAgent = collision.collider.GetComponentInParent<ArmAgent>();
        if (strikerAgent != null)
        {
            strikerAgent.ValidateAndRewardStrike(collision);
            return;
        }

        // Fallback: if we couldn't find a striker agent, fall back to the target's
        // owning agent (previous behavior) so collisions aren't lost.
        var ownerAgent = GetComponentInParent<ArmAgent>();
        if (ownerAgent == null) return;

        // Prefer direct transform match for mallet
        var malletT = ownerAgent.GetMalletTransform();
        if (malletT != null && (collision.gameObject == malletT.gameObject || collision.transform.IsChildOf(malletT)))
        {
            ownerAgent.ValidateAndRewardStrike(collision);
            return;
        }

        // Fallback: compare rigidbodies
        var malletRb = ownerAgent.GetMalletRigidbody();
        if (malletRb != null && collision.rigidbody == malletRb)
        {
            ownerAgent.ValidateAndRewardStrike(collision);
        }
    }
}
