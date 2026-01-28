using UnityEngine;

[RequireComponent(typeof(Collider))]
public class TargetHitProxy : MonoBehaviour
{
    private void OnCollisionEnter(Collision collision)
    {
        var agent = GetComponentInParent<ArmAgent>();
        if (agent == null) return;

        // Prefer direct transform match for mallet
        var malletT = agent.GetMalletTransform();
        if (malletT != null && (collision.gameObject == malletT.gameObject || collision.transform.IsChildOf(malletT)))
        {
            agent.ValidateAndRewardStrike(collision);
            return;
        }

        // Fallback: compare rigidbodies
        var malletRb = agent.GetMalletRigidbody();
        if (malletRb != null && collision.rigidbody == malletRb)
        {
            agent.ValidateAndRewardStrike(collision);
        }
    }
}
