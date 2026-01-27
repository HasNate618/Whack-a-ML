using UnityEngine;

// Simple controller to adjust Time.timeScale for faster training runs.
public class TimeScaleController : MonoBehaviour
{
    [Header("Time Scale Settings")]
    [SerializeField] private float timeScale = 10f;

    private float _originalFixedDeltaTime;

    private void Awake()
    {
        _originalFixedDeltaTime = Time.fixedDeltaTime;
    }

    private void Update()
    {
        // Always apply the configured timeScale each frame.
        Time.timeScale = Mathf.Max(0f, timeScale);
        if (Time.timeScale > 0f)
        {
            Time.fixedDeltaTime = _originalFixedDeltaTime / Time.timeScale;
        }
    }
}
