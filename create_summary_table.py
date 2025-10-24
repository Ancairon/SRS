import pandas as pd

# Load and create summary
df = pd.read_csv('eval_results/rf_holdout_evaluation_results.csv')

# Calculate derived metrics
df['distance_ratio'] = df['avg_distance_holdout'] / df['avg_distance_known']
df['confidence_diff'] = df['avg_proba_known'] - df['avg_proba_holdout']

# Create summary table
summary = pd.DataFrame({
    'Class': df['excluded_class'],
    'Detection Rate': df['detection_rate'].apply(lambda x: f"{x:.1%}"),
    'FPR': df['known_flagged_rate'].apply(lambda x: f"{x:.1%}"),
    'Distance Ratio': df['distance_ratio'].apply(lambda x: f"{x:.1f}x"),
    'Conf Diff': df['confidence_diff'].apply(lambda x: f"{x:+.3f}"),
    'Test Acc': df['test_accuracy'].apply(lambda x: f"{x:.3f}"),
    'Verdict': df.apply(lambda row: 
        '🟢 Excellent' if row['detection_rate'] > 0.95 and row['known_flagged_rate'] < 0.20
        else '🟡 Good' if row['detection_rate'] > 0.70
        else '🟠 Moderate' if row['detection_rate'] > 0.40
        else '🔴 Poor', axis=1)
})

# Sort by detection rate
summary = summary.sort_values('Detection Rate', ascending=False)

print("="*120)
print("COMPREHENSIVE SUMMARY: One-Class-Out Holdout Detection Performance")
print("="*120)
print("\n" + summary.to_string(index=False))

print("\n\n" + "="*120)
print("LEGEND:")
print("="*120)
print("Detection Rate:  % of holdout samples correctly identified as 'unknown' (higher is better)")
print("FPR:             False Positive Rate - % of known samples incorrectly flagged (lower is better)")
print("Distance Ratio:  How far holdout is from training data (higher = more distinct)")
print("Conf Diff:       Confidence difference (Known - Holdout); positive = model less confident on holdout")
print("Test Acc:        Accuracy on known classes when this class is held out")
print("Verdict:         Overall assessment for holdout detection suitability")

print("\n\n" + "="*120)
print("KEY FINDINGS:")
print("="*120)
print("\n1. BEST PERFORMERS (Easy to detect as unknown):")
print("   🟢 face_landmarking     - 100% detection, 68x distance ratio (massively distinct)")
print("   🟢 audio_classification - 100% detection, 5x distance (distinct audio features)")
print("   🟢 object_detection     - 99% detection, 10x distance (unique visual patterns)")

print("\n2. WORST PERFORMERS (Confused with known classes):")
print("   🔴 gesture_recognition  - 7% detection, 2.6x distance (too similar to other classes)")
print("   🔴 face_detection       - 38% detection, 4.2x distance (overlaps with face tasks)")

print("\n3. CRITICAL INSIGHT:")
print("   💡 Distance metric is the key differentiator!")
print("      - Classes >5x distance ratio: easily detected")
print("      - Classes <3x distance ratio: difficult to distinguish")
print("      - Confidence scores are unreliable (RF overconfidence)")

print("\n4. TRADE-OFF CHALLENGE:")
print("   ⚖️  Average FPR is 15.2% - we flag many known samples as uncertain")
print("      - This is acceptable if detecting unknowns is critical")
print("      - For production, may need threshold tuning per use case")

print("\n5. PRACTICAL RECOMMENDATIONS:")
print("   ✓ Use face_landmarking, object_detection, audio for open-set scenarios")
print("   ✗ Avoid gesture_recognition, face_detection as holdout classes")
print("   🔧 Consider ensemble methods or deep metric learning for better separation")

print("\n" + "="*120)

# Save summary
summary.to_csv('eval_results/rf_holdout_summary.csv', index=False)
print("\nSummary table saved to: eval_results/rf_holdout_summary.csv")
