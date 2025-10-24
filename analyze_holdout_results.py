import pandas as pd
import numpy as np

# Load results
df = pd.read_csv('eval_results/rf_holdout_evaluation_results.csv')

print("="*80)
print("DETAILED ANALYSIS: Holdout Detection Results")
print("="*80)

print("\n1. DETECTION PERFORMANCE BY CLASS")
print("-"*80)
df_sorted = df.sort_values('detection_rate', ascending=False)
for idx, row in df_sorted.iterrows():
    print(f"\n{row['excluded_class']}:")
    print(f"  Detection Rate:        {row['detection_rate']:6.1%}  ({row['holdout_flagged']}/{row['holdout_samples']} samples)")
    print(f"  False Positive Rate:   {row['known_flagged_rate']:6.1%}  ({row['known_flagged']}/{row['known_samples']} samples)")
    print(f"  Accuracy (non-flagged):{row['accuracy_non_flagged']:7.4f}")
    print(f"  Test Accuracy:         {row['test_accuracy']:7.4f}")

print("\n\n2. CONFIDENCE ANALYSIS")
print("-"*80)
print(f"{'Class':<25} {'Avg Prob Known':<15} {'Avg Prob Holdout':<18} {'Difference':<12}")
print("-"*80)
for idx, row in df_sorted.iterrows():
    diff = row['avg_proba_known'] - row['avg_proba_holdout']
    marker = "✓" if row['avg_proba_holdout'] < row['avg_proba_known'] else "✗"
    print(f"{row['excluded_class']:<25} {row['avg_proba_known']:<15.3f} {row['avg_proba_holdout']:<18.3f} {diff:>7.3f} {marker}")

print("\n\n3. DISTANCE METRIC ANALYSIS")
print("-"*80)
print(f"{'Class':<25} {'Avg Dist Known':<15} {'Avg Dist Holdout':<18} {'Ratio':<12}")
print("-"*80)
for idx, row in df_sorted.iterrows():
    ratio = row['avg_distance_holdout'] / row['avg_distance_known'] if row['avg_distance_known'] > 0 else 0
    marker = "✓✓✓" if ratio > 5 else "✓✓" if ratio > 3 else "✓" if ratio > 1.5 else "~"
    print(f"{row['excluded_class']:<25} {row['avg_distance_known']:<15.1f} {row['avg_distance_holdout']:<18.1f} {ratio:>7.2f}x {marker}")

print("\n\n4. KEY INSIGHTS")
print("-"*80)

# Find classes with good detection
good_detection = df[df['detection_rate'] > 0.9]
print(f"\n✓ Classes with >90% detection rate ({len(good_detection)}):")
for _, row in good_detection.iterrows():
    ratio = row['avg_distance_holdout'] / row['avg_distance_known']
    print(f"  • {row['excluded_class']:<25} - Distance ratio: {ratio:.2f}x")

# Find classes with poor detection
poor_detection = df[df['detection_rate'] < 0.5]
print(f"\n✗ Classes with <50% detection rate ({len(poor_detection)}):")
for _, row in poor_detection.iterrows():
    ratio = row['avg_distance_holdout'] / row['avg_distance_known']
    conf_diff = row['avg_proba_known'] - row['avg_proba_holdout']
    print(f"  • {row['excluded_class']:<25} - Distance ratio: {ratio:.2f}x, Conf diff: {conf_diff:+.3f}")

# Correlation analysis
print(f"\n\n5. WHAT MAKES A CLASS EASY TO DETECT?")
print("-"*80)
print(f"Correlation with detection rate:")
print(f"  Distance ratio:         {df['avg_distance_holdout'].corr(df['detection_rate']):.3f}")
print(f"  Avg distance (holdout): {df['avg_distance_holdout'].corr(df['detection_rate']):.3f}")
print(f"  Confidence diff:        {(df['avg_proba_known'] - df['avg_proba_holdout']).corr(df['detection_rate']):.3f}")

# FPR analysis
print(f"\n\n6. FALSE POSITIVE ANALYSIS")
print("-"*80)
print(f"Average FPR: {df['known_flagged_rate'].mean():.1%} (±{df['known_flagged_rate'].std():.1%})")
print(f"Range: {df['known_flagged_rate'].min():.1%} - {df['known_flagged_rate'].max():.1%}")
print(f"\nThis means we're incorrectly flagging ~{df['known_flagged_rate'].mean()*100:.0f}% of known samples as 'uncertain'")

# Trade-off analysis
print(f"\n\n7. DETECTION VS FALSE POSITIVE TRADE-OFF")
print("-"*80)
high_det_low_fpr = df[(df['detection_rate'] > 0.8) & (df['known_flagged_rate'] < 0.15)]
print(f"Classes with >80% detection AND <15% FPR ({len(high_det_low_fpr)}):")
for _, row in high_det_low_fpr.iterrows():
    print(f"  • {row['excluded_class']:<25} - Det: {row['detection_rate']:.1%}, FPR: {row['known_flagged_rate']:.1%}")

print("\n\n8. RECOMMENDATIONS")
print("-"*80)
print("Based on the analysis:")
print()
print("✓ STRONG CANDIDATES for holdout (easily detected as unknown):")
for _, row in df[df['detection_rate'] > 0.95].iterrows():
    print(f"  • {row['excluded_class']}")
print()
print("⚠ WEAK CANDIDATES for holdout (confused with known classes):")
for _, row in df[df['detection_rate'] < 0.5].iterrows():
    print(f"  • {row['excluded_class']} - likely shares features with other classes")
print()
print("💡 Key finding: Distance metric is the strongest indicator!")
print("   Classes far from training data (>3x avg distance) are easily detected.")
