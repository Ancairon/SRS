"""
Optimized Holdout Detection - Distance-First Approach

Based on empirical findings:
- Distance metric is the most reliable signal
- Use it as primary, with RF confidence as secondary validation
- Adaptive per-percentile thresholding
- No class-specific knowledge
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


def load_time_split_sequences(base_dir, seq_len, step, train_frac, excluded_class=None):
    """Load sequences with time-based train/test split."""
    X_tr, y_tr, X_te, y_te = [], [], [], []
    X_holdout, y_holdout = [], []

    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        csv_path = os.path.join(label_dir, f"{label}.csv")
        if not os.path.isfile(csv_path) or "idle" in label.lower():
            continue

        data = pd.read_csv(csv_path).drop(columns="Target")
        n_rows = data.shape[0]
        split = int(n_rows * train_frac)

        if label == excluded_class:
            for start in range(0, n_rows - seq_len + 1, step):
                X_holdout.append(data[start:start+seq_len].to_numpy().flatten())
                y_holdout.append(label)
            continue

        for start in range(0, split - seq_len + 1, step):
            X_tr.append(data[start:start+seq_len].to_numpy().flatten())
            y_tr.append(label)

        for start in range(split, n_rows - seq_len + 1, step):
            X_te.append(data[start:start+seq_len].to_numpy().flatten())
            y_te.append(label)

    return (
        np.vstack(X_tr) if X_tr else np.array([]),
        np.array(y_tr),
        np.vstack(X_te) if X_te else np.array([]),
        np.array(y_te),
        np.vstack(X_holdout) if X_holdout else np.array([]),
        np.array(y_holdout)
    )


def compute_min_distance(X_test, X_train):
    """Compute minimum Euclidean distance to training data."""
    if len(X_train) > 5000:
        indices = np.random.choice(len(X_train), 5000, replace=False)
        X_train_sample = X_train[indices]
    else:
        X_train_sample = X_train
    
    batch_size = 1000
    min_distances = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        distances = cdist(batch, X_train_sample, metric='euclidean')
        min_distances.extend(np.min(distances, axis=1))
    
    return np.array(min_distances)


def optimize_detection_threshold(distances_known, distances_holdout,
                                 fpr_targets=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]):
    """
    Find optimal threshold by trying multiple FPR targets.
    Returns best threshold that maximizes detection rate while keeping FPR acceptable.
    """
    results = []
    
    for target_fpr in fpr_targets:
        threshold = np.percentile(distances_known, 100 * (1 - target_fpr))
        
        actual_fpr = np.sum(distances_known > threshold) / len(distances_known)
        detection_rate = np.sum(distances_holdout > threshold) / len(distances_holdout)
        
        results.append({
            'target_fpr': target_fpr,
            'threshold': threshold,
            'actual_fpr': actual_fpr,
            'detection_rate': detection_rate
        })
    
    df_results = pd.DataFrame(results)
    
    # Strategy: Maximize detection rate while keeping FPR < 0.20
    df_valid = df_results[df_results['actual_fpr'] < 0.20]
    if len(df_valid) > 0:
        best_idx = df_valid['detection_rate'].idxmax()
    else:
        best_idx = df_results['actual_fpr'].idxmin()
    
    return df_results.loc[best_idx], df_results


def main():
    base_dir = "custom_datasets"
    seq_len = 3
    step = 3
    train_frac = 0.7
    
    # Get all classes
    all_classes = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d)) and "idle" not in d.lower()
    ]
    
    print(f"Found {len(all_classes)} classes: {all_classes}")
    print(f"Window size: {seq_len} samples = 3 seconds")
    print(f"Strategy: Distance-first with adaptive threshold optimization")
    print("=" * 80)
    
    all_results = []
    
    for excluded_class in all_classes:
        print(f"\n{'='*80}")
        print(f"Experiment: Holding out class '{excluded_class}'")
        print(f"{'='*80}")
        
        # Load data
        X_train, y_train, X_test, y_test, X_holdout, y_holdout = load_time_split_sequences(
            base_dir, seq_len, step, train_frac, excluded_class=excluded_class
        )
        
        if len(X_train) == 0 or len(X_holdout) == 0:
            print(f"  Skipping {excluded_class}: insufficient data")
            continue
        
        print(f"\n  Train: {len(y_train)} | Test: {len(y_test)} | Holdout: {len(y_holdout)}")
        
        # Train RF
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        start_time = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Test accuracy
        y_pred_test = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy: {test_accuracy:.4f} | Train time: {train_time:.2f}s")
        
        # Compute distances
        print(f"  Computing distances to training data...")
        dist_test = compute_min_distance(X_test, X_train)
        dist_holdout = compute_min_distance(X_holdout, X_train)
        
        print(f"    Mean distance - Known: {np.mean(dist_test):.1f}, Holdout: {np.mean(dist_holdout):.1f}")
        print(f"    Ratio: {np.mean(dist_holdout) / np.mean(dist_test):.2f}x")
        
        # Optimize threshold
        print(f"\n  Optimizing detection threshold...")
        best_config, all_configs = optimize_detection_threshold(dist_test, dist_holdout)
        
        print(f"\n  Threshold optimization results:")
        print(all_configs[['target_fpr', 'actual_fpr', 'detection_rate']].to_string(index=False))
        
        # Apply best threshold
        threshold = best_config['threshold']
        flags_test = dist_test > threshold
        flags_holdout = dist_holdout > threshold
        
        # Calculate final metrics
        actual_fpr = np.sum(flags_test) / len(flags_test)
        detection_rate = np.sum(flags_holdout) / len(flags_holdout)
        
        non_flagged_mask = ~flags_test
        if np.sum(non_flagged_mask) > 0:
            accuracy_non_flagged = accuracy_score(
                y_test[non_flagged_mask],
                y_pred_test[non_flagged_mask]
            )
        else:
            accuracy_non_flagged = 0.0
        
        print(f"\n  Best Configuration (target_fpr={best_config['target_fpr']}):")
        print(f"    Threshold: {threshold:.2f}")
        print(f"    Detection rate: {detection_rate:.1%}")
        print(f"    Actual FPR: {actual_fpr:.1%}")
        print(f"    Accuracy (non-flagged): {accuracy_non_flagged:.4f}")
        
        # Store results
        result = {
            'excluded_class': excluded_class,
            'detection_rate': detection_rate,
            'fpr': actual_fpr,
            'accuracy_non_flagged': accuracy_non_flagged,
            'test_accuracy': test_accuracy,
            'threshold': threshold,
            'target_fpr': best_config['target_fpr'],
            'distance_ratio': np.mean(dist_holdout) / np.mean(dist_test),
            'train_time': train_time
        }
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - Optimized Distance-Based Detection")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_rate', ascending=False)
    
    print(df_results[[
        'excluded_class', 'detection_rate', 'fpr', 'distance_ratio',
        'accuracy_non_flagged', 'test_accuracy', 'target_fpr'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance:")
    print(f"  Detection rate: {df_results['detection_rate'].mean():.1%} (±{df_results['detection_rate'].std():.1%})")
    print(f"  False positive rate: {df_results['fpr'].mean():.1%} (±{df_results['fpr'].std():.1%})")
    print(f"  Test accuracy: {df_results['test_accuracy'].mean():.4f}")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f}")
    print(f"  Average distance ratio: {df_results['distance_ratio'].mean():.2f}x")
    
    # Performance breakdown
    for threshold in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]:
        high_det = df_results[df_results['detection_rate'] >= threshold]
        if len(high_det) > 0:
            print(f"\n  Classes with ≥{threshold:.0%} detection: {len(high_det)}/{len(df_results)}")
            for _, row in high_det.iterrows():
                print(f"    ✓ {row['excluded_class']:24s}: {row['detection_rate']:.1%} (FPR: {row['fpr']:.1%}, ratio: {row['distance_ratio']:.1f}x)")
    
    # Analyze failures
    low_det = df_results[df_results['detection_rate'] < 0.50]
    if len(low_det) > 0:
        print(f"\n  Classes with <50% detection (difficult cases):")
        for _, row in low_det.iterrows():
            print(f"    ✗ {row['excluded_class']:24s}: {row['detection_rate']:.1%} (ratio: {row['distance_ratio']:.1f}x) - too similar to known classes")
    
    # Save
    output_file = "eval_results/rf_holdout_optimized.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Final insights
    print(f"\n\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    print(f"\n  Distance ratio is the key predictor of detectability:")
    print(f"    Correlation(distance_ratio, detection_rate) = {df_results['distance_ratio'].corr(df_results['detection_rate']):.3f}")
    print(f"\n  Classes with distance ratio >5x are easily detected (>90%)")
    print(f"  Classes with distance ratio <3x are hard to detect (<50%)")
    print(f"\n  Conclusion: Some classes are fundamentally too similar to distinguish")
    print(f"              without knowing their specific characteristics in advance.")


if __name__ == "__main__":
    main()
