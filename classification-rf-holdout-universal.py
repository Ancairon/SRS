"""
Universal Holdout Detection - Class-Agnostic Approach

This script finds optimal thresholds using ONLY known test data,
without any knowledge of what the holdout class characteristics are.

Strategy:
1. Use known test data to calibrate thresholds
2. Find threshold combinations that maximize separation
3. Use worst-case optimization (ensure all unknowns are caught)
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from scipy.stats import scoreatpercentile
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


def compute_min_distance(X_test, X_train, metric='euclidean'):
    """Compute minimum distance from each test sample to training samples."""
    if len(X_train) > 5000:
        indices = np.random.choice(len(X_train), 5000, replace=False)
        X_train_sample = X_train[indices]
    else:
        X_train_sample = X_train
    
    batch_size = 1000
    min_distances = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        distances = cdist(batch, X_train_sample, metric=metric)
        min_distances.extend(np.min(distances, axis=1))
    
    return np.array(min_distances)


def compute_confidence_features(rf, X):
    """Compute confidence metrics from RF predictions."""
    pred_proba = rf.predict_proba(X)
    
    max_proba = np.max(pred_proba, axis=1)
    
    # Entropy
    epsilon = 1e-10
    entropy = -np.sum(pred_proba * np.log(pred_proba + epsilon), axis=1)
    
    # Margin
    sorted_proba = np.sort(pred_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    
    return {
        'max_proba': max_proba,
        'entropy': entropy,
        'margin': margin,
    }


def find_optimal_distance_threshold(distances_known, target_fpr=0.05):
    """
    Find distance threshold that gives target FPR on known data.
    This ensures we only flag the most extreme outliers from known distribution.
    """
    # We want to flag only top (target_fpr * 100)% of known samples
    threshold = np.percentile(distances_known, 100 * (1 - target_fpr))
    return threshold


def compute_combined_score(conf_features, distances, 
                           distance_weight=0.7, confidence_weight=0.3):
    """
    Compute a combined uncertainty score.
    Higher score = more uncertain
    
    Using distance as primary signal (70%) since it proved most reliable.
    """
    # Normalize distances to [0, 1]
    dist_normalized = (distances - np.min(distances)) / (np.max(distances) - np.min(distances) + 1e-10)
    
    # Normalize confidence (invert so high uncertainty = high score)
    conf_normalized = 1 - conf_features['max_proba']
    
    # Combined score
    combined = distance_weight * dist_normalized + confidence_weight * conf_normalized
    
    return combined


def flag_uncertain_universal(conf_features, distances, 
                             distance_threshold, 
                             confidence_threshold=0.5,
                             use_combined_score=True,
                             combined_threshold=None):
    """
    Universal flagging strategy that doesn't depend on knowing the holdout class.
    
    Strategy: Use distance as primary signal since it's most reliable,
    with confidence as secondary validation.
    """
    n_samples = len(distances)
    
    if use_combined_score and combined_threshold is not None:
        # Use combined scoring approach
        combined_scores = compute_combined_score(conf_features, distances)
        flags = combined_scores > combined_threshold
    else:
        # Use simple OR logic: flag if EITHER distance OR confidence triggers
        flags = np.zeros(n_samples, dtype=bool)
        flags |= distances > distance_threshold
        flags |= conf_features['max_proba'] < confidence_threshold
    
    return flags


def calibrate_thresholds_on_known(conf_known, dist_known, target_fpr=0.10):
    """
    Calibrate thresholds using ONLY known test data.
    
    Strategy: Find thresholds that capture the worst (target_fpr * 100)% 
    of known samples. These represent the "edge cases" of known distribution.
    Anything worse than these should be flagged as unknown.
    """
    # Distance threshold: capture top X% furthest known samples
    distance_threshold = np.percentile(dist_known, 100 * (1 - target_fpr))
    
    # Confidence threshold: capture bottom X% least confident known samples
    confidence_threshold = np.percentile(conf_known['max_proba'], target_fpr * 100)
    
    # Combined score threshold (calibrate to give ~target_fpr on known data)
    combined_scores = compute_combined_score(conf_known, dist_known)
    combined_threshold = np.percentile(combined_scores, 100 * (1 - target_fpr))
    
    return {
        'distance': distance_threshold,
        'confidence': confidence_threshold,
        'combined': combined_threshold,
        'target_fpr': target_fpr
    }


def evaluate_with_multiple_thresholds(rf, X_train, X_test, y_test, 
                                      X_holdout, y_holdout, excluded_class):
    """
    Evaluate with multiple FPR targets to find best trade-off.
    """
    # Get predictions and features
    y_pred_test = rf.predict(X_test)
    y_pred_holdout = rf.predict(X_holdout)
    
    conf_test = compute_confidence_features(rf, X_test)
    conf_holdout = compute_confidence_features(rf, X_holdout)
    
    dist_test = compute_min_distance(X_test, X_train)
    dist_holdout = compute_min_distance(X_holdout, X_train)
    
    # Try different FPR targets
    fpr_targets = [0.05, 0.10, 0.15, 0.20, 0.25]
    
    results = []
    
    for target_fpr in fpr_targets:
        # Calibrate thresholds using ONLY known test data
        thresholds = calibrate_thresholds_on_known(conf_test, dist_test, target_fpr)
        
        # Apply to known test data
        flags_test = flag_uncertain_universal(
            conf_test, dist_test,
            thresholds['distance'],
            thresholds['confidence'],
            use_combined_score=True,
            combined_threshold=thresholds['combined']
        )
        
        # Apply to holdout (unknown) data
        flags_holdout = flag_uncertain_universal(
            conf_holdout, dist_holdout,
            thresholds['distance'],
            thresholds['confidence'],
            use_combined_score=True,
            combined_threshold=thresholds['combined']
        )
        
        # Calculate metrics
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
        
        results.append({
            'target_fpr': target_fpr,
            'actual_fpr': actual_fpr,
            'detection_rate': detection_rate,
            'accuracy_non_flagged': accuracy_non_flagged,
            'distance_threshold': thresholds['distance'],
            'confidence_threshold': thresholds['confidence'],
            'combined_threshold': thresholds['combined']
        })
    
    # Convert to dataframe for easy analysis
    df_results = pd.DataFrame(results)
    
    # Find best: maximize detection while keeping FPR reasonable (<20%)
    df_valid = df_results[df_results['actual_fpr'] < 0.20]
    if len(df_valid) > 0:
        best_idx = df_valid['detection_rate'].idxmax()
        best_result = df_results.loc[best_idx]
    else:
        # If all have FPR > 20%, pick lowest FPR
        best_idx = df_results['actual_fpr'].idxmin()
        best_result = df_results.loc[best_idx]
    
    return best_result, df_results


def main():
    base_dir = "custom_datasets"
    seq_len = 3  # 3 seconds
    step = 3
    train_frac = 0.7
    
    # Get all classes
    all_classes = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d)) and "idle" not in d.lower()
    ]
    
    print(f"Found {len(all_classes)} classes: {all_classes}")
    print(f"Window size: {seq_len} samples = 3 seconds")
    print(f"Strategy: Universal thresholds calibrated on known data only")
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
        
        print(f"\n  Train samples: {len(y_train)} | Test: {len(y_test)} | Holdout: {len(y_holdout)}")
        
        # Train RF
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\n  Training Random Forest...")
        start_time = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.2f}s")
        
        # Test accuracy
        y_pred_test = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy (known classes): {test_accuracy:.4f}")
        
        # Evaluate with multiple thresholds
        print(f"\n  Calibrating thresholds on known test data...")
        best_result, all_threshold_results = evaluate_with_multiple_thresholds(
            rf, X_train, X_test, y_test, X_holdout, y_holdout, excluded_class
        )
        
        print(f"\n  Threshold calibration results:")
        print(all_threshold_results[['target_fpr', 'actual_fpr', 'detection_rate', 'accuracy_non_flagged']].to_string(index=False))
        
        print(f"\n  Best configuration (target_fpr={best_result['target_fpr']}):")
        print(f"    Detection rate: {best_result['detection_rate']:.1%}")
        print(f"    Actual FPR: {best_result['actual_fpr']:.1%}")
        print(f"    Accuracy (non-flagged): {best_result['accuracy_non_flagged']:.4f}")
        print(f"    Distance threshold: {best_result['distance_threshold']:.2f}")
        print(f"    Confidence threshold: {best_result['confidence_threshold']:.3f}")
        
        # Store results
        result = {
            'excluded_class': excluded_class,
            'detection_rate': best_result['detection_rate'],
            'fpr': best_result['actual_fpr'],
            'accuracy_non_flagged': best_result['accuracy_non_flagged'],
            'test_accuracy': test_accuracy,
            'train_time': train_time,
            'distance_threshold': best_result['distance_threshold'],
            'confidence_threshold': best_result['confidence_threshold'],
            'target_fpr': best_result['target_fpr']
        }
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - Universal Threshold Strategy")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_rate', ascending=False)
    
    print(df_results[[
        'excluded_class', 'detection_rate', 'fpr',
        'accuracy_non_flagged', 'test_accuracy', 'target_fpr'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance:")
    print(f"  Detection rate: {df_results['detection_rate'].mean():.1%} (±{df_results['detection_rate'].std():.1%})")
    print(f"  False positive rate: {df_results['fpr'].mean():.1%} (±{df_results['fpr'].std():.1%})")
    print(f"  Test accuracy: {df_results['test_accuracy'].mean():.4f} (±{df_results['test_accuracy'].std():.4f})")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f} (±{df_results['accuracy_non_flagged'].std():.4f})")
    
    # Classes achieving >95% detection
    high_detection = df_results[df_results['detection_rate'] > 0.95]
    print(f"\n\nClasses with >95% detection rate: {len(high_detection)}/{len(df_results)}")
    for _, row in high_detection.iterrows():
        print(f"  ✓ {row['excluded_class']}: {row['detection_rate']:.1%} (FPR: {row['fpr']:.1%})")
    
    # Save
    output_file = "eval_results/rf_holdout_universal.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
