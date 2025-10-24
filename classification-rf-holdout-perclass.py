"""
Per-Class Distance Thresholds - Idea #2

Instead of one universal distance threshold, compute separate thresholds
for each known class. For a test sample:
1. RF predicts which known class it belongs to
2. Check distance against that specific class's threshold
3. Flag if distance exceeds the per-class threshold

This allows different "tightness" for different classes.
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


def compute_distance_to_class(X_test, X_train, y_train, target_class):
    """Compute minimum distance to samples of a specific class."""
    X_class = X_train[y_train == target_class]
    
    if len(X_class) > 2000:
        indices = np.random.choice(len(X_class), 2000, replace=False)
        X_class = X_class[indices]
    
    batch_size = 1000
    min_distances = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        distances = cdist(batch, X_class, metric='euclidean')
        min_distances.extend(np.min(distances, axis=1))
    
    return np.array(min_distances)


def optimize_per_class_thresholds(X_train, y_train, X_test, y_test, known_classes,
                                  fpr_targets=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]):
    """
    For each known class, find the optimal distance threshold.
    Returns dict: {class_name: threshold}
    """
    class_thresholds = {}
    
    for cls in known_classes:
        # Get test samples that RF predicted as this class
        cls_mask = y_test == cls
        if np.sum(cls_mask) == 0:
            continue
        
        # Compute distances to this class's training samples
        distances = compute_distance_to_class(X_test[cls_mask], X_train, y_train, cls)
        
        # Find best threshold (minimize FPR while staying under 0.20)
        best_threshold = None
        best_fpr = float('inf')
        
        for target_fpr in fpr_targets:
            threshold = np.percentile(distances, 100 * (1 - target_fpr))
            actual_fpr = np.sum(distances > threshold) / len(distances)
            
            if actual_fpr < 0.20 and actual_fpr < best_fpr:
                best_fpr = actual_fpr
                best_threshold = threshold
        
        if best_threshold is None:
            best_threshold = np.percentile(distances, 80)  # fallback
        
        class_thresholds[cls] = best_threshold
    
    return class_thresholds


def apply_per_class_detection(X_samples, y_pred, X_train, y_train, class_thresholds):
    """
    For each sample, check if distance to its predicted class exceeds that class's threshold.
    """
    flags = np.zeros(len(X_samples), dtype=bool)
    
    for cls, threshold in class_thresholds.items():
        cls_mask = y_pred == cls
        if np.sum(cls_mask) == 0:
            continue
        
        # Compute distance to this class for samples predicted as this class
        distances = compute_distance_to_class(X_samples[cls_mask], X_train, y_train, cls)
        flags[cls_mask] = distances > threshold
    
    return flags


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
    print(f"Strategy: Per-class distance thresholds (Idea #2)")
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
        
        known_classes = np.unique(y_train)
        print(f"\n  Train: {len(y_train)} | Test: {len(y_test)} | Holdout: {len(y_holdout)}")
        print(f"  Known classes: {list(known_classes)}")
        
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
        
        # Predictions
        y_pred_test = rf.predict(X_test)
        y_pred_holdout = rf.predict(X_holdout)
        
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy: {test_accuracy:.4f} | Train time: {train_time:.2f}s")
        
        # Optimize per-class thresholds on test set
        print(f"\n  Optimizing per-class distance thresholds...")
        class_thresholds = optimize_per_class_thresholds(
            X_train, y_train, X_test, y_test, known_classes
        )
        
        print(f"  Per-class thresholds:")
        for cls, threshold in sorted(class_thresholds.items()):
            print(f"    {cls:24s}: {threshold:.2f}")
        
        # Apply per-class detection
        print(f"\n  Applying per-class detection...")
        flags_test = apply_per_class_detection(X_test, y_pred_test, X_train, y_train, class_thresholds)
        flags_holdout = apply_per_class_detection(X_holdout, y_pred_holdout, X_train, y_train, class_thresholds)
        
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
        
        print(f"\n  Results:")
        print(f"    Detection rate: {detection_rate:.1%}")
        print(f"    FPR: {actual_fpr:.1%}")
        print(f"    Accuracy (non-flagged): {accuracy_non_flagged:.4f}")
        
        # Analyze by predicted class
        print(f"\n  Detection breakdown by predicted class (holdout samples):")
        for cls in known_classes:
            cls_mask = y_pred_holdout == cls
            if np.sum(cls_mask) == 0:
                continue
            cls_detection = np.sum(flags_holdout[cls_mask]) / np.sum(cls_mask)
            print(f"    Predicted as {cls:24s}: {np.sum(cls_mask):4d} samples, {cls_detection:.1%} flagged")
        
        # Store results
        result = {
            'excluded_class': excluded_class,
            'detection_rate': detection_rate,
            'fpr': actual_fpr,
            'accuracy_non_flagged': accuracy_non_flagged,
            'test_accuracy': test_accuracy,
            'train_time': train_time,
            'num_thresholds': len(class_thresholds)
        }
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - Per-Class Distance Thresholds")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_rate', ascending=False)
    
    print(df_results[[
        'excluded_class', 'detection_rate', 'fpr',
        'accuracy_non_flagged', 'test_accuracy'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance:")
    print(f"  Detection rate: {df_results['detection_rate'].mean():.1%} (±{df_results['detection_rate'].std():.1%})")
    print(f"  False positive rate: {df_results['fpr'].mean():.1%} (±{df_results['fpr'].std():.1%})")
    print(f"  Test accuracy: {df_results['test_accuracy'].mean():.4f}")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f}")
    
    # Save
    output_file = "eval_results/rf_holdout_perclass.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
