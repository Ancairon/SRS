"""
k-NN Distance Features - Idea #3

Instead of just minimum distance, use richer distance statistics:
- Mean distance to k nearest neighbors
- Std deviation of k nearest neighbors
- Ratio of distances (k-th / 1st)
- Local Outlier Factor concept

These capture local density and neighborhood structure better.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
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


def compute_knn_features(X_test, X_train, k=10):
    """
    Compute k-NN based distance features:
    - min_dist: distance to nearest neighbor
    - mean_dist_k: mean distance to k nearest neighbors
    - std_dist_k: std deviation of distances to k nearest neighbors
    - dist_ratio: k-th distance / 1st distance (spread)
    """
    # Sample training data if too large
    if len(X_train) > 5000:
        indices = np.random.choice(len(X_train), 5000, replace=False)
        X_train_sample = X_train[indices]
    else:
        X_train_sample = X_train
    
    # Use NearestNeighbors for efficient k-NN
    nbrs = NearestNeighbors(n_neighbors=min(k, len(X_train_sample)), 
                           metric='euclidean', 
                           n_jobs=-1)
    nbrs.fit(X_train_sample)
    
    distances, indices = nbrs.kneighbors(X_test)
    
    # Compute features
    min_dist = distances[:, 0]  # 1st nearest neighbor
    mean_dist_k = np.mean(distances, axis=1)  # mean of k neighbors
    std_dist_k = np.std(distances, axis=1)  # std of k neighbors
    
    # Ratio: how spread out are the neighbors?
    # High ratio = isolated point (far from dense region)
    if distances.shape[1] > 1:
        dist_ratio = distances[:, -1] / (distances[:, 0] + 1e-10)
    else:
        dist_ratio = np.ones(len(X_test))
    
    return {
        'min_dist': min_dist,
        'mean_dist_k': mean_dist_k,
        'std_dist_k': std_dist_k,
        'dist_ratio': dist_ratio
    }


def optimize_knn_thresholds(features_known, features_holdout, metric_name,
                           fpr_targets=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]):
    """
    Find optimal threshold for a given k-NN metric.
    """
    results = []
    
    for target_fpr in fpr_targets:
        threshold = np.percentile(features_known, 100 * (1 - target_fpr))
        
        actual_fpr = np.sum(features_known > threshold) / len(features_known)
        detection_rate = np.sum(features_holdout > threshold) / len(features_holdout)
        
        results.append({
            'metric': metric_name,
            'target_fpr': target_fpr,
            'threshold': threshold,
            'actual_fpr': actual_fpr,
            'detection_rate': detection_rate
        })
    
    df_results = pd.DataFrame(results)
    
    # Maximize detection while keeping FPR < 0.20
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
    k = 10  # number of nearest neighbors
    
    # Get all classes
    all_classes = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d)) and "idle" not in d.lower()
    ]
    
    print(f"Found {len(all_classes)} classes: {all_classes}")
    print(f"Window size: {seq_len} samples = 3 seconds")
    print(f"Strategy: k-NN distance features (k={k}) - Idea #3")
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
        
        y_pred_test = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy: {test_accuracy:.4f} | Train time: {train_time:.2f}s")
        
        # Compute k-NN features
        print(f"\n  Computing k-NN distance features (k={k})...")
        features_test = compute_knn_features(X_test, X_train, k=k)
        features_holdout = compute_knn_features(X_holdout, X_train, k=k)
        
        # Compare metrics
        print(f"\n  Feature statistics:")
        for metric_name in ['min_dist', 'mean_dist_k', 'std_dist_k', 'dist_ratio']:
            mean_known = np.mean(features_test[metric_name])
            mean_holdout = np.mean(features_holdout[metric_name])
            ratio = mean_holdout / mean_known if mean_known > 0 else 0
            print(f"    {metric_name:12s} - Known: {mean_known:8.2f}, Holdout: {mean_holdout:8.2f}, Ratio: {ratio:.2f}x")
        
        # Try each metric and find best
        print(f"\n  Optimizing thresholds for each metric...")
        metric_results = {}
        
        for metric_name in ['min_dist', 'mean_dist_k', 'std_dist_k', 'dist_ratio']:
            best_config, all_configs = optimize_knn_thresholds(
                features_test[metric_name],
                features_holdout[metric_name],
                metric_name
            )
            metric_results[metric_name] = best_config
        
        # Show comparison
        print(f"\n  Metric comparison:")
        comparison_data = []
        for metric_name, config in metric_results.items():
            comparison_data.append({
                'metric': metric_name,
                'detection': config['detection_rate'],
                'fpr': config['actual_fpr']
            })
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Use best metric
        best_metric_name = max(metric_results.items(), 
                              key=lambda x: x[1]['detection_rate'])[0]
        best_config = metric_results[best_metric_name]
        
        print(f"\n  Best metric: {best_metric_name}")
        print(f"    Detection rate: {best_config['detection_rate']:.1%}")
        print(f"    FPR: {best_config['actual_fpr']:.1%}")
        print(f"    Threshold: {best_config['threshold']:.2f}")
        
        # Apply best threshold
        threshold = best_config['threshold']
        flags_test = features_test[best_metric_name] > threshold
        flags_holdout = features_holdout[best_metric_name] > threshold
        
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
        
        # Try ensemble: combine multiple metrics with OR logic
        print(f"\n  Testing ensemble (OR combination of all metrics)...")
        ensemble_flags_test = np.zeros(len(X_test), dtype=bool)
        ensemble_flags_holdout = np.zeros(len(X_holdout), dtype=bool)
        
        for metric_name, config in metric_results.items():
            threshold = config['threshold']
            ensemble_flags_test |= features_test[metric_name] > threshold
            ensemble_flags_holdout |= features_holdout[metric_name] > threshold
        
        ensemble_fpr = np.sum(ensemble_flags_test) / len(ensemble_flags_test)
        ensemble_detection = np.sum(ensemble_flags_holdout) / len(ensemble_flags_holdout)
        
        ensemble_non_flagged = ~ensemble_flags_test
        if np.sum(ensemble_non_flagged) > 0:
            ensemble_accuracy = accuracy_score(
                y_test[ensemble_non_flagged],
                y_pred_test[ensemble_non_flagged]
            )
        else:
            ensemble_accuracy = 0.0
        
        print(f"    Ensemble detection: {ensemble_detection:.1%}")
        print(f"    Ensemble FPR: {ensemble_fpr:.1%}")
        print(f"    Ensemble accuracy (non-flagged): {ensemble_accuracy:.4f}")
        
        # Store results (use ensemble if better)
        if ensemble_detection > detection_rate and ensemble_fpr < 0.25:
            print(f"  → Using ensemble (better detection)")
            result = {
                'excluded_class': excluded_class,
                'detection_rate': ensemble_detection,
                'fpr': ensemble_fpr,
                'accuracy_non_flagged': ensemble_accuracy,
                'test_accuracy': test_accuracy,
                'best_metric': 'ensemble',
                'train_time': train_time
            }
        else:
            print(f"  → Using single metric: {best_metric_name}")
            result = {
                'excluded_class': excluded_class,
                'detection_rate': detection_rate,
                'fpr': actual_fpr,
                'accuracy_non_flagged': accuracy_non_flagged,
                'test_accuracy': test_accuracy,
                'best_metric': best_metric_name,
                'train_time': train_time
            }
        
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - k-NN Distance Features")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_rate', ascending=False)
    
    print(df_results[[
        'excluded_class', 'detection_rate', 'fpr', 'best_metric',
        'accuracy_non_flagged', 'test_accuracy'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance:")
    print(f"  Detection rate: {df_results['detection_rate'].mean():.1%} (±{df_results['detection_rate'].std():.1%})")
    print(f"  False positive rate: {df_results['fpr'].mean():.1%} (±{df_results['fpr'].std():.1%})")
    print(f"  Test accuracy: {df_results['test_accuracy'].mean():.4f}")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f}")
    
    print(f"\n  Best metric usage:")
    print(df_results['best_metric'].value_counts().to_string())
    
    # Save
    output_file = "eval_results/rf_holdout_knn.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
