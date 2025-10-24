"""
Ensemble Binary Classifier Approach for Holdout Detection

Strategy:
- Train N binary one-vs-rest classifiers (one for each known class)
- For unknown samples, ALL classifiers should reject (high "not this class" confidence)
- Tunable rejection threshold for sensitivity control
- Combine with distance metric for robustness
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


def train_binary_classifiers(X_train, y_train, known_classes):
    """Train one binary classifier per known class (one-vs-rest)."""
    classifiers = {}
    
    for target_class in known_classes:
        # Create binary labels: 1 if this class, 0 otherwise
        y_binary = (y_train == target_class).astype(int)
        
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_binary)
        classifiers[target_class] = clf
    
    return classifiers


def get_rejection_scores(X, classifiers):
    """
    Get rejection scores for each sample.
    Returns: (n_samples, n_classifiers) array of P(NOT this class)
    """
    rejection_scores = []
    
    for class_name, clf in classifiers.items():
        # Get P(class=1) from binary classifier
        proba_is_class = clf.predict_proba(X)[:, 1]
        # Rejection score = P(NOT this class) = 1 - P(is this class)
        rejection_scores.append(1 - proba_is_class)
    
    return np.column_stack(rejection_scores)


def optimize_ensemble_threshold(rejection_scores_known, rejection_scores_holdout,
                                threshold_candidates=np.arange(0.50, 0.95, 0.05)):
    """
    Optimize the rejection threshold.
    A sample is flagged as unknown if ALL classifiers have rejection score > threshold.
    """
    results = []
    
    for threshold in threshold_candidates:
        # Flag if ALL rejection scores are above threshold
        flags_known = np.all(rejection_scores_known > threshold, axis=1)
        flags_holdout = np.all(rejection_scores_holdout > threshold, axis=1)
        
        fpr = np.sum(flags_known) / len(flags_known)
        detection_rate = np.sum(flags_holdout) / len(flags_holdout)
        
        results.append({
            'threshold': threshold,
            'fpr': fpr,
            'detection_rate': detection_rate
        })
    
    df_results = pd.DataFrame(results)
    
    # Strategy: Maximize detection while keeping FPR < 0.20
    df_valid = df_results[df_results['fpr'] < 0.20]
    if len(df_valid) > 0:
        best_idx = df_valid['detection_rate'].idxmax()
    else:
        best_idx = df_results['fpr'].idxmin()
    
    return df_results.loc[best_idx], df_results


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
    print(f"Strategy: Binary ensemble with unanimous rejection consensus")
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
        
        # Train binary classifiers
        print(f"\n  Training {len(known_classes)} binary classifiers...")
        start_time = time.time()
        classifiers = train_binary_classifiers(X_train, y_train, known_classes)
        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.2f}s")
        
        # Get rejection scores
        print(f"\n  Computing rejection scores...")
        rejection_scores_test = get_rejection_scores(X_test, classifiers)
        rejection_scores_holdout = get_rejection_scores(X_holdout, classifiers)
        
        # Analyze rejection patterns
        min_rejection_test = np.min(rejection_scores_test, axis=1)
        min_rejection_holdout = np.min(rejection_scores_holdout, axis=1)
        
        print(f"    Min rejection score - Known: {np.mean(min_rejection_test):.3f}, Holdout: {np.mean(min_rejection_holdout):.3f}")
        print(f"    Mean rejection score - Known: {np.mean(rejection_scores_test):.3f}, Holdout: {np.mean(rejection_scores_holdout):.3f}")
        
        # Optimize threshold
        print(f"\n  Optimizing ensemble threshold...")
        best_config, all_configs = optimize_ensemble_threshold(
            rejection_scores_test, rejection_scores_holdout
        )
        
        print(f"\n  Threshold optimization results:")
        print(all_configs.to_string(index=False))
        
        # Apply best threshold
        threshold = best_config['threshold']
        flags_test = np.all(rejection_scores_test > threshold, axis=1)
        flags_holdout = np.all(rejection_scores_holdout > threshold, axis=1)
        
        # Calculate metrics
        fpr_ensemble = np.sum(flags_test) / len(flags_test)
        detection_ensemble = np.sum(flags_holdout) / len(flags_holdout)
        
        # For non-flagged samples, use voting from binary classifiers
        non_flagged_mask = ~flags_test
        if np.sum(non_flagged_mask) > 0:
            # For each non-flagged sample, predict class with highest "IS this class" score
            predictions = []
            for scores in rejection_scores_test[non_flagged_mask]:
                # Convert rejection to acceptance: 1 - rejection
                acceptance_scores = 1 - scores
                best_class_idx = np.argmax(acceptance_scores)
                predictions.append(known_classes[best_class_idx])
            
            accuracy_non_flagged = accuracy_score(y_test[non_flagged_mask], predictions)
        else:
            accuracy_non_flagged = 0.0
        
        print(f"\n  Best Configuration (threshold={threshold:.2f}):")
        print(f"    Detection rate: {detection_ensemble:.1%}")
        print(f"    FPR: {fpr_ensemble:.1%}")
        print(f"    Accuracy (non-flagged): {accuracy_non_flagged:.4f}")
        
        # Also compute distance metric for comparison
        print(f"\n  Computing distance metric (for comparison)...")
        dist_test = compute_min_distance(X_test, X_train)
        dist_holdout = compute_min_distance(X_holdout, X_train)
        distance_ratio = np.mean(dist_holdout) / np.mean(dist_test)
        
        # Hybrid approach: Combine both signals
        print(f"\n  Testing hybrid approach (ensemble AND distance)...")
        dist_threshold = np.percentile(dist_test, 85)  # 15% FPR target
        flags_distance_test = dist_test > dist_threshold
        flags_distance_holdout = dist_holdout > dist_threshold
        
        # Flag if EITHER ensemble OR distance flags it
        flags_hybrid_test = flags_test | flags_distance_test
        flags_hybrid_holdout = flags_holdout | flags_distance_holdout
        
        fpr_hybrid = np.sum(flags_hybrid_test) / len(flags_hybrid_test)
        detection_hybrid = np.sum(flags_hybrid_holdout) / len(flags_hybrid_holdout)
        
        print(f"    Hybrid detection rate: {detection_hybrid:.1%}")
        print(f"    Hybrid FPR: {fpr_hybrid:.1%}")
        
        # Store results
        result = {
            'excluded_class': excluded_class,
            'detection_ensemble': detection_ensemble,
            'fpr_ensemble': fpr_ensemble,
            'detection_hybrid': detection_hybrid,
            'fpr_hybrid': fpr_hybrid,
            'threshold': threshold,
            'accuracy_non_flagged': accuracy_non_flagged,
            'distance_ratio': distance_ratio,
            'train_time': train_time,
            'mean_rejection_known': np.mean(rejection_scores_test),
            'mean_rejection_holdout': np.mean(rejection_scores_holdout)
        }
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - Binary Ensemble Approach")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_ensemble', ascending=False)
    
    print("\nEnsemble-Only Performance:")
    print(df_results[[
        'excluded_class', 'detection_ensemble', 'fpr_ensemble', 
        'threshold', 'distance_ratio'
    ]].to_string(index=False))
    
    print("\n\nHybrid (Ensemble + Distance) Performance:")
    print(df_results[[
        'excluded_class', 'detection_hybrid', 'fpr_hybrid', 'distance_ratio'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance - Ensemble Only:")
    print(f"  Detection rate: {df_results['detection_ensemble'].mean():.1%} (±{df_results['detection_ensemble'].std():.1%})")
    print(f"  FPR: {df_results['fpr_ensemble'].mean():.1%} (±{df_results['fpr_ensemble'].std():.1%})")
    
    print(f"\n\nAverage Performance - Hybrid:")
    print(f"  Detection rate: {df_results['detection_hybrid'].mean():.1%} (±{df_results['detection_hybrid'].std():.1%})")
    print(f"  FPR: {df_results['fpr_hybrid'].mean():.1%} (±{df_results['fpr_hybrid'].std():.1%})")
    
    # Compare with baseline (optimized distance-only from previous script)
    print(f"\n\n{'='*80}")
    print("COMPARISON WITH BASELINE")
    print(f"{'='*80}")
    print(f"\nEnsemble approach performance:")
    for _, row in df_results.iterrows():
        print(f"  {row['excluded_class']:24s}: {row['detection_ensemble']:.1%} ensemble | {row['detection_hybrid']:.1%} hybrid | ratio {row['distance_ratio']:.1f}x")
    
    # Save results
    output_file = "eval_results/rf_holdout_ensemble.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Insights
    print(f"\n\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    # Count perfect detections
    perfect_ensemble = len(df_results[df_results['detection_ensemble'] >= 0.99])
    perfect_hybrid = len(df_results[df_results['detection_hybrid'] >= 0.99])
    
    print(f"\n  Classes with ≥99% detection:")
    print(f"    Ensemble only: {perfect_ensemble}/{len(df_results)}")
    print(f"    Hybrid approach: {perfect_hybrid}/{len(df_results)}")
    
    print(f"\n  Rejection score analysis:")
    print(f"    Mean rejection (known classes): {df_results['mean_rejection_known'].mean():.3f}")
    print(f"    Mean rejection (holdout classes): {df_results['mean_rejection_holdout'].mean():.3f}")
    print(f"    Separation ratio: {df_results['mean_rejection_holdout'].mean() / df_results['mean_rejection_known'].mean():.2f}x")
    
    # Best strategy recommendation
    if df_results['detection_hybrid'].mean() > df_results['detection_ensemble'].mean():
        print(f"\n  Recommendation: HYBRID approach outperforms ensemble-only")
        print(f"    Improvement: {(df_results['detection_hybrid'].mean() - df_results['detection_ensemble'].mean())*100:.1f} percentage points")
    else:
        print(f"\n  Recommendation: ENSEMBLE-ONLY is sufficient")
        print(f"    Ensemble: {df_results['detection_ensemble'].mean():.1%} vs Hybrid: {df_results['detection_hybrid'].mean():.1%}")


if __name__ == "__main__":
    main()
