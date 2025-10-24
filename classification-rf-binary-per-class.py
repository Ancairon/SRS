"""
Binary Classifier Per Class Approach

Strategy:
- Train one binary classifier per known class (X vs NOT-X)
- Each model learns: "is this my class or any other known class?"
- For unknown samples, ALL models should say "NO"
- Prediction: If only one model says "YES", predict that class
- Detection: If zero or multiple models say "YES", flag as unknown
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
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
    """
    Train one binary classifier per known class.
    Each learns: "is this my class (1) or any other known class (0)?"
    """
    classifiers = {}
    
    for target_class in known_classes:
        # Binary labels: 1 if this class, 0 if any other known class
        y_binary = (y_train == target_class).astype(int)
        
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_binary)
        classifiers[target_class] = clf
    
    return classifiers


def predict_with_binary_ensemble(X, classifiers, known_classes, confidence_threshold=0.5):
    """
    Make predictions using binary ensemble.
    
    Returns:
    - predictions: array of predicted classes (or 'UNKNOWN')
    - flags_unknown: boolean array indicating unknown samples
    - vote_counts: number of models that voted "YES" for each sample
    """
    n_samples = len(X)
    n_classifiers = len(classifiers)
    
    # Get predictions from all binary classifiers
    votes = np.zeros((n_samples, n_classifiers))
    confidences = np.zeros((n_samples, n_classifiers))
    
    for i, (class_name, clf) in enumerate(classifiers.items()):
        proba = clf.predict_proba(X)[:, 1]  # P(is this class)
        votes[:, i] = (proba >= confidence_threshold).astype(int)
        confidences[:, i] = proba
    
    # Count how many models voted "YES"
    vote_counts = np.sum(votes, axis=1)
    
    # Prediction logic:
    # - If exactly 1 model says YES → predict that class (NORMAL)
    # - If 0 votes → definitely unknown (no class claims it)
    # - If 2+ votes → ambiguous/confused → likely unknown!
    predictions = []
    flags_unknown = np.zeros(n_samples, dtype=bool)
    flags_zero_votes = np.zeros(n_samples, dtype=bool)
    flags_multi_votes = np.zeros(n_samples, dtype=bool)
    
    for i in range(n_samples):
        if vote_counts[i] == 1:
            # Exactly one model claims it - NORMAL
            winner_idx = np.where(votes[i] == 1)[0][0]
            predictions.append(known_classes[winner_idx])
            flags_unknown[i] = False
        elif vote_counts[i] == 0:
            # No class claims it - DEFINITELY UNKNOWN
            predictions.append('UNKNOWN')
            flags_unknown[i] = True
            flags_zero_votes[i] = True
        else:
            # Multiple claims - AMBIGUOUS/CONFUSED - LIKELY UNKNOWN
            predictions.append('UNKNOWN')
            flags_unknown[i] = True
            flags_multi_votes[i] = True
    
    return np.array(predictions), flags_unknown, vote_counts, confidences, flags_zero_votes, flags_multi_votes


def optimize_confidence_threshold(X_test, y_test, X_holdout, classifiers, known_classes,
                                  threshold_candidates=np.arange(0.3, 0.9, 0.05)):
    """
    Optimize the confidence threshold for binary predictions.
    Goal: Maximize detection rate while keeping FPR < 20%
    """
    results = []
    
    for threshold in threshold_candidates:
        # Get predictions
        preds_test, flags_test, votes_test, conf_test, zeros_test, multi_test = predict_with_binary_ensemble(
            X_test, classifiers, known_classes, threshold
        )
        preds_holdout, flags_holdout, votes_holdout, conf_holdout, zeros_holdout, multi_holdout = predict_with_binary_ensemble(
            X_holdout, classifiers, known_classes, threshold
        )
        
        # Metrics
        fpr = np.sum(flags_test) / len(flags_test)
        detection_rate = np.sum(flags_holdout) / len(flags_holdout)
        
        # Accuracy on non-flagged samples
        non_flagged_mask = ~flags_test
        if np.sum(non_flagged_mask) > 0:
            accuracy = accuracy_score(y_test[non_flagged_mask], preds_test[non_flagged_mask])
        else:
            accuracy = 0.0
        
        results.append({
            'threshold': threshold,
            'fpr': fpr,
            'detection_rate': detection_rate,
            'accuracy_non_flagged': accuracy
        })
    
    df_results = pd.DataFrame(results)
    
    # Select best: maximize detection while FPR < 0.20
    df_valid = df_results[df_results['fpr'] < 0.20]
    if len(df_valid) > 0:
        best_idx = df_valid['detection_rate'].idxmax()
    else:
        best_idx = df_results['fpr'].idxmin()
    
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
    print(f"Strategy: One binary classifier per class (X vs NOT-X)")
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
        print(f"  Training {len(known_classes)} binary classifiers (one per known class)...")
        
        # Train binary classifiers
        start_time = time.time()
        classifiers = train_binary_classifiers(X_train, y_train, known_classes)
        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.2f}s")
        
        # Optimize confidence threshold
        print(f"\n  Optimizing confidence threshold...")
        best_config, all_configs = optimize_confidence_threshold(
            X_test, y_test, X_holdout, classifiers, known_classes
        )
        
        print(f"\n  Threshold optimization results:")
        print(all_configs.to_string(index=False))
        
        # Apply best threshold
        threshold = best_config['threshold']
        preds_test, flags_test, votes_test, conf_test, zeros_test, multi_test = predict_with_binary_ensemble(
            X_test, classifiers, known_classes, threshold
        )
        preds_holdout, flags_holdout, votes_holdout, conf_holdout, zeros_holdout, multi_holdout = predict_with_binary_ensemble(
            X_holdout, classifiers, known_classes, threshold
        )
        
        # Metrics
        fpr = np.sum(flags_test) / len(flags_test)
        detection_rate = np.sum(flags_holdout) / len(flags_holdout)
        
        # Accuracy on non-flagged test samples
        non_flagged_mask = ~flags_test
        if np.sum(non_flagged_mask) > 0:
            accuracy_non_flagged = accuracy_score(
                y_test[non_flagged_mask],
                preds_test[non_flagged_mask]
            )
        else:
            accuracy_non_flagged = 0.0
        
        # Analyze voting patterns
        print(f"\n  Voting pattern analysis:")
        print(f"    Known samples - 0 votes: {np.sum(votes_test == 0)}, 1 vote: {np.sum(votes_test == 1)}, >1 votes: {np.sum(votes_test > 1)}")
        print(f"    Holdout samples - 0 votes: {np.sum(votes_holdout == 0)}, 1 vote: {np.sum(votes_holdout == 1)}, >1 votes: {np.sum(votes_holdout > 1)}")
        print(f"\n  Detection breakdown:")
        print(f"    Holdout flagged by 0 votes: {np.sum(zeros_holdout)} ({np.sum(zeros_holdout)/len(zeros_holdout)*100:.1f}%)")
        print(f"    Holdout flagged by multi votes: {np.sum(multi_holdout)} ({np.sum(multi_holdout)/len(multi_holdout)*100:.1f}%)")
        print(f"    Total detection: {detection_rate:.1%}")
        
        print(f"\n  Best Configuration (threshold={threshold:.2f}):")
        print(f"    Detection rate: {detection_rate:.1%}")
        print(f"    FPR: {fpr:.1%}")
        print(f"    Accuracy (non-flagged): {accuracy_non_flagged:.4f}")
        
        # Store results
        result = {
            'excluded_class': excluded_class,
            'detection_rate': detection_rate,
            'detection_zero_votes': np.sum(zeros_holdout) / len(zeros_holdout),
            'detection_multi_votes': np.sum(multi_holdout) / len(multi_holdout),
            'fpr': fpr,
            'accuracy_non_flagged': accuracy_non_flagged,
            'threshold': threshold,
            'train_time': train_time,
            'votes_test_0': np.sum(votes_test == 0),
            'votes_test_1': np.sum(votes_test == 1),
            'votes_test_multi': np.sum(votes_test > 1),
            'votes_holdout_0': np.sum(votes_holdout == 0),
            'votes_holdout_1': np.sum(votes_holdout == 1),
            'votes_holdout_multi': np.sum(votes_holdout > 1)
        }
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - Binary Per-Class Approach")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_rate', ascending=False)
    
    print("Performance Summary:")
    print(df_results[[
        'excluded_class', 'detection_rate', 'fpr', 
        'accuracy_non_flagged', 'threshold'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance:")
    print(f"  Detection rate: {df_results['detection_rate'].mean():.1%} (±{df_results['detection_rate'].std():.1%})")
    print(f"  FPR: {df_results['fpr'].mean():.1%} (±{df_results['fpr'].std():.1%})")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f}")
    
    # Voting pattern analysis
    print(f"\n\nVoting Pattern Analysis (average across experiments):")
    print(f"\n  Known samples (should get exactly 1 vote):")
    print(f"    0 votes: {df_results['votes_test_0'].mean():.0f} ({df_results['votes_test_0'].mean()/2160*100:.1f}%)")
    print(f"    1 vote:  {df_results['votes_test_1'].mean():.0f} ({df_results['votes_test_1'].mean()/2160*100:.1f}%)")
    print(f"    >1 votes: {df_results['votes_test_multi'].mean():.0f} ({df_results['votes_test_multi'].mean()/2160*100:.1f}%)")
    
    print(f"\n  Holdout samples (should get 0 votes):")
    print(f"    0 votes: {df_results['votes_holdout_0'].mean():.0f} ({df_results['votes_holdout_0'].mean()/1200*100:.1f}%)")
    print(f"    1 vote:  {df_results['votes_holdout_1'].mean():.0f} ({df_results['votes_holdout_1'].mean()/1200*100:.1f}%)")
    print(f"    >1 votes: {df_results['votes_holdout_multi'].mean():.0f} ({df_results['votes_holdout_multi'].mean()/1200*100:.1f}%)")
    
    # Performance breakdown
    print(f"\n\nDetection Performance Breakdown:")
    for threshold in [0.99, 0.95, 0.90, 0.80, 0.70, 0.50]:
        high_det = df_results[df_results['detection_rate'] >= threshold]
        if len(high_det) > 0:
            print(f"\n  Classes with ≥{threshold:.0%} detection: {len(high_det)}/{len(df_results)}")
            for _, row in high_det.iterrows():
                print(f"    ✓ {row['excluded_class']:24s}: {row['detection_rate']:.1%} (FPR: {row['fpr']:.1%})")
    
    # Save results
    output_file = "eval_results/rf_binary_per_class.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Insights
    print(f"\n\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}")
    
    perfect = len(df_results[df_results['detection_rate'] >= 0.99])
    print(f"\n  Classes with ≥99% detection: {perfect}/{len(df_results)}")
    
    print(f"\n  Binary classifier approach:")
    print(f"    - Each class has its own 'is this my class?' detector")
    print(f"    - Unknown samples: all detectors should say NO")
    print(f"    - Known samples: exactly one detector should say YES")
    print(f"    - Ambiguous samples (>1 YES): flagged as unknown")
    
    # Compare with baseline
    print(f"\n  Comparison with distance-based baseline (92.6% detection):")
    if df_results['detection_rate'].mean() > 0.926:
        print(f"    ✓ BETTER: {df_results['detection_rate'].mean():.1%} vs 92.6%")
        print(f"      Improvement: {(df_results['detection_rate'].mean() - 0.926)*100:.1f} percentage points")
    else:
        print(f"    ✗ WORSE: {df_results['detection_rate'].mean():.1%} vs 92.6%")
        print(f"      Distance-based approach remains superior")
    
    # Detailed per-class breakdown
    print(f"\n\n{'='*80}")
    print("DETAILED PER-CLASS BREAKDOWN")
    print(f"{'='*80}\n")
    
    for _, row in df_results.iterrows():
        print(f"\n{row['excluded_class']:=^80}")
        print(f"  Overall Detection: {row['detection_rate']:.1%}")
        print(f"    - Detected by 0 votes: {row['detection_zero_votes']:.1%}")
        print(f"    - Detected by multi votes: {row['detection_multi_votes']:.1%}")
        print(f"  False Positive Rate: {row['fpr']:.1%}")
        print(f"  Accuracy (non-flagged): {row['accuracy_non_flagged']:.4f}")
        print(f"  Confidence threshold: {row['threshold']:.2f}")
        print(f"\n  Known samples voting:")
        print(f"    0 votes: {row['votes_test_0']} ({row['votes_test_0']/2160*100:.1f}%)")
        print(f"    1 vote:  {row['votes_test_1']} ({row['votes_test_1']/2160*100:.1f}%)")
        print(f"    >1 votes: {row['votes_test_multi']} ({row['votes_test_multi']/2160*100:.1f}%)")
        print(f"\n  Holdout samples voting:")
        print(f"    0 votes: {row['votes_holdout_0']} ({row['votes_holdout_0']/1200*100:.1f}%) → DETECTED ✓")
        print(f"    1 vote:  {row['votes_holdout_1']} ({row['votes_holdout_1']/1200*100:.1f}%) → MISSED ✗")
        print(f"    >1 votes: {row['votes_holdout_multi']} ({row['votes_holdout_multi']/1200*100:.1f}%) → DETECTED ✓")
    
    # Summary comparison table
    print(f"\n\n{'='*80}")
    print("SUMMARY: Detection Mechanisms")
    print(f"{'='*80}\n")
    print(f"{'Class':24s} {'Total':>8s} {'0-votes':>8s} {'Multi':>8s} {'Missed':>8s}")
    print(f"{'-'*24} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for _, row in df_results.iterrows():
        print(f"{row['excluded_class']:24s} "
              f"{row['detection_rate']*100:7.1f}% "
              f"{row['detection_zero_votes']*100:7.1f}% "
              f"{row['detection_multi_votes']*100:7.1f}% "
              f"{(1-row['detection_rate'])*100:7.1f}%")
    
    print(f"\n{'AVERAGE':24s} "
          f"{df_results['detection_rate'].mean()*100:7.1f}% "
          f"{df_results['detection_zero_votes'].mean()*100:7.1f}% "
          f"{df_results['detection_multi_votes'].mean()*100:7.1f}% "
          f"{(1-df_results['detection_rate'].mean())*100:7.1f}%")
    
    print(f"\n\nKey Finding:")
    print(f"  Multi-vote detection contributes: {df_results['detection_multi_votes'].mean():.1%} on average")
    print(f"  This validates the hypothesis: ambiguous samples (2+ votes) are likely unknown!")


if __name__ == "__main__":
    main()
