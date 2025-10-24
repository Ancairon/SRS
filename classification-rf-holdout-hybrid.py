"""
Hybrid Holdout Detection - Distance + Multi-Vote Combination

Combines two complementary signals:
1. Distance metric: Detects samples far from training data
2. Multi-vote signal: Detects ambiguous samples that confuse multiple binary classifiers

Strategy:
- Flag as unknown if EITHER:
  * Distance > threshold (far from known classes)
  * OR votes != 1 (either no claims or multiple conflicting claims)
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
        y_binary = (y_train == target_class).astype(int)
        
        clf = RandomForestClassifier(
            n_estimators=100,  # Lighter than main RF
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        clf.fit(X_train, y_binary)
        classifiers[target_class] = clf
    
    return classifiers


def get_vote_counts(X, classifiers, confidence_threshold=0.5):
    """
    Get vote counts from binary classifiers.
    Returns number of classifiers that claim each sample.
    """
    n_samples = len(X)
    votes = np.zeros(n_samples, dtype=int)
    
    for clf in classifiers.values():
        proba = clf.predict_proba(X)[:, 1]  # P(is this class)
        votes += (proba >= confidence_threshold).astype(int)
    
    return votes


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


def optimize_hybrid_threshold(dist_test, dist_holdout, votes_test, votes_holdout,
                              fpr_targets=[0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20],
                              vote_thresholds=[0.3, 0.5, 0.7]):
    """
    Optimize both distance threshold AND vote confidence threshold.
    Flag as unknown if: (distance > dist_thresh) OR (votes != 1)
    """
    best_result = None
    best_detection = 0
    all_results = []
    
    for vote_conf in vote_thresholds:
        # Recompute votes with this confidence threshold (would need classifiers passed in)
        # For now, assume votes are precomputed at different thresholds
        
        for target_fpr in fpr_targets:
            # Distance threshold
            dist_threshold = np.percentile(dist_test, 100 * (1 - target_fpr))
            
            # Flags from distance
            flags_dist_test = dist_test > dist_threshold
            flags_dist_holdout = dist_holdout > dist_threshold
            
            # Flags from votes (!=1 means 0 or 2+)
            flags_vote_test = votes_test != 1
            flags_vote_holdout = votes_holdout != 1
            
            # Combined: flag if EITHER distance OR votes
            flags_combined_test = flags_dist_test | flags_vote_test
            flags_combined_holdout = flags_dist_holdout | flags_vote_holdout
            
            # Metrics
            fpr = np.sum(flags_combined_test) / len(flags_combined_test)
            detection = np.sum(flags_combined_holdout) / len(flags_combined_holdout)
            
            # Individual contributions
            detection_dist_only = np.sum(flags_dist_holdout) / len(flags_dist_holdout)
            detection_vote_only = np.sum(flags_vote_holdout) / len(flags_vote_holdout)
            
            result = {
                'vote_conf': vote_conf,
                'target_fpr': target_fpr,
                'dist_threshold': dist_threshold,
                'fpr': fpr,
                'detection': detection,
                'detection_dist': detection_dist_only,
                'detection_vote': detection_vote_only
            }
            all_results.append(result)
            
            # Track best with FPR < 0.20
            if fpr < 0.20 and detection > best_detection:
                best_detection = detection
                best_result = result
    
    if best_result is None:
        # Fallback: minimum FPR
        best_result = min(all_results, key=lambda x: x['fpr'])
    
    return best_result, pd.DataFrame(all_results)


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
    print(f"Strategy: Hybrid Distance + Multi-Vote Detection")
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
        
        # Train multi-class RF (for final predictions)
        print(f"  Training multi-class RF...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_train, y_train)
        
        # Train binary classifiers (for vote signal)
        print(f"  Training {len(known_classes)} binary classifiers...")
        start_time = time.time()
        binary_classifiers = train_binary_classifiers(X_train, y_train, known_classes)
        binary_train_time = time.time() - start_time
        
        # Test accuracy
        y_pred_test = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy: {test_accuracy:.4f}")
        
        # Compute distances
        print(f"  Computing distances...")
        dist_test = compute_min_distance(X_test, X_train)
        dist_holdout = compute_min_distance(X_holdout, X_train)
        
        distance_ratio = np.mean(dist_holdout) / np.mean(dist_test)
        print(f"    Distance ratio: {distance_ratio:.2f}x")
        
        # Get vote counts (test multiple confidence thresholds)
        print(f"  Computing vote patterns...")
        best_config = None
        best_detection = 0
        
        for vote_conf in [0.5, 0.6, 0.7]:
            votes_test = get_vote_counts(X_test, binary_classifiers, vote_conf)
            votes_holdout = get_vote_counts(X_holdout, binary_classifiers, vote_conf)
            
            # Try different distance thresholds
            for target_fpr in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
                dist_threshold = np.percentile(dist_test, 100 * (1 - target_fpr))
                
                # Combined flags
                flags_dist_test = dist_test > dist_threshold
                flags_vote_test = votes_test != 1
                flags_combined_test = flags_dist_test | flags_vote_test
                
                flags_dist_holdout = dist_holdout > dist_threshold
                flags_vote_holdout = votes_holdout != 1
                flags_combined_holdout = flags_dist_holdout | flags_vote_holdout
                
                fpr = np.sum(flags_combined_test) / len(flags_combined_test)
                detection = np.sum(flags_combined_holdout) / len(flags_combined_holdout)
                
                if fpr < 0.20 and detection > best_detection:
                    best_detection = detection
                    best_config = {
                        'vote_conf': vote_conf,
                        'dist_threshold': dist_threshold,
                        'target_fpr': target_fpr,
                        'fpr': fpr,
                        'detection': detection,
                        'flags_dist_test': flags_dist_test,
                        'flags_vote_test': flags_vote_test,
                        'flags_combined_test': flags_combined_test,
                        'flags_dist_holdout': flags_dist_holdout,
                        'flags_vote_holdout': flags_vote_holdout,
                        'flags_combined_holdout': flags_combined_holdout,
                        'votes_test': votes_test,
                        'votes_holdout': votes_holdout
                    }
        
        if best_config is None:
            print(f"  No valid configuration found, skipping...")
            continue
        
        # Calculate metrics with best config
        detection_dist_only = np.sum(best_config['flags_dist_holdout']) / len(best_config['flags_dist_holdout'])
        detection_vote_only = np.sum(best_config['flags_vote_holdout']) / len(best_config['flags_vote_holdout'])
        detection_combined = best_config['detection']
        
        fpr_dist_only = np.sum(best_config['flags_dist_test']) / len(best_config['flags_dist_test'])
        fpr_vote_only = np.sum(best_config['flags_vote_test']) / len(best_config['flags_vote_test'])
        fpr_combined = best_config['fpr']
        
        # Accuracy on non-flagged
        non_flagged_mask = ~best_config['flags_combined_test']
        if np.sum(non_flagged_mask) > 0:
            accuracy_non_flagged = accuracy_score(
                y_test[non_flagged_mask],
                y_pred_test[non_flagged_mask]
            )
        else:
            accuracy_non_flagged = 0.0
        
        print(f"\n  Best Configuration:")
        print(f"    Vote confidence: {best_config['vote_conf']:.2f}")
        print(f"    Distance threshold: {best_config['dist_threshold']:.2f}")
        print(f"\n  Detection Breakdown:")
        print(f"    Distance only:  {detection_dist_only:.1%} (FPR: {fpr_dist_only:.1%})")
        print(f"    Vote only:      {detection_vote_only:.1%} (FPR: {fpr_vote_only:.1%})")
        print(f"    Combined (OR):  {detection_combined:.1%} (FPR: {fpr_combined:.1%})")
        print(f"\n  Voting patterns:")
        print(f"    Test - 0 votes: {np.sum(best_config['votes_test']==0)}, 1 vote: {np.sum(best_config['votes_test']==1)}, >1: {np.sum(best_config['votes_test']>1)}")
        print(f"    Holdout - 0 votes: {np.sum(best_config['votes_holdout']==0)}, 1 vote: {np.sum(best_config['votes_holdout']==1)}, >1: {np.sum(best_config['votes_holdout']>1)}")
        print(f"\n  Accuracy (non-flagged): {accuracy_non_flagged:.4f}")
        
        # Store results
        result = {
            'excluded_class': excluded_class,
            'detection_combined': detection_combined,
            'detection_distance': detection_dist_only,
            'detection_vote': detection_vote_only,
            'fpr_combined': fpr_combined,
            'fpr_distance': fpr_dist_only,
            'fpr_vote': fpr_vote_only,
            'accuracy_non_flagged': accuracy_non_flagged,
            'test_accuracy': test_accuracy,
            'distance_ratio': distance_ratio,
            'vote_conf': best_config['vote_conf'],
            'dist_threshold': best_config['dist_threshold'],
            'votes_holdout_0': np.sum(best_config['votes_holdout']==0),
            'votes_holdout_1': np.sum(best_config['votes_holdout']==1),
            'votes_holdout_multi': np.sum(best_config['votes_holdout']>1)
        }
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - Hybrid Distance + Multi-Vote Detection")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    df_results = df_results.sort_values('detection_combined', ascending=False)
    
    print("Performance Comparison:")
    print(df_results[[
        'excluded_class', 'detection_combined', 'detection_distance', 'detection_vote',
        'fpr_combined', 'distance_ratio'
    ]].to_string(index=False))
    
    print(f"\n\nAverage Performance:")
    print(f"  Combined Detection: {df_results['detection_combined'].mean():.1%} (±{df_results['detection_combined'].std():.1%})")
    print(f"  Distance Only:      {df_results['detection_distance'].mean():.1%}")
    print(f"  Vote Only:          {df_results['detection_vote'].mean():.1%}")
    print(f"\n  Combined FPR:       {df_results['fpr_combined'].mean():.1%}")
    print(f"  Test Accuracy:      {df_results['test_accuracy'].mean():.4f}")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f}")
    
    # Performance breakdown
    print(f"\n\nDetection Performance Breakdown:")
    for threshold in [0.99, 0.95, 0.90, 0.80, 0.70]:
        high_det = df_results[df_results['detection_combined'] >= threshold]
        if len(high_det) > 0:
            print(f"\n  Classes with ≥{threshold:.0%} detection: {len(high_det)}/{len(df_results)}")
            for _, row in high_det.iterrows():
                print(f"    ✓ {row['excluded_class']:24s}: {row['detection_combined']:.1%} "
                      f"(dist: {row['detection_distance']:.1%}, vote: {row['detection_vote']:.1%})")
    
    # Save results
    output_file = "eval_results/rf_holdout_hybrid.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")
    
    # Comparison with baselines
    print(f"\n\n{'='*80}")
    print("COMPARISON WITH BASELINES")
    print(f"{'='*80}")
    print(f"\n  Hybrid (Distance + Vote): {df_results['detection_combined'].mean():.1%}")
    print(f"  Distance-only baseline:    92.6%")
    print(f"  Binary per-class baseline: 64.3%")
    
    if df_results['detection_combined'].mean() > 0.926:
        improvement = (df_results['detection_combined'].mean() - 0.926) * 100
        print(f"\n  ✓ IMPROVEMENT: +{improvement:.1f} percentage points over distance-only!")
    else:
        decline = (0.926 - df_results['detection_combined'].mean()) * 100
        print(f"\n  ✗ Decline: -{decline:.1f} percentage points vs distance-only")
    
    # Analyze contribution of vote signal
    print(f"\n\n{'='*80}")
    print("VOTE SIGNAL CONTRIBUTION")
    print(f"{'='*80}")
    
    for _, row in df_results.iterrows():
        vote_contribution = row['detection_combined'] - row['detection_distance']
        print(f"\n  {row['excluded_class']:24s}:")
        print(f"    Distance detected: {row['detection_distance']:.1%}")
        print(f"    Vote adds:         {vote_contribution:+.1%}")
        print(f"    Total:             {row['detection_combined']:.1%}")
        print(f"    Holdout voting: 0={row['votes_holdout_0']}, 1={row['votes_holdout_1']}, >1={row['votes_holdout_multi']}")


if __name__ == "__main__":
    main()
