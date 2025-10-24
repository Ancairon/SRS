import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import LocalOutlierFactor
from scipy.spatial.distance import cdist
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

def load_time_split_sequences(base_dir, seq_len, step, train_frac, excluded_class=None):
    """
    Load sequences with time-based train/test split.
    If excluded_class is provided, it will only be in test set.
    """
    X_tr, y_tr, X_te, y_te = [], [], [], []
    X_holdout, y_holdout = [], []

    for label in sorted(os.listdir(base_dir)):
        label_dir = os.path.join(base_dir, label)
        csv_path = os.path.join(label_dir, f"{label}.csv")
        if not os.path.isfile(csv_path) or "idle" in label:
            continue

        data = pd.read_csv(csv_path).drop(columns="Target")
        n_rows = data.shape[0]
        split = int(n_rows * train_frac)

        # If this is the excluded class, only add to holdout set
        if label == excluded_class:
            for start in range(0, n_rows - seq_len + 1, step):
                X_holdout.append(data[start:start+seq_len].to_numpy().flatten())
                y_holdout.append(label)
            continue

        # Regular train/test split for known classes
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

def compute_confidence_features(rf, X, y_pred):
    """
    Compute multiple confidence metrics for predictions.
    """
    # Get prediction probabilities
    pred_proba = rf.predict_proba(X)
    
    # Max probability (basic confidence)
    max_proba = np.max(pred_proba, axis=1)
    
    # Entropy (uncertainty measure)
    epsilon = 1e-10
    entropy = -np.sum(pred_proba * np.log(pred_proba + epsilon), axis=1)
    
    # Margin (difference between top two probabilities)
    sorted_proba = np.sort(pred_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    
    return {
        'max_proba': max_proba,
        'entropy': entropy,
        'margin': margin,
        'pred_proba': pred_proba
    }

def compute_distance_to_training(X_test, X_train, metric='euclidean'):
    """
    Compute minimum distance from each test sample to training samples.
    """
    # For efficiency, sample training data if too large
    if len(X_train) > 5000:
        indices = np.random.choice(len(X_train), 5000, replace=False)
        X_train_sample = X_train[indices]
    else:
        X_train_sample = X_train
    
    # Compute distances in batches to avoid memory issues
    batch_size = 1000
    min_distances = []
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        distances = cdist(batch, X_train_sample, metric=metric)
        min_distances.extend(np.min(distances, axis=1))
    
    return np.array(min_distances)

def flag_uncertain_predictions(confidence_features, min_distances, 
                               proba_threshold=0.5, 
                               entropy_threshold=None,
                               margin_threshold=0.2,
                               distance_threshold=None):
    """
    Flag predictions as uncertain based on multiple criteria.
    """
    flags = np.zeros(len(confidence_features['max_proba']), dtype=bool)
    
    # Low confidence (max probability)
    flags |= confidence_features['max_proba'] < proba_threshold
    
    # High entropy (if threshold provided)
    if entropy_threshold is not None:
        flags |= confidence_features['entropy'] > entropy_threshold
    
    # Low margin
    flags |= confidence_features['margin'] < margin_threshold
    
    # Large distance to training data (if threshold provided)
    if distance_threshold is not None:
        flags |= min_distances > distance_threshold
    
    return flags

def compute_adaptive_thresholds(confidence_features, min_distances, percentile=90):
    """
    Compute adaptive thresholds based on training data statistics.
    """
    thresholds = {
        'proba': np.percentile(confidence_features['max_proba'], 100 - percentile),
        'entropy': np.percentile(confidence_features['entropy'], percentile),
        'margin': np.percentile(confidence_features['margin'], 100 - percentile),
        'distance': np.percentile(min_distances, percentile)
    }
    return thresholds

def evaluate_holdout_detection(rf, X_train, X_test_known, y_test_known, 
                               X_holdout, y_holdout, excluded_class):
    """
    Evaluate how well the model can detect the held-out class as uncertain.
    """
    results = {}
    
    # Predict on known test samples
    y_pred_known = rf.predict(X_test_known)
    conf_known = compute_confidence_features(rf, X_test_known, y_pred_known)
    dist_known = compute_distance_to_training(X_test_known, X_train)
    
    # Compute adaptive thresholds from known test data (use median/percentiles)
    # Use lower percentile for probability/margin (want to flag low values)
    # Use higher percentile for distance (want to flag high values)
    proba_threshold = max(0.6, np.percentile(conf_known['max_proba'], 5))
    margin_threshold = max(0.3, np.percentile(conf_known['margin'], 5))
    distance_threshold = np.percentile(dist_known, 95)  # Top 5% of distances
    
    thresholds = {
        'proba': proba_threshold,
        'entropy': None,  # Not using entropy for now
        'margin': margin_threshold,
        'distance': distance_threshold
    }
    
    print(f"\n  Adaptive thresholds:")
    print(f"    Probability (5th percentile, min 0.6): {thresholds['proba']:.3f}")
    print(f"    Margin (5th percentile, min 0.3): {thresholds['margin']:.3f}")
    print(f"    Distance (95th percentile): {thresholds['distance']:.3f}")
    
    # Predict on holdout (unknown) samples
    y_pred_holdout = rf.predict(X_holdout)
    conf_holdout = compute_confidence_features(rf, X_holdout, y_pred_holdout)
    dist_holdout = compute_distance_to_training(X_holdout, X_train)
    
    # Flag uncertain predictions
    flags_known = flag_uncertain_predictions(
        conf_known, dist_known,
        proba_threshold=thresholds['proba'],
        entropy_threshold=None,
        margin_threshold=thresholds['margin'],
        distance_threshold=thresholds['distance']
    )
    
    flags_holdout = flag_uncertain_predictions(
        conf_holdout, dist_holdout,
        proba_threshold=thresholds['proba'],
        entropy_threshold=None,
        margin_threshold=thresholds['margin'],
        distance_threshold=thresholds['distance']
    )
    
    # Calculate metrics
    known_flagged = np.sum(flags_known)
    known_total = len(flags_known)
    holdout_flagged = np.sum(flags_holdout)
    holdout_total = len(flags_holdout)
    
    # Detection rate: how many holdout samples were correctly flagged
    detection_rate = holdout_flagged / holdout_total if holdout_total > 0 else 0
    
    # False positive rate: how many known samples were incorrectly flagged
    fpr = known_flagged / known_total if known_total > 0 else 0
    
    # Accuracy on non-flagged known samples
    non_flagged_mask = ~flags_known
    if np.sum(non_flagged_mask) > 0:
        accuracy_non_flagged = accuracy_score(
            y_test_known[non_flagged_mask],
            y_pred_known[non_flagged_mask]
        )
    else:
        accuracy_non_flagged = 0.0
    
    results = {
        'excluded_class': excluded_class,
        'known_samples': known_total,
        'known_flagged': known_flagged,
        'known_flagged_rate': fpr,
        'holdout_samples': holdout_total,
        'holdout_flagged': holdout_flagged,
        'detection_rate': detection_rate,
        'accuracy_non_flagged': accuracy_non_flagged,
        'thresholds': thresholds,
        'avg_proba_known': np.mean(conf_known['max_proba']),
        'avg_proba_holdout': np.mean(conf_holdout['max_proba']),
        'avg_distance_known': np.mean(dist_known),
        'avg_distance_holdout': np.mean(dist_holdout),
    }
    
    return results

def main():
    base_dir   = "custom_datasets"
    seq_len    = 3
    step       = 3
    train_frac = 0.7

    # Get all available classes (excluding idle)
    all_classes = [
        d for d in sorted(os.listdir(base_dir))
        if os.path.isdir(os.path.join(base_dir, d)) and "idle" not in d.lower()
    ]
    
    print(f"Found {len(all_classes)} classes: {all_classes}\n")
    print("=" * 80)
    
    # Store results for all experiments
    all_results = []
    
    # Iterate through each class, holding it out
    for excluded_class in all_classes:
        print(f"\n{'='*80}")
        print(f"Experiment: Holding out class '{excluded_class}'")
        print(f"{'='*80}")
        
        # Load data with one class held out
        X_train, y_train, X_test, y_test, X_holdout, y_holdout = load_time_split_sequences(
            base_dir, seq_len, step, train_frac, excluded_class=excluded_class
        )
        
        if len(X_train) == 0 or len(X_holdout) == 0:
            print(f"  Skipping {excluded_class}: insufficient data")
            continue
        
        print(f"\n  Train samples: {len(y_train)} (from {len(np.unique(y_train))} classes)")
        print(f"  Test samples (known): {len(y_test)}")
        print(f"  Holdout samples ({excluded_class}): {len(y_holdout)}")
        
        # Train Random Forest on known classes only
        rf = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1
        )
        
        start_train = time.time()
        rf.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        print(f"\n  Training time: {train_time:.3f} seconds")
        
        # Evaluate on known test set
        start_infer = time.time()
        y_pred_test = rf.predict(X_test)
        infer_time = time.time() - start_infer
        
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy (known classes): {test_accuracy:.4f}")
        print(f"  Inference time: {infer_time:.3f} seconds")
        
        # Evaluate holdout detection
        results = evaluate_holdout_detection(
            rf, X_train, X_test, y_test, X_holdout, y_holdout, excluded_class
        )
        results['train_time'] = train_time
        results['test_accuracy'] = test_accuracy
        results['n_known_classes'] = len(np.unique(y_train))
        
        print(f"\n  Holdout Detection Results:")
        print(f"    Detection rate (holdout flagged): {results['detection_rate']:.1%}")
        print(f"    False positive rate (known flagged): {results['known_flagged_rate']:.1%}")
        print(f"    Accuracy on non-flagged known: {results['accuracy_non_flagged']:.4f}")
        print(f"    Avg confidence - Known: {results['avg_proba_known']:.3f}, Holdout: {results['avg_proba_holdout']:.3f}")
        print(f"    Avg distance - Known: {results['avg_distance_known']:.3f}, Holdout: {results['avg_distance_holdout']:.3f}")
        
        all_results.append(results)
    
    # Summary across all experiments
    print(f"\n\n{'='*80}")
    print("SUMMARY: Holdout Detection Performance by Class")
    print(f"{'='*80}\n")
    
    df_results = pd.DataFrame(all_results)
    
    # Sort by detection rate
    df_results = df_results.sort_values('detection_rate', ascending=False)
    
    print(df_results[[
        'excluded_class', 
        'detection_rate', 
        'known_flagged_rate',
        'accuracy_non_flagged',
        'test_accuracy',
        'holdout_samples'
    ]].to_string(index=False))
    
    print(f"\n\nAverage metrics across all classes:")
    print(f"  Detection rate: {df_results['detection_rate'].mean():.1%} (±{df_results['detection_rate'].std():.1%})")
    print(f"  False positive rate: {df_results['known_flagged_rate'].mean():.1%} (±{df_results['known_flagged_rate'].std():.1%})")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f} (±{df_results['accuracy_non_flagged'].std():.4f})")
    
    # Identify best and worst classes for holdout
    print(f"\n\nBest classes for holdout detection (easiest to detect as unknown):")
    top_3 = df_results.nlargest(3, 'detection_rate')
    for idx, row in top_3.iterrows():
        print(f"  {row['excluded_class']}: {row['detection_rate']:.1%} detection rate")
    
    print(f"\nWorst classes for holdout detection (hardest to detect as unknown):")
    bottom_3 = df_results.nsmallest(3, 'detection_rate')
    for idx, row in bottom_3.iterrows():
        print(f"  {row['excluded_class']}: {row['detection_rate']:.1%} detection rate")
    
    # Save results
    output_file = "eval_results/rf_holdout_evaluation_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
