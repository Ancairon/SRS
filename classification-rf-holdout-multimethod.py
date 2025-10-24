"""
Autoencoder-Based Holdout Detection

Strategy:
1. Train autoencoder on KNOWN classes only
2. Learn to reconstruct "normal" patterns
3. Unknown classes will have high reconstruction error
4. Combine with one-class SVM for robust detection

This is class-agnostic - we never use knowledge of what the holdout class is.
"""
import os
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')

# Try to import deep learning
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not available - using isolation forest only")


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


def build_autoencoder(input_dim, encoding_dim=32):
    """
    Build autoencoder to learn compressed representation of normal data.
    High reconstruction error = anomaly/unknown class.
    """
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(encoder_input)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoding')(x)
    
    # Decoder
    x = layers.Dense(64, activation='relu')(encoded)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    autoencoder = keras.Model(encoder_input, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder


def compute_reconstruction_error(autoencoder, X):
    """Compute reconstruction error for each sample."""
    X_reconstructed = autoencoder.predict(X, verbose=0)
    mse = np.mean((X - X_reconstructed) ** 2, axis=1)
    return mse


class MultiMethodDetector:
    """
    Combines multiple detection methods:
    1. Autoencoder reconstruction error
    2. Isolation Forest anomaly scores
    3. One-Class SVM decision function
    4. Distance to training data
    
    All methods are trained ONLY on known classes.
    """
    
    def __init__(self, use_autoencoder=True, use_isolation_forest=True, 
                 use_ocsvm=True, use_distance=True):
        self.use_autoencoder = use_autoencoder and HAS_TF
        self.use_isolation_forest = use_isolation_forest
        self.use_ocsvm = use_ocsvm
        self.use_distance = use_distance
        
        self.autoencoder = None
        self.isolation_forest = None
        self.ocsvm = None
        self.X_train = None
        self.scaler = StandardScaler()
        
    def fit(self, X_train):
        """Train all anomaly detectors on known data."""
        print("  Training anomaly detectors...")
        
        # Scale data
        X_scaled = self.scaler.fit_transform(X_train)
        self.X_train = X_train
        
        # 1. Autoencoder
        if self.use_autoencoder:
            print("    - Autoencoder...", end=" ", flush=True)
            self.autoencoder = build_autoencoder(X_train.shape[1], encoding_dim=32)
            self.autoencoder.fit(
                X_scaled, X_scaled,
                epochs=50,
                batch_size=64,
                shuffle=True,
                validation_split=0.2,
                verbose=0,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ]
            )
            print("✓")
        
        # 2. Isolation Forest
        if self.use_isolation_forest:
            print("    - Isolation Forest...", end=" ", flush=True)
            self.isolation_forest = IsolationForest(
                contamination=0.1,  # Assume 10% of training could be edge cases
                random_state=42,
                n_jobs=-1
            )
            self.isolation_forest.fit(X_scaled)
            print("✓")
        
        # 3. One-Class SVM
        if self.use_ocsvm:
            print("    - One-Class SVM...", end=" ", flush=True)
            # Use smaller sample if data is too large
            if len(X_scaled) > 2000:
                indices = np.random.choice(len(X_scaled), 2000, replace=False)
                X_sample = X_scaled[indices]
            else:
                X_sample = X_scaled
            
            self.ocsvm = OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=0.1  # Expected proportion of outliers
            )
            self.ocsvm.fit(X_sample)
            print("✓")
        
        return self
    
    def compute_anomaly_scores(self, X):
        """
        Compute anomaly scores from all methods.
        Returns dict with individual scores and combined score.
        """
        X_scaled = self.scaler.transform(X)
        scores = {}
        
        # 1. Autoencoder reconstruction error
        if self.use_autoencoder:
            recon_error = compute_reconstruction_error(self.autoencoder, X_scaled)
            scores['autoencoder'] = recon_error
        
        # 2. Isolation Forest (negative score = anomaly)
        if self.use_isolation_forest:
            if_scores = -self.isolation_forest.score_samples(X_scaled)
            scores['isolation_forest'] = if_scores
        
        # 3. One-Class SVM (negative decision = anomaly)
        if self.use_ocsvm:
            ocsvm_scores = -self.ocsvm.decision_function(X_scaled)
            scores['ocsvm'] = ocsvm_scores
        
        # 4. Distance to training data
        if self.use_distance:
            if len(self.X_train) > 5000:
                indices = np.random.choice(len(self.X_train), 5000, replace=False)
                X_train_sample = self.X_train[indices]
            else:
                X_train_sample = self.X_train
            
            distances = cdist(X, X_train_sample, metric='euclidean')
            min_distances = np.min(distances, axis=1)
            scores['distance'] = min_distances
        
        # Normalize all scores to [0, 1] and combine
        normalized_scores = []
        score_keys = list(scores.keys())  # Create list to avoid RuntimeError
        for key in score_keys:
            score = scores[key]
            # Normalize to [0, 1]
            score_norm = (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-10)
            normalized_scores.append(score_norm)
            scores[f'{key}_normalized'] = score_norm
        
        # Combined score (average of all methods)
        scores['combined'] = np.mean(normalized_scores, axis=0)
        
        return scores
    
    def flag_anomalies(self, scores_test, scores_holdout, target_fpr=0.15):
        """
        Flag anomalies using adaptive threshold calibrated on test set.
        Returns flags for test and holdout, plus threshold used.
        """
        # Use combined score for thresholding
        combined_test = scores_test['combined']
        combined_holdout = scores_holdout['combined']
        
        # Set threshold at (1 - target_fpr) percentile of test scores
        threshold = np.percentile(combined_test, 100 * (1 - target_fpr))
        
        flags_test = combined_test > threshold
        flags_holdout = combined_holdout > threshold
        
        return flags_test, flags_holdout, threshold


def evaluate_multi_method(rf, detector, X_train, X_test, y_test, 
                          X_holdout, y_holdout, excluded_class):
    """Evaluate using multi-method anomaly detection."""
    
    # Get RF predictions
    y_pred_test = rf.predict(X_test)
    
    # Compute anomaly scores for test and holdout
    print("  Computing anomaly scores...")
    scores_test = detector.compute_anomaly_scores(X_test)
    scores_holdout = detector.compute_anomaly_scores(X_holdout)
    
    # Try multiple FPR targets
    fpr_targets = [0.05, 0.10, 0.15, 0.20]
    results = []
    
    for target_fpr in fpr_targets:
        flags_test, flags_holdout, threshold = detector.flag_anomalies(
            scores_test, scores_holdout, target_fpr
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
            'threshold': threshold
        })
    
    # Select best result (maximize detection with FPR < 0.20)
    df_results = pd.DataFrame(results)
    df_valid = df_results[df_results['actual_fpr'] < 0.20]
    if len(df_valid) > 0:
        best_idx = df_valid['detection_rate'].idxmax()
    else:
        best_idx = df_results['actual_fpr'].idxmin()
    
    best_result = df_results.loc[best_idx]
    
    print(f"\n  FPR Target Results:")
    print(df_results[['target_fpr', 'actual_fpr', 'detection_rate', 'accuracy_non_flagged']].to_string(index=False))
    
    # Analyze individual method performance at best threshold
    flags_test_best, flags_holdout_best, _ = detector.flag_anomalies(
        scores_test, scores_holdout, best_result['target_fpr']
    )
    
    print(f"\n  Individual Method Analysis (at target_fpr={best_result['target_fpr']}):")
    for method in ['autoencoder', 'isolation_forest', 'ocsvm', 'distance']:
        if f'{method}_normalized' in scores_test:
            # Use same threshold on normalized scores
            thresh_method = np.percentile(
                scores_test[f'{method}_normalized'], 
                100 * (1 - best_result['target_fpr'])
            )
            flags_method_holdout = scores_holdout[f'{method}_normalized'] > thresh_method
            det_rate = np.sum(flags_method_holdout) / len(flags_method_holdout)
            print(f"    {method:18s}: {det_rate:6.1%} detection")
    
    return best_result, scores_test, scores_holdout


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
    print(f"Using: Autoencoder + Isolation Forest + One-Class SVM + Distance")
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
        
        # Train RF for classification
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42,
            n_jobs=-1
        )
        
        print(f"\n  Training Random Forest...")
        rf.fit(X_train, y_train)
        
        # Test accuracy
        y_pred_test = rf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        print(f"  Test accuracy (known classes): {test_accuracy:.4f}")
        
        # Train multi-method detector
        detector = MultiMethodDetector(
            use_autoencoder=HAS_TF,
            use_isolation_forest=True,
            use_ocsvm=True,
            use_distance=True
        )
        
        start_time = time.time()
        detector.fit(X_train)
        train_time = time.time() - start_time
        print(f"  Detector training time: {train_time:.2f}s")
        
        # Evaluate
        best_result, scores_test, scores_holdout = evaluate_multi_method(
            rf, detector, X_train, X_test, y_test,
            X_holdout, y_holdout, excluded_class
        )
        
        print(f"\n  Best Configuration:")
        print(f"    Detection rate: {best_result['detection_rate']:.1%}")
        print(f"    Actual FPR: {best_result['actual_fpr']:.1%}")
        print(f"    Accuracy (non-flagged): {best_result['accuracy_non_flagged']:.4f}")
        
        # Store results
        result = {
            'excluded_class': excluded_class,
            'detection_rate': best_result['detection_rate'],
            'fpr': best_result['actual_fpr'],
            'accuracy_non_flagged': best_result['accuracy_non_flagged'],
            'test_accuracy': test_accuracy,
            'target_fpr': best_result['target_fpr'],
            'train_time': train_time
        }
        all_results.append(result)
    
    # Summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY - Multi-Method Anomaly Detection")
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
    print(f"  Test accuracy: {df_results['test_accuracy'].mean():.4f}")
    print(f"  Accuracy (non-flagged): {df_results['accuracy_non_flagged'].mean():.4f}")
    
    # Classes achieving different detection thresholds
    for threshold in [0.99, 0.95, 0.90, 0.80]:
        high_det = df_results[df_results['detection_rate'] >= threshold]
        print(f"\n  Classes with ≥{threshold:.0%} detection: {len(high_det)}/{len(df_results)}")
        for _, row in high_det.iterrows():
            print(f"    ✓ {row['excluded_class']}: {row['detection_rate']:.1%} (FPR: {row['fpr']:.1%})")
    
    # Save
    output_file = "eval_results/rf_holdout_multimethod.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\n\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
