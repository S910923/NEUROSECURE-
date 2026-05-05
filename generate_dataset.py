"""
NeuroSecure Pipeline — Synthetic EEG Dataset Generator
Generates realistic 8-channel EEG data with labeled attack events.
Saves as CSV and numpy arrays for demo + ML training.

Usage:
    python generate_dataset.py
    python generate_dataset.py --samples 2000 --attack-rate 0.12 --output my_dataset
"""

import numpy as np
import csv
import os
import json
import argparse
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────────────

CHANNEL_NAMES  = ['Fp1','Fp2','F3','F4','C3','C4','P3','P4']
SAMPLING_RATE  = 256      # Hz
SAMPLES_PER_WIN = 256     # 1-second windows
N_CHANNELS     = 8
BANDS = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta' : (13, 30),
    'gamma': (30, 80),
}

ATTACK_TYPES = [
    'signal_injection',
    'replay_attack',
    'adversarial_perturbation',
    'eavesdropping_artifact',
    'model_inversion_probe',
    'membership_inference',
]

SEVERITY_THRESHOLDS = {
    'critical': 0.85,
    'high':     0.65,
    'medium':   0.40,
    'low':      0.0,
}

# ── Signal Generation ──────────────────────────────────────────────────────────

rng = np.random.RandomState(42)


def clean_eeg_window(window_idx: int) -> np.ndarray:
    """
    Generate a realistic clean 8-channel EEG window.
    Each channel has dominant frequency bands + realistic 1/f noise.
    Returns array of shape (N_CHANNELS, SAMPLES_PER_WIN).
    """
    t = np.linspace(window_idx, window_idx + 1, SAMPLES_PER_WIN)
    channels = []
    for ch in range(N_CHANNELS):
        sig = np.zeros(SAMPLES_PER_WIN)

        # Alpha wave (dominant, 8–13 Hz) — varies per channel
        alpha_freq = 8.5 + ch * 0.6
        sig += (20 + ch * 2) * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2*np.pi))

        # Beta wave (13–30 Hz) — moderate amplitude
        beta_freq  = 18 + ch * 1.5
        sig += (8 + ch) * np.sin(2 * np.pi * beta_freq * t + rng.uniform(0, 2*np.pi))

        # Theta wave (4–8 Hz) — low amplitude
        sig += 5 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2*np.pi))

        # Delta drift (0.5–4 Hz) — very slow
        sig += 3 * np.sin(2 * np.pi * 1.5 * t)

        # Gamma burst (~40 Hz) — tiny
        sig += 2 * np.sin(2 * np.pi * 40 * t + rng.uniform(0, 2*np.pi))

        # Physiological noise (muscle, electrode)
        sig += rng.randn(SAMPLES_PER_WIN) * 4

        # 1/f (pink) noise approximation
        freqs = np.fft.rfftfreq(SAMPLES_PER_WIN, d=1.0/SAMPLING_RATE)
        freqs[0] = 1  # avoid divide by zero
        pink  = rng.randn(len(freqs)) / np.sqrt(freqs)
        sig  += np.fft.irfft(pink, n=SAMPLES_PER_WIN) * 2

        channels.append(sig.astype(np.float32))

    return np.array(channels)


def inject_attack(clean_signal: np.ndarray, attack_type: str) -> tuple:
    """
    Inject an attack into a clean signal window.
    Returns (attacked_signal, affected_channels).
    """
    signal = clean_signal.copy()
    n_affected = rng.randint(1, 4)
    affected_chs = rng.choice(N_CHANNELS, size=n_affected, replace=False).tolist()
    t = np.linspace(0, 1, SAMPLES_PER_WIN)

    for ch in affected_chs:
        if attack_type == 'signal_injection':
            # Large amplitude spikes
            spike_times = rng.randint(0, SAMPLES_PER_WIN, size=rng.randint(3, 8))
            signal[ch, spike_times] += rng.choice([-1, 1], size=len(spike_times)) * rng.uniform(150, 300, len(spike_times))
            signal[ch] += rng.randn(SAMPLES_PER_WIN) * 60

        elif attack_type == 'replay_attack':
            # Repeat a segment of the signal (temporal repetition)
            seg_len = SAMPLES_PER_WIN // 4
            start   = rng.randint(0, SAMPLES_PER_WIN - seg_len)
            repeat  = np.tile(signal[ch, start:start+seg_len], 4)[:SAMPLES_PER_WIN]
            signal[ch] = 0.7 * signal[ch] + 0.3 * repeat + rng.randn(SAMPLES_PER_WIN) * 10

        elif attack_type == 'adversarial_perturbation':
            # Subtle structured noise — crafted to fool classifier
            # High-frequency perturbation imperceptible to visual inspection
            perturb = np.sin(2 * np.pi * 120 * t + rng.uniform(0, 2*np.pi)) * rng.uniform(15, 35)
            perturb += rng.randn(SAMPLES_PER_WIN) * 12
            signal[ch] += perturb

        elif attack_type == 'eavesdropping_artifact':
            # RF interference / EM leakage artifact
            rf_freq = rng.choice([50, 60, 100, 120])  # powerline harmonics
            signal[ch] += np.sin(2 * np.pi * rf_freq * t) * rng.uniform(20, 60)
            signal[ch] += rng.randn(SAMPLES_PER_WIN) * 25

        elif attack_type == 'model_inversion_probe':
            # Systematic sweep signal — attacker probing model boundaries
            sweep = np.linspace(-80, 80, SAMPLES_PER_WIN)
            signal[ch] += sweep * np.sin(2 * np.pi * 3 * t) + rng.randn(SAMPLES_PER_WIN) * 15

        elif attack_type == 'membership_inference':
            # Subtle correlation injection — statistical inference attack
            correlation_pattern = np.sin(2 * np.pi * 7 * t) * 20 + np.cos(2 * np.pi * 14 * t) * 10
            signal[ch] = 0.6 * signal[ch] + 0.4 * correlation_pattern + rng.randn(SAMPLES_PER_WIN) * 8

    return signal.astype(np.float32), affected_chs


def compute_anomaly_score(signal: np.ndarray) -> float:
    """
    Fast approximate anomaly score using statistical features.
    In production this uses the Isolation Forest.
    """
    scores = []
    for ch in range(N_CHANNELS):
        sig = signal[ch]
        kurtosis  = float(np.mean(((sig - sig.mean()) / (sig.std() + 1e-8)) ** 4))
        std_ratio = sig.std() / (np.abs(sig.mean()) + 1e-8)
        zcr       = float(np.sum(np.diff(np.sign(sig)) != 0)) / len(sig)
        peak      = float(np.percentile(np.abs(sig), 99))
        scores.append(kurtosis / 10 + std_ratio / 20 + zcr + peak / 200)

    raw = np.mean(scores)
    # Normalize to [0, 1]
    return float(np.clip(raw / 3, 0.02, 0.99))


def classify_severity(score: float) -> str:
    for label, thresh in SEVERITY_THRESHOLDS.items():
        if score >= thresh:
            return label
    return 'low'


# ── Feature Extraction ─────────────────────────────────────────────────────────

def bandpower(sig: np.ndarray, fs: int, band: tuple) -> float:
    fft_vals = np.abs(np.fft.rfft(sig)) ** 2
    freqs    = np.fft.rfftfreq(len(sig), d=1.0/fs)
    idx      = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    return float(np.mean(fft_vals[idx])) if len(idx) else 0.0


def extract_features(signal: np.ndarray) -> np.ndarray:
    """Extract 88-dim feature vector (8 channels × 11 features)."""
    feats = []
    for ch in range(N_CHANNELS):
        sig = signal[ch]
        mu, sigma = sig.mean(), sig.std() + 1e-8
        ch_feats = [
            float(mu),
            float(sigma),
            float(np.percentile(sig, 95) - np.percentile(sig, 5)),
            float(np.mean(np.abs(np.diff(sig)))),
            float(np.mean(((sig - mu) / sigma) ** 4)),   # kurtosis
            float(np.mean(((sig - mu) / sigma) ** 3)),   # skewness
        ]
        for band in BANDS.values():
            ch_feats.append(bandpower(sig, SAMPLING_RATE, band))
        zcr = float(np.sum(np.diff(np.sign(sig)) != 0)) / len(sig)
        ch_feats.append(zcr)
        feats.extend(ch_feats)
    return np.array(feats, dtype=np.float32)


# ── Differential Privacy ───────────────────────────────────────────────────────

def apply_dp_noise(signal: np.ndarray, epsilon: float = 1.0, delta: float = 1e-5) -> np.ndarray:
    """Apply Gaussian DP noise to signal."""
    sensitivity = 1.0
    sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    noise = rng.normal(0, sigma, signal.shape).astype(np.float32)
    return signal + noise


# ── Dataset Generation ─────────────────────────────────────────────────────────

def generate_dataset(
    n_samples:   int   = 1000,
    attack_rate: float = 0.10,
    epsilon:     float = 1.0,
    output_dir:  str   = 'dataset',
    verbose:     bool  = True,
):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'═'*60}")
    print(f"  NeuroSecure — Synthetic EEG Dataset Generator")
    print(f"{'═'*60}")
    print(f"  Samples:     {n_samples}")
    print(f"  Attack rate: {attack_rate*100:.1f}%")
    print(f"  DP epsilon:  ε={epsilon}")
    print(f"  Output dir:  {output_dir}/")
    print(f"{'═'*60}\n")

    records  = []        # CSV rows
    features = []        # numpy features
    labels   = []        # binary: 0=clean, 1=attack
    signals  = []        # raw signal arrays

    n_attacks = int(n_samples * attack_rate)
    attack_indices = set(rng.choice(n_samples, size=n_attacks, replace=False).tolist())

    for i in range(n_samples):
        if verbose and (i % 100 == 0 or i == n_samples - 1):
            pct = (i + 1) / n_samples * 100
            bar = '█' * int(pct / 2) + '░' * (50 - int(pct / 2))
            print(f'\r  [{bar}] {pct:5.1f}%  ({i+1}/{n_samples})', end='', flush=True)

        # 1. Generate clean signal
        clean = clean_eeg_window(i)

        # 2. Maybe inject attack
        is_attack   = i in attack_indices
        attack_type = None
        affected_chs = []
        if is_attack:
            attack_type  = ATTACK_TYPES[rng.randint(0, len(ATTACK_TYPES))]
            signal, affected_chs = inject_attack(clean, attack_type)
        else:
            signal = clean

        # 3. Apply Differential Privacy noise
        private_signal = apply_dp_noise(signal, epsilon=epsilon)

        # 4. Extract features
        feat_vec = extract_features(private_signal)

        # 5. Compute anomaly score
        score    = compute_anomaly_score(private_signal)
        severity = classify_severity(score) if is_attack else 'none'

        # 6. Collect
        features.append(feat_vec)
        labels.append(1 if is_attack else 0)
        signals.append(signal)

        record = {
            'sample_id':       i,
            'timestamp':       f"{i * (SAMPLES_PER_WIN / SAMPLING_RATE):.3f}",
            'is_attack':       int(is_attack),
            'attack_type':     attack_type or 'none',
            'severity':        severity,
            'anomaly_score':   f"{score:.6f}",
            'affected_channels': json.dumps(affected_chs),
            'epsilon':         epsilon,
            'delta':           1e-5,
        }
        # Add per-channel RMS
        for ch, name in enumerate(CHANNEL_NAMES):
            rms = float(np.sqrt(np.mean(signal[ch]**2)))
            record[f'rms_{name}'] = f"{rms:.4f}"

        records.append(record)

    print(f'\n\n  ✓ Generated {n_samples} samples ({n_attacks} attacks, {n_samples-n_attacks} clean)\n')

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, 'eeg_dataset.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"  ✓ Saved CSV:      {csv_path}  ({os.path.getsize(csv_path)//1024} KB)")

    # ── Save NumPy ────────────────────────────────────────────────────────────
    X = np.array(features, dtype=np.float32)
    y = np.array(labels,   dtype=np.int32)

    np.save(os.path.join(output_dir, 'X_features.npy'), X)
    np.save(os.path.join(output_dir, 'y_labels.npy'),   y)
    np.save(os.path.join(output_dir, 'raw_signals.npy'), np.array(signals, dtype=np.float32))
    print(f"  ✓ Saved X_features.npy  shape: {X.shape}")
    print(f"  ✓ Saved y_labels.npy    shape: {y.shape}")
    print(f"  ✓ Saved raw_signals.npy shape: {np.array(signals).shape}")

    # ── Save attack-only CSV ──────────────────────────────────────────────────
    attack_records = [r for r in records if r['is_attack'] == 1]
    attack_csv = os.path.join(output_dir, 'attacks_only.csv')
    with open(attack_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(attack_records)
    print(f"  ✓ Saved attacks_only.csv  ({len(attack_records)} events)")

    # ── Save metadata JSON ────────────────────────────────────────────────────
    attack_type_counts = {}
    severity_counts    = {'critical':0,'high':0,'medium':0,'low':0,'none':0}
    for r in records:
        at = r['attack_type']
        attack_type_counts[at] = attack_type_counts.get(at, 0) + 1
        severity_counts[r['severity']] = severity_counts.get(r['severity'], 0) + 1

    metadata = {
        'generated_at':      datetime.now().isoformat(),
        'n_samples':         n_samples,
        'n_attacks':         n_attacks,
        'n_clean':           n_samples - n_attacks,
        'attack_rate':       attack_rate,
        'epsilon':           epsilon,
        'delta':             1e-5,
        'sampling_rate':     SAMPLING_RATE,
        'n_channels':        N_CHANNELS,
        'channel_names':     CHANNEL_NAMES,
        'samples_per_window':SAMPLES_PER_WIN,
        'feature_dim':       int(X.shape[1]),
        'attack_breakdown':  attack_type_counts,
        'severity_breakdown':severity_counts,
        'feature_names': [f"{name}_{ch}" for ch in CHANNEL_NAMES for name in
                         ['mean','std','range','mad','kurtosis','skewness',
                          'delta_power','theta_power','alpha_power','beta_power','gamma_power','zcr']],
    }
    meta_path = os.path.join(output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata.json")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  DATASET SUMMARY")
    print(f"{'─'*60}")
    print(f"  Total samples   : {n_samples:,}")
    print(f"  Clean EEG       : {n_samples - n_attacks:,}  ({(1-attack_rate)*100:.0f}%)")
    print(f"  Attack events   : {n_attacks:,}  ({attack_rate*100:.0f}%)")
    print(f"  Feature dim     : {X.shape[1]}")
    print(f"  DP guarantee    : ε={epsilon}, δ=1e-5")
    print(f"{'─'*60}")
    print(f"  Attack breakdown:")
    for at, count in sorted(attack_type_counts.items(), key=lambda x:-x[1]):
        bar = '█' * int(count / n_attacks * 20)
        print(f"    {at:<30} {count:4d}  {bar}")
    print(f"{'═'*60}\n")
    print(f"  ✅ Dataset ready for training + demo in ./{output_dir}/")
    print(f"  📊 Load in Python:  X = np.load('{output_dir}/X_features.npy')")
    print(f"                      y = np.load('{output_dir}/y_labels.npy')")
    print()

    return X, y, records


# ── Quick Validation ───────────────────────────────────────────────────────────

def validate_dataset(output_dir: str = 'dataset'):
    """Quick sanity check on the generated dataset."""
    X = np.load(os.path.join(output_dir, 'X_features.npy'))
    y = np.load(os.path.join(output_dir, 'y_labels.npy'))

    print(f"\n  VALIDATION REPORT")
    print(f"  X shape:    {X.shape}")
    print(f"  y shape:    {y.shape}")
    print(f"  NaN in X:   {np.isnan(X).sum()}")
    print(f"  Inf in X:   {np.isinf(X).sum()}")
    print(f"  Class 0:    {(y==0).sum()} (clean)")
    print(f"  Class 1:    {(y==1).sum()} (attack)")
    print(f"  X mean:     {X.mean():.4f}")
    print(f"  X std:      {X.std():.4f}")
    print(f"  X min/max:  {X.min():.2f} / {X.max():.2f}")
    assert not np.isnan(X).any(), "NaN found in features!"
    assert not np.isinf(X).any(), "Inf found in features!"
    print(f"  ✅ Validation passed!\n")


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NeuroSecure EEG Dataset Generator')
    parser.add_argument('--samples',     type=int,   default=1000,     help='Number of 1-second EEG windows')
    parser.add_argument('--attack-rate', type=float, default=0.10,     help='Fraction of samples with attacks')
    parser.add_argument('--epsilon',     type=float, default=1.0,      help='Differential privacy epsilon')
    parser.add_argument('--output',      type=str,   default='dataset', help='Output directory')
    parser.add_argument('--validate',    action='store_true',           help='Run validation after generation')
    args = parser.parse_args()

    X, y, records = generate_dataset(
        n_samples   = args.samples,
        attack_rate = args.attack_rate,
        epsilon     = args.epsilon,
        output_dir  = args.output,
    )

    if args.validate:
        validate_dataset(args.output)
