"""
NeuroSecure ML Pipeline
- BCIAttackDetector: feature extraction + Isolation Forest anomaly detection
- PrivacyEngine: Gaussian/Laplace differential privacy noise injection
"""

import numpy as np
from typing import Tuple, List
import warnings
warnings.filterwarnings("ignore")

# ── Feature Extraction ─────────────────────────────────────────────────────────

def bandpower(signal: np.ndarray, fs: int, band: Tuple[float, float]) -> float:
    """Estimate power in a frequency band via Welch's method (manual FFT)."""
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    idx = np.where((freqs >= band[0]) & (freqs <= band[1]))[0]
    return float(np.mean(fft_vals[idx])) if len(idx) else 0.0


BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 80),
}


def extract_channel_features(ch_signal: np.ndarray, fs: int) -> np.ndarray:
    """Per-channel feature vector: statistical + spectral."""
    feats = []
    # Statistical
    feats.append(np.mean(ch_signal))
    feats.append(np.std(ch_signal))
    feats.append(float(np.percentile(ch_signal, 95) - np.percentile(ch_signal, 5)))  # range
    feats.append(float(np.mean(np.abs(np.diff(ch_signal)))))   # mean absolute diff
    # Kurtosis & skewness (manual)
    mu, sigma = np.mean(ch_signal), np.std(ch_signal) + 1e-8
    feats.append(float(np.mean(((ch_signal - mu) / sigma) ** 4)))   # kurtosis
    feats.append(float(np.mean(((ch_signal - mu) / sigma) ** 3)))   # skewness
    # Band powers
    for band in BANDS.values():
        feats.append(bandpower(ch_signal, fs, band))
    # Zero crossing rate
    zcr = float(np.sum(np.diff(np.sign(ch_signal)) != 0)) / len(ch_signal)
    feats.append(zcr)
    return np.array(feats, dtype=np.float32)


# ── Isolation Forest (pure numpy, no sklearn) ─────────────────────────────────

class IsolationTree:
    """Single tree in an Isolation Forest."""

    def __init__(self, max_depth: int = 10):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: np.ndarray, depth: int = 0):
        n, d = X.shape
        if n <= 1 or depth >= self.max_depth:
            return {"type": "leaf", "size": n}
        feat = np.random.randint(0, d)
        lo, hi = X[:, feat].min(), X[:, feat].max()
        if lo == hi:
            return {"type": "leaf", "size": n}
        split = np.random.uniform(lo, hi)
        left_mask = X[:, feat] < split
        node = {
            "type": "internal",
            "feat": feat,
            "split": split,
            "left": self.fit(X[left_mask], depth + 1),
            "right": self.fit(X[~left_mask], depth + 1),
        }
        self.tree = node
        return node

    def path_length(self, x: np.ndarray, node=None, depth: int = 0) -> float:
        if node is None:
            node = self.tree
        if node is None or node["type"] == "leaf":
            size = node["size"] if node else 1
            return depth + _c(size)
        if x[node["feat"]] < node["split"]:
            return self.path_length(x, node["left"], depth + 1)
        return self.path_length(x, node["right"], depth + 1)


def _c(n: int) -> float:
    """Average path length for BST with n nodes."""
    if n <= 1:
        return 0.0
    return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n


class IsolationForest:
    """Lightweight Isolation Forest — no sklearn dependency."""

    def __init__(self, n_estimators: int = 50, max_samples: int = 128, contamination: float = 0.1):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.trees: List[IsolationTree] = []
        self.fitted = False
        self._threshold = 0.0

    def fit(self, X: np.ndarray):
        self.trees = []
        n = min(self.max_samples, len(X))
        max_depth = int(np.ceil(np.log2(n))) + 1
        for _ in range(self.n_estimators):
            idx = np.random.choice(len(X), size=n, replace=False)
            tree = IsolationTree(max_depth=max_depth)
            tree.tree = tree.fit(X[idx])
            self.trees.append(tree)
        self.fitted = True
        # Calibrate threshold on training data
        scores = self._raw_scores(X)
        self._threshold = float(np.percentile(scores, 100 * (1 - self.contamination)))
        return self

    def _raw_scores(self, X: np.ndarray) -> np.ndarray:
        n = min(self.max_samples, len(X))
        c = _c(n)
        scores = []
        for x in X:
            lengths = [t.path_length(x, t.tree) for t in self.trees]
            avg = np.mean(lengths)
            score = 2 ** (-avg / (c + 1e-8))
            scores.append(score)
        return np.array(scores)

    def anomaly_score(self, x: np.ndarray) -> float:
        """Return normalized anomaly score [0,1] for a single sample."""
        if not self.fitted:
            return float(np.random.uniform(0.1, 0.4))   # fallback before training
        n = min(self.max_samples, 128)
        c = _c(n)
        lengths = [t.path_length(x, t.tree) for t in self.trees]
        avg = np.mean(lengths)
        return float(2 ** (-avg / (c + 1e-8)))


# ── Attack Detector ────────────────────────────────────────────────────────────

class BCIAttackDetector:

    def __init__(self):
        self._model = IsolationForest(n_estimators=30, max_samples=64, contamination=0.08)
        self._trained = False
        self._train_on_init()

    def _train_on_init(self):
        """Pre-train on synthetic clean EEG so model is immediately usable."""
        rng = np.random.RandomState(42)
        n_train = 200
        X = []
        for _ in range(n_train):
            t = np.linspace(0, 1, 128)
            channels = []
            for ch in range(8):
                freq = 8 + ch * 2
                sig = 20 * np.sin(2 * np.pi * freq * t) + rng.randn(128) * 4
                channels.append(sig)
            arr = np.array(channels)
            feats = self._featurize(arr, 256)
            X.append(feats)
        X = np.array(X)
        self._model.fit(X)
        self._trained = True

    def _featurize(self, signal_arr: np.ndarray, fs: int) -> np.ndarray:
        ch_feats = [extract_channel_features(ch, fs) for ch in signal_arr]
        return np.concatenate(ch_feats)

    def extract_features(self, signal_arr: np.ndarray, fs: int) -> np.ndarray:
        return self._featurize(signal_arr, fs)

    def predict(self, features: np.ndarray) -> Tuple[float, List[int]]:
        """Returns (anomaly_score, affected_channel_indices)."""
        score = self._model.anomaly_score(features)
        # Identify which channels are anomalous
        n_ch = 8
        feat_per_ch = len(features) // n_ch
        ch_scores = []
        for i in range(n_ch):
            ch_feat = features[i * feat_per_ch:(i + 1) * feat_per_ch]
            std = float(np.std(ch_feat))
            ch_scores.append(std)
        median_std = np.median(ch_scores)
        affected = [i for i, s in enumerate(ch_scores) if s > median_std * 1.8]
        return score, affected

    def status(self) -> dict:
        return {
            "model": "IsolationForest",
            "trained": self._trained,
            "n_estimators": self._model.n_estimators,
            "contamination": self._model.contamination,
        }


# ── Privacy Engine ─────────────────────────────────────────────────────────────

class PrivacyEngine:
    """
    Differential Privacy via Gaussian or Laplace mechanism.
    Adds calibrated noise to BCI signals before any processing.
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, mechanism: str = "gaussian"):
        self.epsilon = epsilon
        self.delta = delta
        self.mechanism = mechanism
        self._total_budget_used = 0.0

    def _gaussian_noise_scale(self, sensitivity: float) -> float:
        """σ = sensitivity * sqrt(2 * ln(1.25/δ)) / ε"""
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

    def _laplace_noise_scale(self, sensitivity: float) -> float:
        """b = sensitivity / ε"""
        return sensitivity / self.epsilon

    def privatize(self, signal: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Add DP noise to signal.
        Returns (noisy_signal, noise_scale).
        """
        # L2 sensitivity estimate: assume clipped to norm 1
        sensitivity = 1.0
        if self.mechanism == "gaussian":
            scale = self._gaussian_noise_scale(sensitivity)
            noise = np.random.normal(0, scale, signal.shape)
        else:
            scale = self._laplace_noise_scale(sensitivity)
            noise = np.random.laplace(0, scale, signal.shape)

        self._total_budget_used += self.epsilon
        return signal + noise.astype(np.float32), scale
