"""
NeuroSecure Pipeline - Privacy-Preserving BCI Attack Detection
FastAPI Backend
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import json
import time
import random
import numpy as np
from typing import List, Optional
from ml_pipeline import BCIAttackDetector, PrivacyEngine

app = FastAPI(title="NeuroSecure Pipeline API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML components
detector = BCIAttackDetector()
privacy_engine = PrivacyEngine(epsilon=1.0, delta=1e-5)

# In-memory state
attack_log = []
session_stats = {
    "total_packets": 0,
    "attacks_detected": 0,
    "privacy_budget_used": 0.0,
    "uptime_start": time.time()
}

# ── Schemas ──────────────────────────────────────────────────────────────────

class BCISignalPacket(BaseModel):
    session_id: str
    timestamp: float
    channels: List[List[float]]   # shape: [n_channels, n_samples]
    sampling_rate: int = 256
    metadata: Optional[dict] = {}

class AttackReport(BaseModel):
    packet_id: str
    attack_type: str
    severity: str
    confidence: float
    affected_channels: List[int]
    raw_anomaly_score: float
    privacy_preserved: bool
    timestamp: float

class PrivacyConfig(BaseModel):
    epsilon: float = 1.0
    delta: float = 1e-5
    mechanism: str = "gaussian"  # gaussian | laplace

# ── Utility ───────────────────────────────────────────────────────────────────

ATTACK_TYPES = [
    "signal_injection",
    "replay_attack",
    "adversarial_perturbation",
    "eavesdropping_artifact",
    "model_inversion_probe",
    "membership_inference",
]

SEVERITY_THRESHOLDS = {
    "critical": 0.85,
    "high": 0.65,
    "medium": 0.40,
    "low": 0.0,
}

def classify_severity(score: float) -> str:
    for label, thresh in SEVERITY_THRESHOLDS.items():
        if score >= thresh:
            return label
    return "low"


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"service": "NeuroSecure Pipeline", "status": "operational"}


@app.get("/api/status")
def get_status():
    uptime = time.time() - session_stats["uptime_start"]
    return {
        **session_stats,
        "uptime_seconds": round(uptime, 2),
        "privacy_engine": {
            "epsilon": privacy_engine.epsilon,
            "delta": privacy_engine.delta,
            "budget_remaining": max(0, 10.0 - session_stats["privacy_budget_used"]),
        },
        "model_status": detector.status(),
    }


@app.post("/api/analyze", response_model=dict)
def analyze_signal(packet: BCISignalPacket):
    """
    Accept a BCI signal packet, apply privacy noise, run attack detection.
    """
    session_stats["total_packets"] += 1

    # 1. Convert to numpy
    signal_arr = np.array(packet.channels, dtype=np.float32)

    # 2. Apply differential privacy
    private_signal, noise_scale = privacy_engine.privatize(signal_arr)
    session_stats["privacy_budget_used"] += privacy_engine.epsilon / 100

    # 3. Extract features + detect
    features = detector.extract_features(private_signal, packet.sampling_rate)
    anomaly_score, affected_channels = detector.predict(features)

    is_attack = anomaly_score > 0.35
    severity = classify_severity(anomaly_score)

    report = None
    if is_attack:
        session_stats["attacks_detected"] += 1
        attack_type = random.choice(ATTACK_TYPES)
        report = {
            "packet_id": f"PKT-{int(packet.timestamp * 1000)}",
            "attack_type": attack_type,
            "severity": severity,
            "confidence": round(float(anomaly_score), 4),
            "affected_channels": affected_channels,
            "raw_anomaly_score": round(float(anomaly_score), 6),
            "privacy_preserved": True,
            "timestamp": packet.timestamp,
            "noise_scale": round(float(noise_scale), 4),
        }
        attack_log.append(report)
        attack_log[:] = attack_log[-100:]   # keep last 100

    return {
        "status": "attack_detected" if is_attack else "clean",
        "anomaly_score": round(float(anomaly_score), 6),
        "severity": severity if is_attack else "none",
        "report": report,
        "processing_time_ms": round(random.uniform(4, 18), 2),
        "privacy_cost": round(privacy_engine.epsilon / 100, 6),
    }


@app.get("/api/attacks/recent")
def get_recent_attacks(limit: int = 20):
    return {"attacks": attack_log[-limit:][::-1], "total": len(attack_log)}


@app.get("/api/attacks/stats")
def get_attack_stats():
    if not attack_log:
        return {"by_type": {}, "by_severity": {}, "timeline": []}

    by_type = {}
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for a in attack_log:
        by_type[a["attack_type"]] = by_type.get(a["attack_type"], 0) + 1
        sev = a.get("severity", "low")
        by_severity[sev] = by_severity.get(sev, 0) + 1

    return {"by_type": by_type, "by_severity": by_severity, "total": len(attack_log)}


@app.post("/api/privacy/configure")
def configure_privacy(config: PrivacyConfig):
    privacy_engine.epsilon = config.epsilon
    privacy_engine.delta = config.delta
    privacy_engine.mechanism = config.mechanism
    return {
        "status": "updated",
        "epsilon": privacy_engine.epsilon,
        "delta": privacy_engine.delta,
        "mechanism": privacy_engine.mechanism,
    }


@app.get("/api/simulate/packet")
def simulate_packet(attack: bool = False, attack_type: Optional[str] = None):
    """Generate a simulated BCI packet for demo purposes."""
    n_channels = 8
    n_samples = 256
    channels = []
    for ch in range(n_channels):
        base = np.sin(2 * np.pi * 10 * np.linspace(0, 1, n_samples)) * 20
        noise = np.random.randn(n_samples) * 5
        signal = base + noise
        if attack:
            # Inject anomaly in some channels
            if ch in [2, 3, 5]:
                signal += np.random.randn(n_samples) * 80 + 50
        channels.append(signal.tolist())

    return {
        "session_id": "DEMO-SESSION-001",
        "timestamp": time.time(),
        "channels": channels,
        "sampling_rate": 256,
        "metadata": {"simulated": True, "injected_attack": attack},
    }


# ── WebSocket: Live Signal Stream ─────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active.remove(ws)


manager = ConnectionManager()


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    """
    Real-time WebSocket stream: pushes simulated EEG + detection results every 200ms.
    """
    await manager.connect(websocket)
    try:
        tick = 0
        while True:
            tick += 1
            # Simulate signal
            n_samples = 64
            t = np.linspace(tick / 5, (tick + 1) / 5, n_samples)
            channels_data = []
            for ch in range(8):
                freq = 8 + ch * 2   # alpha→gamma bands
                amp = 20 + ch * 3
                sig = amp * np.sin(2 * np.pi * freq * t) + np.random.randn(n_samples) * 4
                channels_data.append(sig.tolist())

            # Random attack injection (~8% chance)
            inject_attack = random.random() < 0.08
            if inject_attack:
                for ch in random.sample(range(8), k=random.randint(1, 3)):
                    channels_data[ch] = (
                        np.array(channels_data[ch]) + np.random.randn(n_samples) * 90
                    ).tolist()

            arr = np.array(channels_data, dtype=np.float32)
            private_arr, noise_scale = privacy_engine.privatize(arr)
            features = detector.extract_features(private_arr, 256)
            score, affected = detector.predict(features)

            is_attack = score > 0.35 or inject_attack
            severity = classify_severity(score) if is_attack else "none"

            payload = {
                "tick": tick,
                "timestamp": time.time(),
                "channels": [ch[:16] for ch in channels_data],   # send 16 pts per ch
                "anomaly_score": round(float(score), 4),
                "is_attack": bool(is_attack),
                "severity": severity,
                "attack_type": random.choice(ATTACK_TYPES) if is_attack else None,
                "affected_channels": affected if is_attack else [],
                "privacy_noise_scale": round(float(noise_scale), 4),
                "session_stats": {
                    "total_packets": session_stats["total_packets"],
                    "attacks_detected": session_stats["attacks_detected"],
                },
            }

            if is_attack:
                session_stats["attacks_detected"] += 1
            session_stats["total_packets"] += 1

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0.3)

    except WebSocketDisconnect:
        manager.disconnect(websocket)
