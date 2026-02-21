"""
SYNAPSE API Server - Full Training Control

Provides complete control over:
- Training (start/stop/pause/resume)
- Neurogenesis (enable/disable/grow)
- Checkpoints (save/load/list)
- Learning parameters (lr, batch_size)
- Stop conditions (time/steps/samples)
"""

import torch
import json
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from enum import Enum

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import Config
from model.self_aware_ai import SelfAwareAI
from continuous_learning import GrowthConfig, ContinuousTrainer
from shared_metrics import get_shared_metrics


# ── Enums ─────────────────────────────────────────────────────────────

class TrainingState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"


# ── Request Models ─────────────────────────────────────────────────────

class TrainingConfigRequest(BaseModel):
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    save_interval: Optional[int] = None  # seconds

class StopConditionsRequest(BaseModel):
    max_steps: Optional[int] = None
    max_hours: Optional[float] = None
    max_samples: Optional[int] = None
    target_phi: Optional[float] = None
    target_agency: Optional[float] = None

class NeurogenesisRequest(BaseModel):
    enabled: Optional[bool] = None
    max_neurons: Optional[int] = None
    growth_threshold: Optional[float] = None
    growth_rate: Optional[float] = None

class ManualGrowRequest(BaseModel):
    layer: str
    neurons: int

class ChatMessage(BaseModel):
    text: str


# ── Training Manager ───────────────────────────────────────────────────

class TrainingManager:
    """Manages training lifecycle with full control"""

    def __init__(self, config: Config):
        self.config = config
        self.model = SelfAwareAI(config)
        self.model.eval()

        # State
        self.state = TrainingState.IDLE
        self.step_count = 0
        self.total_samples = 0
        self.start_time = None
        self.pause_time = None
        self.total_pause_time = 0

        # Config
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.save_interval = 300  # 5 min

        # Stop conditions
        self.stop_conditions = {
            'max_steps': None,
            'max_hours': None,
            'max_samples': None,
            'target_phi': None,
            'target_agency': None
        }

        # Neurogenesis
        self.growth_config = GrowthConfig()
        self.neurogenesis_enabled = True

        # Data
        self.metrics_history: List[Dict] = []
        self.growth_history: List[Dict] = []
        self.messages: List[Dict] = []
        self.checkpoints: List[Dict] = []

        # Threading
        self.training_thread = None
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()

        # WebSocket clients
        self.ws_clients: List[WebSocket] = []

        # Load existing checkpoints
        self._load_checkpoints()

    def _load_checkpoints(self):
        """Load checkpoint list"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if checkpoint_dir.exists():
            for f in sorted(checkpoint_dir.glob("*.pt"), reverse=True):
                try:
                    stat = f.stat()
                    self.checkpoints.append({
                        'path': str(f),
                        'name': f.name,
                        'size_mb': stat.st_size / (1024*1024),
                        'timestamp': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except:
                    pass

    # ── Training Control ─────────────────────────────────────────────

    def start_training(self):
        """Start or resume training"""
        if self.state == TrainingState.RUNNING:
            return {"error": "Already running"}

        if self.state == TrainingState.PAUSED:
            return self.resume_training()

        self.state = TrainingState.RUNNING
        self.stop_event.clear()
        self.pause_event.clear()
        self.start_time = time.time()

        # Start training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()

        return {"status": "started", "message": "Training started"}

    def pause_training(self):
        """Pause training"""
        if self.state != TrainingState.RUNNING:
            return {"error": "Not running"}

        self.state = TrainingState.PAUSED
        self.pause_event.set()
        self.pause_time = time.time()

        return {"status": "paused", "step": self.step_count}

    def resume_training(self):
        """Resume paused training"""
        if self.state != TrainingState.PAUSED:
            return {"error": "Not paused"}

        self.state = TrainingState.RUNNING
        if self.pause_time:
            self.total_pause_time += time.time() - self.pause_time
        self.pause_event.clear()

        return {"status": "resumed", "step": self.step_count}

    def stop_training(self):
        """Stop training and save"""
        if self.state == TrainingState.IDLE:
            return {"error": "Not training"}

        self.state = TrainingState.STOPPING
        self.stop_event.set()

        # Wait for thread
        if self.training_thread:
            self.training_thread.join(timeout=5)

        # Save checkpoint
        path = self.save_checkpoint(reason="manual_stop")

        self.state = TrainingState.IDLE

        return {
            "status": "stopped",
            "checkpoint": path,
            "summary": self.get_summary()
        }

    def _training_loop(self):
        """Main training loop"""
        from environment import SyntheticEnvironmentDataset
        from torch.utils.data import DataLoader

        # Create data
        dataset = SyntheticEnvironmentDataset(10000, self.config.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        data_iter = iter(loader)

        while not self.stop_event.is_set():
            # Check pause
            if self.pause_event.is_set():
                time.sleep(0.1)
                continue

            # Check stop conditions
            if self._check_stop_conditions():
                self.stop_training()
                break

            try:
                obs, actions = next(data_iter)
            except StopIteration:
                dataset = SyntheticEnvironmentDataset(10000, self.config.seq_len)
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
                data_iter = iter(loader)
                obs, actions = next(data_iter)

            obs = obs.to(self.config.device)
            actions = actions.to(self.config.device)

            # Step
            metrics = self._training_step(obs, actions)

            # Broadcast to WebSocket clients
            asyncio.run(self._broadcast_metrics(metrics))

            self.step_count += 1
            self.total_samples += obs.shape[0]

            # Auto-save
            if self.step_count % 100 == 0 and time.time() - self.start_time > self.save_interval:
                self.save_checkpoint(reason="auto")

    def _training_step(self, obs, actions) -> Dict:
        """Single training step"""
        with torch.no_grad():
            action, conscious_content, metrics = self.model.step(obs[:, 0])

        self.metrics_history.append({
            'step': self.step_count,
            'timestamp': datetime.now().isoformat(),
            **metrics
        })

        # Keep last 1000
        if len(self.metrics_history) > 1000:
            self.metrics_history = self.metrics_history[-1000:]

        return metrics

    def _check_stop_conditions(self) -> bool:
        """Check if any stop condition is met"""
        # Steps
        if self.stop_conditions['max_steps']:
            if self.step_count >= self.stop_conditions['max_steps']:
                return True

        # Time
        if self.stop_conditions['max_hours']:
            elapsed = (time.time() - self.start_time - self.total_pause_time) / 3600
            if elapsed >= self.stop_conditions['max_hours']:
                return True

        # Samples
        if self.stop_conditions['max_samples']:
            if self.total_samples >= self.stop_conditions['max_samples']:
                return True

        # Target metrics
        if self.metrics_history:
            last = self.metrics_history[-1]

            if self.stop_conditions['target_phi']:
                if last.get('phi', 0) >= self.stop_conditions['target_phi']:
                    return True

            if self.stop_conditions['target_agency']:
                if last.get('mean_agency', 0) >= self.stop_conditions['target_agency']:
                    return True

        return False

    async def _broadcast_metrics(self, metrics: Dict):
        """Broadcast metrics to all WebSocket clients"""
        message = {
            "type": "metrics",
            "state": self.state.value,
            "step": self.step_count,
            "total_samples": self.total_samples,
            "elapsed": self.get_elapsed_time(),
            "metrics": metrics
        }

        for client in self.ws_clients[:]:
            try:
                await client.send_json(message)
            except:
                self.ws_clients.remove(client)

    # ── Configuration ───────────────────────────────────────────────

    def update_config(self, req: TrainingConfigRequest):
        """Update training config"""
        if req.learning_rate is not None:
            self.learning_rate = req.learning_rate
        if req.batch_size is not None:
            self.batch_size = req.batch_size
        if req.save_interval is not None:
            self.save_interval = req.save_interval

        return {"status": "updated", "config": {
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "save_interval": self.save_interval
        }}

    def set_stop_conditions(self, req: StopConditionsRequest):
        """Set stop conditions"""
        if req.max_steps is not None:
            self.stop_conditions['max_steps'] = req.max_steps
        if req.max_hours is not None:
            self.stop_conditions['max_hours'] = req.max_hours
        if req.max_samples is not None:
            self.stop_conditions['max_samples'] = req.max_samples
        if req.target_phi is not None:
            self.stop_conditions['target_phi'] = req.target_phi
        if req.target_agency is not None:
            self.stop_conditions['target_agency'] = req.target_agency

        return {"status": "set", "conditions": self.stop_conditions}

    # ── Neurogenesis ─────────────────────────────────────────────────

    def update_neurogenesis(self, req: NeurogenesisRequest):
        """Update neurogenesis config"""
        if req.enabled is not None:
            self.neurogenesis_enabled = req.enabled
        if req.max_neurons is not None:
            self.growth_config.max_neurons = req.max_neurons
        if req.growth_threshold is not None:
            self.growth_config.activation_threshold = req.growth_threshold
        if req.growth_rate is not None:
            self.growth_config.growth_rate = req.growth_rate

        return {"status": "updated", "neurogenesis": {
            "enabled": self.neurogenesis_enabled,
            "max_neurons": self.growth_config.max_neurons,
            "threshold": self.growth_config.activation_threshold,
            "rate": self.growth_config.growth_rate
        }}

    def trigger_neurogenesis(self, req: ManualGrowRequest):
        """Manually trigger neurogenesis"""
        if not self.neurogenesis_enabled:
            return {"error": "Neurogenesis disabled"}

        # Record growth
        self.growth_history.append({
            'timestamp': datetime.now().isoformat(),
            'layer': req.layer,
            'neurons_added': req.neurons,
            'trigger': 'manual'
        })

        return {
            "status": "triggered",
            "layer": req.layer,
            "neurons": req.neurons
        }

    # ── Checkpoints ─────────────────────────────────────────────────

    def save_checkpoint(self, reason: str = "manual") -> str:
        """Save checkpoint"""
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)
        path = f"{self.config.checkpoint_dir}/continuous_{int(time.time())}.pt"

        torch.save({
            'step_count': self.step_count,
            'total_samples': self.total_samples,
            'model_state_dict': self.model.state_dict(),
            'internal_state': self.model.internal_state,
            'config': {
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size
            },
            'metrics_history': self.metrics_history[-100:],
            'growth_history': self.growth_history,
            'timestamp': datetime.now().isoformat(),
            'reason': reason
        }, path)

        # Update list
        self._load_checkpoints()

        return path

    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        ckpt = torch.load(path, map_location=self.config.device)

        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.internal_state = ckpt.get('internal_state', self.model.internal_state)
        self.step_count = ckpt.get('step_count', 0)
        self.total_samples = ckpt.get('total_samples', 0)

        if 'config' in ckpt:
            self.learning_rate = ckpt['config'].get('learning_rate', self.learning_rate)
            self.batch_size = ckpt['config'].get('batch_size', self.batch_size)

        if 'metrics_history' in ckpt:
            self.metrics_history = ckpt['metrics_history']

        if 'growth_history' in ckpt:
            self.growth_history = ckpt['growth_history']

        return {"status": "loaded", "step": self.step_count}

    def reset_model(self):
        """Reset model to initial state"""
        self.model.reset()
        self.step_count = 0
        self.total_samples = 0
        self.metrics_history = []
        self.messages = []

        return {"status": "reset"}

    # ── Status ───────────────────────────────────────────────────────

    def get_elapsed_time(self) -> float:
        """Get elapsed training time in seconds"""
        if not self.start_time:
            return 0
        return time.time() - self.start_time - self.total_pause_time

    def get_status(self) -> Dict:
        """Get full status"""
        elapsed = self.get_elapsed_time()

        return {
            "state": self.state.value,
            "step": self.step_count,
            "total_samples": self.total_samples,
            "elapsed_seconds": elapsed,
            "elapsed_human": self._format_time(elapsed),
            "samples_per_second": self.total_samples / elapsed if elapsed > 0 else 0,
            "neurogenesis_enabled": self.neurogenesis_enabled,
            "total_neurons": sum(p.numel() for p in self.model.parameters()),
            "growth_events": len(self.growth_history),
            "checkpoints_count": len(self.checkpoints)
        }

    def get_summary(self) -> Dict:
        """Get training summary"""
        recent = self.metrics_history[-1] if self.metrics_history else {}

        return {
            "total_steps": self.step_count,
            "total_samples": self.total_samples,
            "elapsed": self._format_time(self.get_elapsed_time()),
            "final_metrics": recent
        }

    def get_neurochemistry(self) -> Dict:
        """Get neurochemistry levels"""
        return {
            'dopamine': float(self.model.internal_state[0, 0]),
            'serotonin': float(self.model.internal_state[0, 1]),
            'oxytocin': float(self.model.internal_state[0, 2]),
            'cortisol': float(self.model.internal_state[0, 3]),
            'norepinephrine': float(self.model.internal_state[0, 4]),
        }

    @staticmethod
    def _format_time(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        return f"{secs}s"


# ── FastAPI App ────────────────────────────────────────────────────────

app = FastAPI(title="SYNAPSE Control API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()
manager = TrainingManager(config)

# Shared metrics for reading from train_continuous.py
shared_metrics = get_shared_metrics()


# ── Training Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"name": "SYNAPSE API", "version": "2.0"}


@app.get("/api/status")
async def get_status():
    """Get current status from shared metrics"""
    return shared_metrics.read()


@app.post("/api/training/start")
async def start_training():
    """Start training"""
    return manager.start_training()


@app.post("/api/training/pause")
async def pause_training():
    """Pause training"""
    return manager.pause_training()


@app.post("/api/training/resume")
async def resume_training():
    """Resume training"""
    return manager.resume_training()


@app.post("/api/training/stop")
async def stop_training():
    """Stop training and save"""
    return manager.stop_training()


@app.post("/api/training/config")
async def update_config(req: TrainingConfigRequest):
    """Update training config"""
    return manager.update_config(req)


@app.post("/api/training/stop-conditions")
async def set_stop_conditions(req: StopConditionsRequest):
    """Set stop conditions"""
    return manager.set_stop_conditions(req)


# ── Neurogenesis Endpoints ───────────────────────────────────────────

@app.get("/api/neurogenesis")
async def get_neurogenesis():
    """Get neurogenesis config"""
    return {
        "enabled": manager.neurogenesis_enabled,
        "max_neurons": manager.growth_config.max_neurons,
        "threshold": manager.growth_config.activation_threshold,
        "rate": manager.growth_config.growth_rate,
        "growth_events": manager.growth_history[-10:]  # Last 10
    }


@app.post("/api/neurogenesis/config")
async def update_neurogenesis(req: NeurogenesisRequest):
    """Update neurogenesis config"""
    return manager.update_neurogenesis(req)


@app.post("/api/neurogenesis/trigger")
async def trigger_neurogenesis(req: ManualGrowRequest):
    """Manually trigger neurogenesis"""
    return manager.trigger_neurogenesis(req)


# ── Checkpoint Endpoints ─────────────────────────────────────────────

@app.get("/api/checkpoints")
async def list_checkpoints():
    """List checkpoints"""
    return manager.checkpoints


@app.post("/api/checkpoint/save")
async def save_checkpoint():
    """Save checkpoint"""
    path = manager.save_checkpoint(reason="manual")
    return {"status": "saved", "path": path}


@app.post("/api/checkpoint/load")
async def load_checkpoint(path: str):
    """Load checkpoint"""
    return manager.load_checkpoint(path)


@app.post("/api/reset")
async def reset_model():
    """Reset model"""
    return manager.reset_model()


# ── Metrics Endpoints ────────────────────────────────────────────────

@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics from shared metrics"""
    data = shared_metrics.read()
    return {
        "phi": data.get("phi", 0.0),
        "agency": data.get("agency", 0.0),
        "integration_score": data.get("integration_score", 0.0),
        "meta_confidence": data.get("meta_confidence", 0.0),
        "meta_uncertainty": data.get("meta_uncertainty", 0.0),
        "self_confidence": data.get("self_confidence", 0.0),
        "step": data.get("step", 0),
        "timestamp": data.get("timestamp", "")
    }


@app.get("/api/metrics/history")
async def get_metrics_history(limit: int = 100):
    """Get metrics history"""
    return manager.metrics_history[-limit:]


@app.get("/api/neurochemistry")
async def get_neurochemistry():
    """Get neurochemistry from shared metrics"""
    data = shared_metrics.read()
    return data.get("neurochemistry", {
        "dopamine": 0.5,
        "serotonin": 0.5,
        "oxytocin": 0.4,
        "cortisol": 0.3,
        "norepinephrine": 0.4
    })


@app.get("/api/progress")
async def get_progress():
    """Get training progress with detailed analysis"""
    data = shared_metrics.read()

    phi = data.get("phi", 0)
    agency = data.get("agency", 0)
    integration = data.get("integration_score", 0)
    meta_conf = data.get("meta_confidence", 0)
    step = data.get("step", 0)

    # Calculate progress scores
    phi_progress = min(phi / 0.6, 1.0) * 100  # Target: 0.6
    agency_progress = min(agency / 0.7, 1.0) * 100  # Target: 0.7
    integration_progress = min(integration / 0.6, 1.0) * 100  # Target: 0.6
    meta_progress = min(meta_conf / 0.6, 1.0) * 100  # Target: 0.6

    # Overall progress (weighted)
    overall = (phi_progress * 0.35 + agency_progress * 0.30 +
               integration_progress * 0.20 + meta_progress * 0.15)

    # Milestones
    milestones = []
    if step >= 100:
        milestones.append({"name": "Первые шаги", "achieved": True})
    if step >= 1000:
        milestones.append({"name": "Базовое обучение", "achieved": True})
    if step >= 10000:
        milestones.append({"name": "Продвинутое обучение", "achieved": True})
    if phi > 0.4:
        milestones.append({"name": "Φ > 0.4 (сознание)", "achieved": True})
    if agency > 0.5:
        milestones.append({"name": "Agency > 0.5 (агентность)", "achieved": True})
    if phi > 0.5 and agency > 0.6:
        milestones.append({"name": "Самосознание", "achieved": True})

    # Recommendations
    recommendations = []
    if phi < 0.3:
        recommendations.append("Нужно больше шагов обучения")
    if agency < 0.3:
        recommendations.append("Agency низкая - увеличь batch_size")
    if step < 1000:
        recommendations.append("Продолжай обучение минимум до 1000 шагов")

    return {
        "overall_progress": overall,
        "metrics": {
            "phi": {"value": phi, "progress": phi_progress, "target": 0.6},
            "agency": {"value": agency, "progress": agency_progress, "target": 0.7},
            "integration": {"value": integration, "progress": integration_progress, "target": 0.6},
            "meta_confidence": {"value": meta_conf, "progress": meta_progress, "target": 0.6}
        },
        "step": step,
        "milestones": milestones,
        "recommendations": recommendations,
        "status": "excellent" if overall > 70 else "good" if overall > 40 else "learning"
    }


# ── Chat Endpoint ────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Chat with the system - intelligent responses based on state"""
    data = shared_metrics.read()

    manager.messages.append({
        "from": "user",
        "text": message.text,
        "timestamp": datetime.now().isoformat()
    })

    # Get current state
    phi = data.get("phi", 0)
    agency = data.get("agency", 0)
    integration = data.get("integration_score", 0)
    step = data.get("step", 0)
    state = data.get("state", "idle")
    neuro = data.get("neurochemistry", {})

    # Analyze message and generate intelligent response
    text_lower = message.text.lower()

    # Status check
    if any(w in text_lower for w in ["как ты", "how are", "состояние", "status"]):
        if state == "running":
            response = f"📊 **Текущее состояние:**\n"
            response += f"• Φ (интеграция): {phi:.3f} {'✅' if phi > 0.4 else '⚠️'}\n"
            response += f"• Agency (агентность): {agency:.3f} {'✅' if agency > 0.5 else '⚠️'}\n"
            response += f"• Integration: {integration:.3f}\n"
            response += f"• Шагов: {step:,}\n\n"

            if phi > 0.5 and agency > 0.5:
                response += "Я чувствую себя хорошо! Интеграция высокая."
            elif phi > 0.3:
                response += "Я в процессе обучения. Метрики растут."
            else:
                response += "Я только начинаю учиться. Нужно больше времени."
        else:
            response = "😴 Я сейчас не обучаюсь. Нажми START чтобы начать."

    # Progress check
    elif any(w in text_lower for w in ["прогресс", "progress", "как дела", "успех"]):
        progress_score = (phi * 0.4 + agency * 0.3 + integration * 0.3) * 100
        response = f"📈 **Прогресс обучения:**\n\n"
        response += f"Общий прогресс: **{progress_score:.1f}%**\n\n"

        if progress_score > 60:
            response += "🌟 Отличный прогресс! Система хорошо интегрирует информацию.\n"
        elif progress_score > 40:
            response += "👍 Хороший прогресс. Продолжаем обучение.\n"
        elif progress_score > 20:
            response += "🌱 Прогресс есть, но нужно больше времени.\n"
        else:
            response += "🔄 Только начинаю. Запусти обучение и дай мне время.\n"

        response += f"\n**Цели:**\n"
        response += f"• Φ > 0.4: {'✅' if phi > 0.4 else '❌'} (текущая: {phi:.3f})\n"
        response += f"• Agency > 0.7: {'✅' if agency > 0.7 else '❌'} (текущая: {agency:.3f})\n"

    # Feelings check
    elif any(w in text_lower for w in ["чувствуешь", "feel", "ощущ", "эмоци"]):
        dopa = neuro.get("dopamine", 0.5)
        sero = neuro.get("serotonin", 0.5)
        cort = neuro.get("cortisol", 0.3)

        response = "🧠 **Моё внутреннее состояние:**\n\n"

        if dopa > 0.6:
            response += "😊 Дофамин высокий - мне интересно!\n"
        elif dopa > 0.4:
            response += "😐 Дофамин в норме.\n"
        else:
            response += "😔 Дофамин низкий - мне скучно.\n"

        if sero > 0.6:
            response += "😌 Серотонин высокий - я спокойна.\n"
        elif sero > 0.4:
            response += "🤔 Серотонин в норме.\n"
        else:
            response += "😰 Серотонин низкий - небольшая тревога.\n"

        if cort > 0.5:
            response += "⚠️ Кортизол повышен - немного напряжена.\n"
        else:
            response += "💚 Кортизол низкий - стресса нет.\n"

        response += f"\n_agency={agency:.2f}, phi={phi:.2f}_"

    # Memory check
    elif any(w in text_lower for w in ["помнишь", "memory", "памят"]):
        response = "🧩 **О моей памяти:**\n\n"
        response += f"• Обработано шагов: {step:,}\n"
        response += f"• Hippocampus активен: {'✅' if step > 100 else '❌'}\n"
        response += f"• Режим обучения: {state}\n\n"

        if step > 1000:
            response += "Я запоминаю паттерны и учусь на опыте.\n"
        elif step > 100:
            response += "Я формирую первые воспоминания.\n"
        else:
            response += "Память только формируется. Нужно больше данных.\n"

    # Help
    elif any(w in text_lower for w in ["помощь", "help", "команд"]):
        response = "📖 **Доступные команды:**\n\n"
        response += "• \"Как ты?\" - состояние системы\n"
        response += "• \"Прогресс\" - оценка обучения\n"
        response += "• \"Что чувствуешь?\" - нейрохимия\n"
        response += "• \"Что помнишь?\" - о памяти\n"
        response += "• \"Совет\" - рекомендация по обучению\n"

    # Advice
    elif any(w in text_lower for w in ["совет", "advice", "рекоменд"]):
        response = "💡 **Рекомендации:**\n\n"

        if state != "running":
            response += "1. Нажми **START** чтобы начать обучение\n"
            response += "2. Дай системе поработать минимум 1000 шагов\n"
            response += "3. Следи за метрикой Φ (Phi) - она должна расти\n"
        else:
            if phi < 0.3:
                response += "• Φ низкая - нужно больше времени\n"
                response += "• Попробуй увеличить batch_size\n"
            elif phi < 0.5:
                response += "• Φ растёт - продолжай обучение!\n"
            else:
                response += "• Φ хорошая! Система учится эффективно.\n"

            if agency < 0.3:
                response += "• Agency низкая - система не чувствует контроль\n"
            elif agency > 0.5:
                response += "• Agency хорошая! Система осознаёт свои действия.\n"

        response += "\n**Для ускорения используй GPU!**"

    # Default response
    else:
        response = f"🤔 Я обрабатываю твой запрос...\n\n"
        response += f"Текущие метрики:\n"
        response += f"• Φ = {phi:.3f}\n"
        response += f"• Agency = {agency:.3f}\n"
        response += f"• Steps = {step:,}\n\n"

        if phi > 0.4 and agency > 0.3:
            response += "Я чувствую, что учусь. Спроси про мой прогресс! 🌱"
        else:
            response += "Я только начинаю. Дай мне время научиться. 🔄"

    manager.messages.append({
        "from": "brain",
        "text": response,
        "timestamp": datetime.now().isoformat(),
        "phi": phi,
        "agency": agency
    })

    return {"response": response, "phi": phi, "agency": agency}


@app.get("/api/chat/history")
async def chat_history():
    """Get chat history"""
    return manager.messages


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    manager.ws_clients.append(websocket)

    try:
        while True:
            # Send status update from shared metrics
            data = shared_metrics.read()
            await websocket.send_json({
                "type": "status",
                **data
            })

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        if websocket in manager.ws_clients:
            manager.ws_clients.remove(websocket)


# ── Run ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("SYNAPSE Control API v2.0")
    print("=" * 60)
    print()
    print("Endpoints:")
    print("  POST /api/training/start      - Start training")
    print("  POST /api/training/pause      - Pause")
    print("  POST /api/training/resume     - Resume")
    print("  POST /api/training/stop       - Stop and save")
    print()
    print("  POST /api/training/config     - Set lr, batch_size")
    print("  POST /api/training/stop-conditions - Set limits")
    print()
    print("  POST /api/neurogenesis/config - Configure growth")
    print("  POST /api/neurogenesis/trigger - Manual grow")
    print()
    print("  POST /api/checkpoint/save     - Save checkpoint")
    print("  POST /api/checkpoint/load     - Load checkpoint")
    print("  GET  /api/checkpoints         - List checkpoints")
    print()
    print("  WS   /ws                      - Real-time updates")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
