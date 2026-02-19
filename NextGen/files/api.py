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


# ── Training Endpoints ─────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"name": "SYNAPSE API", "version": "2.0"}


@app.get("/api/status")
async def get_status():
    """Get current status"""
    return manager.get_status()


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
    """Get current metrics"""
    if manager.metrics_history:
        return manager.metrics_history[-1]
    return {}


@app.get("/api/metrics/history")
async def get_metrics_history(limit: int = 100):
    """Get metrics history"""
    return manager.metrics_history[-limit:]


@app.get("/api/neurochemistry")
async def get_neurochemistry():
    """Get neurochemistry"""
    return manager.get_neurochemistry()


# ── Chat Endpoint ────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Chat with the system"""
    manager.messages.append({
        "from": "user",
        "text": message.text,
        "timestamp": datetime.now().isoformat()
    })

    # Generate response
    if manager.metrics_history:
        last = manager.metrics_history[-1]
        response = f"Φ={last.get('phi',0):.3f}, Agency={last.get('mean_agency',0):.3f}. "
        response += "Я обрабатываю опыт и учусь."
    else:
        response = "Я готова начать обучение."

    manager.messages.append({
        "from": "brain",
        "text": response,
        "timestamp": datetime.now().isoformat()
    })

    return {"response": response}


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
            # Send status update
            await websocket.send_json({
                "type": "status",
                **manager.get_status()
            })

            # If running, do a step and send metrics
            if manager.state == TrainingState.RUNNING:
                # Metrics already sent by training loop
                pass

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
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
