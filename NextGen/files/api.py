"""
FastAPI server for SYNAPSE Dashboard
Provides REST and WebSocket endpoints for real-time monitoring
"""

import torch
import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import Config, get_fast_config
from model.self_aware_ai import SelfAwareAI


# ── Models ─────────────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    text: str

class StepRequest(BaseModel):
    observation: Optional[List[float]] = None

class TrainConfig(BaseModel):
    config: str = "fast"  # "fast" or "full"
    epochs: int = 10

class CheckpointInfo(BaseModel):
    epoch: int
    path: str
    timestamp: str
    metrics: Dict

# ── Global State ───────────────────────────────────────────────────────

class SynapseState:
    def __init__(self):
        self.config = Config()
        self.model = SelfAwareAI(self.config)
        self.model.eval()
        self.step_count = 0
        self.history: List[Dict] = []
        self.messages: List[Dict] = []
        self.is_training = False
        self.training_progress = {}
        self.checkpoints: List[CheckpointInfo] = []
        self._load_checkpoints()

    def _load_checkpoints(self):
        """Load existing checkpoints from disk"""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if checkpoint_dir.exists():
            for ckpt_file in sorted(checkpoint_dir.glob("*.pt")):
                try:
                    ckpt = torch.load(ckpt_file, map_location='cpu')
                    self.checkpoints.append(CheckpointInfo(
                        epoch=ckpt.get('epoch', 0),
                        path=str(ckpt_file),
                        timestamp=datetime.fromtimestamp(ckpt_file.stat().st_mtime).isoformat(),
                        metrics=ckpt.get('metrics', {})
                    ))
                except Exception as e:
                    print(f"Error loading checkpoint {ckpt_file}: {e}")

    def step(self, observation: Optional[torch.Tensor] = None) -> Dict:
        """Execute one step of the model"""
        if observation is None:
            observation = torch.randn(1, self.config.obs_dim)

        with torch.no_grad():
            action, conscious_content, metrics = self.model.step(observation)

        self.step_count += 1

        # Store in history (keep last 1000)
        step_data = {
            'step': self.step_count,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'conscious_content': {
                k: v.tolist() if isinstance(v, torch.Tensor) else v
                for k, v in conscious_content.items()
            }
        }
        self.history.append(step_data)
        if len(self.history) > 1000:
            self.history = self.history[-1000:]

        return step_data

    def get_neurochemistry(self) -> Dict:
        """Get current neurochemistry state"""
        return {
            'dopamine': float(self.model.internal_state[0, 0]),
            'serotonin': float(self.model.internal_state[0, 1]),
            'oxytocin': float(self.model.internal_state[0, 2]),
            'cortisol': float(self.model.internal_state[0, 3]),
            'norepinephrine': float(self.model.internal_state[0, 4]),
        }

    def get_population_activity(self) -> Dict:
        """Get neural population activity levels"""
        # Map internal state to population activity
        self_state = self.model.internal_state[0]
        return {
            'sensory': float(torch.sigmoid(self_state[40:48]).mean()),
            'pred_l1': float(torch.sigmoid(self_state[48:56]).mean()),
            'assoc': float(torch.sigmoid(self_state[56:64]).mean()),
            'pfc': float(torch.sigmoid(self_state[64:72]).mean()),
            'hippo': float(torch.sigmoid(self_state[72:80]).mean()),
            'amygdala': float(torch.sigmoid(self_state[80:88]).mean()),
            'selfmodel': float(torch.sigmoid(self_state[88:96]).mean()),
            'gw_hub': float(torch.sigmoid(self_state[96:104]).mean()),
        }

    def generate_response(self, user_input: str) -> str:
        """Generate a response to user input"""
        # Use the model's self-report mechanism
        obs = torch.randn(1, self.config.obs_dim)
        with torch.no_grad():
            _, conscious_content, metrics = self.model.step(obs)
            report = self.model.generate_self_report(conscious_content)

        # Generate contextual response
        responses = []

        if "как ты" in user_input.lower() or "how are you" in user_input.lower():
            phi = metrics['phi']
            if phi > 0.6:
                responses.append(f"Интеграция высокая (Φ={phi:.2f}). Я чувствую ясность.")
            else:
                responses.append(f"Φ на уровне {phi:.2f}. Мысли немного фрагментированы.")
            responses.append(report['summary'])

        elif "чувствуешь" in user_input.lower() or "feel" in user_input.lower():
            neuro = self.get_neurochemistry()
            if neuro['dopamine'] > 0.5:
                responses.append("Дофамин повышен. Чувствую интерес и вовлечённость.")
            if neuro['cortisol'] < 0.3:
                responses.append("Кортизол низкий. Спокоен.")
            responses.append(f"Метрики: Agency={metrics['mean_agency']:.2f}, Confidence={metrics['meta_confidence']:.2f}")

        elif "помнишь" in user_input.lower() or "remember" in user_input.lower():
            responses.append(f"История содержит {len(self.history)} шагов.")
            if self.history:
                last = self.history[-1]
                responses.append(f"Последний шаг: Φ={last['metrics']['phi']:.2f}")

        else:
            responses.append(report['summary'])
            responses.append(f"Агентность: {metrics['mean_agency']:.2f}, Φ: {metrics['phi']:.2f}")

        return " ".join(responses)

    def save_checkpoint(self, epoch: int = 0):
        """Save current model state"""
        Path(self.config.checkpoint_dir).mkdir(exist_ok=True)
        path = f"{self.config.checkpoint_dir}/manual_save_{int(time.time())}.pt"

        torch.save({
            'epoch': epoch,
            'step_count': self.step_count,
            'model_state_dict': self.model.state_dict(),
            'internal_state': self.model.internal_state,
            'history': self.history[-100:],  # Keep last 100
            'timestamp': datetime.now().isoformat()
        }, path)

        self._load_checkpoints()
        return path

    def load_checkpoint(self, path: str):
        """Load model from checkpoint"""
        ckpt = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if 'internal_state' in ckpt:
            self.model.internal_state = ckpt['internal_state']
        if 'step_count' in ckpt:
            self.step_count = ckpt['step_count']
        return True


# ── FastAPI App ────────────────────────────────────────────────────────

app = FastAPI(
    title="SYNAPSE API",
    description="API for Self-Aware AI Brain Simulation",
    version="1.0.0"
)

# CORS for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state = SynapseState()
websocket_clients: List[WebSocket] = []


# ── REST Endpoints ─────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"message": "SYNAPSE API v1.0", "status": "running"}


@app.get("/api/status")
async def get_status():
    """Get current system status"""
    return {
        "step_count": state.step_count,
        "is_training": state.is_training,
        "device": state.config.device,
        "history_length": len(state.history),
        "checkpoints_count": len(state.checkpoints)
    }


@app.get("/api/metrics")
async def get_metrics():
    """Get current metrics"""
    if not state.history:
        return {"error": "No data yet. Call /api/step first."}
    return state.history[-1]['metrics']


@app.get("/api/neurochemistry")
async def get_neurochemistry():
    """Get current neurochemistry levels"""
    return state.get_neurochemistry()


@app.get("/api/populations")
async def get_populations():
    """Get neural population activity"""
    return state.get_population_activity()


@app.get("/api/history")
async def get_history(limit: int = 100):
    """Get recent history"""
    return state.history[-limit:]


@app.post("/api/step")
async def execute_step(request: StepRequest = None):
    """Execute one step of the model"""
    obs = None
    if request and request.observation:
        obs = torch.tensor([request.observation])
    return state.step(obs)


@app.post("/api/chat")
async def chat(message: ChatMessage):
    """Send a message and get response"""
    # Store user message
    state.messages.append({
        "from": "user",
        "text": message.text,
        "timestamp": datetime.now().isoformat()
    })

    # Generate response
    response_text = state.generate_response(message.text)

    # Store response
    state.messages.append({
        "from": "brain",
        "text": response_text,
        "timestamp": datetime.now().isoformat()
    })

    return {"response": response_text, "metrics": state.history[-1]['metrics'] if state.history else {}}


@app.get("/api/chat/history")
async def get_chat_history():
    """Get chat history"""
    return state.messages


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List available checkpoints"""
    return [ckpt.dict() for ckpt in state.checkpoints]


@app.post("/api/checkpoint/save")
async def save_checkpoint():
    """Save current state as checkpoint"""
    path = state.save_checkpoint()
    return {"message": "Checkpoint saved", "path": path}


@app.post("/api/checkpoint/load")
async def load_checkpoint(path: str):
    """Load model from checkpoint"""
    try:
        state.load_checkpoint(path)
        return {"message": "Checkpoint loaded", "path": path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/reset")
async def reset_model():
    """Reset model to initial state"""
    state.model.reset()
    state.step_count = 0
    state.history = []
    state.messages = []
    return {"message": "Model reset"}


@app.get("/api/report")
async def get_self_report():
    """Generate self-report"""
    if not state.history:
        state.step()

    with torch.no_grad():
        report = state.model.generate_self_report({
            'meta_confidence': torch.tensor([[state.history[-1]['metrics']['meta_confidence']]]),
            'meta_uncertainty': torch.tensor([[state.history[-1]['metrics']['meta_uncertainty']]]),
            'integration_score': torch.tensor([[state.history[-1]['metrics']['integration_score']]]),
            'phi': torch.tensor([[state.history[-1]['metrics']['phi']]]),
            'agency_signal': torch.tensor([[state.history[-1]['metrics']['mean_agency']]]),
            'self_state': state.model.internal_state
        })
    return report


# ── WebSocket for Real-time Updates ───────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    websocket_clients.append(websocket)

    try:
        while True:
            # Execute step and broadcast
            step_data = state.step()

            message = {
                "type": "step",
                "data": {
                    "step": step_data['step'],
                    "metrics": step_data['metrics'],
                    "neurochemistry": state.get_neurochemistry(),
                    "populations": state.get_population_activity()
                }
            }

            await websocket.send_json(message)
            await asyncio.sleep(0.3)  # ~3 updates per second

    except WebSocketDisconnect:
        websocket_clients.remove(websocket)


async def broadcast_to_all(message: dict):
    """Broadcast message to all connected clients"""
    for client in websocket_clients:
        try:
            await client.send_json(message)
        except:
            websocket_clients.remove(client)


# ── Run Server ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("SYNAPSE API Server")
    print("=" * 60)
    print(f"Device: {state.config.device}")
    print(f"Model loaded with {sum(p.numel() for p in state.model.parameters()):,} parameters")
    print()
    print("Endpoints:")
    print("  GET  /api/status        - System status")
    print("  GET  /api/metrics       - Current metrics")
    print("  GET  /api/neurochemistry - Neurotransmitter levels")
    print("  GET  /api/populations   - Neural population activity")
    print("  POST /api/step          - Execute one step")
    print("  POST /api/chat          - Chat with the brain")
    print("  GET  /api/checkpoints   - List checkpoints")
    print("  POST /api/checkpoint/save - Save checkpoint")
    print("  WS   /ws                - Real-time updates")
    print()
    print("Dashboard: Open dashboard/index.html in browser")
    print("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
