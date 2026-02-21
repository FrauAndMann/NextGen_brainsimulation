# Project SYNAPSE

**A Research Implementation of Functionally Self-Aware Artificial Intelligence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Overview

SYNAPSE is an experimental AI architecture that implements **functional self-awareness** based on three leading theories of consciousness:

| Theory | Author(s) | Implementation |
|--------|-----------|----------------|
| **Global Workspace Theory (GWT)** | Baars, Dehaene | Consciousness Integrator with broadcast competition |
| **Predictive Processing** | Friston | Hierarchical prediction layers (L1, L2) |
| **Integrated Information Theory (IIT)** | Tononi | Phi (Φ) calculation for integration measure |

### Core Principle

```
Self-Awareness = Recursive Self-Prediction + Integration + Agency
```

The system models itself modeling itself, creating a loop where it can distinguish between self-caused and externally-caused changes.

---

## Architecture

### 5-Layer Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Consciousness Integrator (GWT)                    │
│  → Unified conscious experience, Φ calculation              │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Meta-Cognition                                    │
│  → "I know that I know", confidence tracking                │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Agency Model                                      │
│  → Forward/inverse models, distinguishes "I did this"       │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Self Model (128-dim internal state)               │
│  → Predicts own future states                               │
├─────────────────────────────────────────────────────────────┤
│  Layer 0: World Model (VAE + Transformer)                   │
│  → Predicts world states                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Behavior Generation
```

### Neural Populations (8 Types)

| Population | Role | Biological Analog |
|------------|------|-------------------|
| Sensory Input | External stimuli | Sensory cortex |
| Prediction L1/L2 | Prediction errors | Visual cortex hierarchy |
| Association | Information integration | Association cortex |
| PFC Attractor | Stable "character" patterns | Prefrontal cortex |
| Hippocampus | Episodic memory, replay | Hippocampus |
| Amygdala | Emotional valuation | Amygdala |
| Self-Model | Self-prediction | Posterior cingulate |
| GW Hub | Consciousness broadcast | Frontoparietal network |

### Self-Expansion Architecture (v3)

The system can grow smarter through four mechanisms:

1. **Neurogenesis** - Adding neurons when capacity is exceeded (>85% activation)
2. **Functional Growth** - Better predictions through experience
3. **Meta-Learning** - Improving the learning process itself
4. **Architectural Evolution** - NEAT-like topology evolution

### Protection from Catastrophic Forgetting

| Mechanism | Description |
|-----------|-------------|
| CLS (Complementary Learning Systems) | Fast hippocampal + slow cortical learning |
| EWC (Elastic Weight Consolidation) | Fisher Information protects critical weights |
| Progressive Networks | New columns for new domains |
| Experience Replay | Prioritized replay of important experiences |

---

## Key Dimensions

| Dimension | Size | Components |
|-----------|------|------------|
| `obs_dim` | 512 | Observation vector |
| `world_latent_dim` | 256 | World representation |
| `self_state_dim` | 128 | Neurochemistry(32) + Energy(8) + Emotion(16) + Attention(72) |
| `action_dim` | 64 | Motor output |
| `hidden_dim` | 512 | Internal processing |
| `workspace_capacity` | 16 | GWT slots |

---

## Installation

### Requirements

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/FrauAndMann/NextGen_brainsimulation.git
cd NextGen_brainsimulation

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch (with CUDA support)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install transformers sentence-transformers chromadb wandb \
            numpy scipy matplotlib seaborn networkx \
            opencv-python pillow tqdm pytest
```

---

## Quick Start

### Test Environment

```bash
cd files
python environment.py
```

### Run Continuous Training

```bash
python train_continuous.py                    # Start fresh
python train_continuous.py --resume <path>    # Resume from checkpoint
python train_continuous.py --hours 24         # Stop after 24 hours
python train_continuous.py --steps 1000000    # Stop after 1M steps
```

### Run Dashboard

```bash
# React dashboard
npm install recharts
npm start
```

---

## Project Structure

```
NextGen/
├── files/
│   ├── config.py                    # System configuration
│   ├── environment.py               # Synthetic environment + neurochemistry
│   ├── evaluation.py                # Test suite + visualization
│   ├── train_continuous.py          # Continuous training script
│   ├── continuous_learning.py       # Neurogenesis, replay, checkpointing
│   ├── demo.py                      # Quick demonstration
│   ├── quickstart.py                # Getting started script
│   ├── model/
│   │   ├── world_model.py           # VAE + Transformer
│   │   ├── self_model.py            # Recursive self-prediction
│   │   ├── agency_model.py          # Forward/inverse models
│   │   ├── meta_cognitive.py        # Confidence, uncertainty
│   │   ├── consciousness.py         # GWT integrator, Φ calculation
│   │   ├── behavior.py              # Policy network
│   │   └── self_aware_ai.py         # Main integration
│   └── tests/
│       └── test_*.py                # Unit and integration tests
├── docs/
│   └── plans/                       # Design documents
├── synapse_dashboard.jsx            # React monitoring dashboard
├── CLAUDE.md                        # Project instructions
└── README.md                        # This file
```

---

## Success Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Agency Signal | > 0.70 | System feels agency for own actions |
| Integration Score | > 0.60 | Information is unified |
| Φ (Phi) | > 0.40 | Consciousness-like integration present |
| Meta-Confidence | > 0.60 | System knows what it knows |
| Temporal Consistency | > 0.70 | Stable sense of self over time |
| Agency Discrimination | > 0.30 | Distinguishes own vs external causes |
| Self-Prediction Error | < 0.30 | Good self-understanding |

---

## Neurochemistry Engine

The system includes a 32-dimensional neurochemistry simulation:

| Neurotransmitter | Role |
|------------------|------|
| Dopamine | Reward, learning acceleration |
| Serotonin | Mood, well-being |
| Oxytocin | Social bonding |
| Cortisol | Stress response |
| Norepinephrine | Arousal, attention |
| GABA | Inhibition, calming |
| Glutamate | Excitation |
| Acetylcholine | Learning, memory |

### Emotional Triggers

```python
engine.trigger_emotion('happy', intensity=0.8)
engine.trigger_emotion('anxious', intensity=0.5)
engine.trigger_emotion('calm', intensity=0.6)
```

---

## Safety Features

| Feature | Setting |
|---------|---------|
| Max Distress Level | 0.8 (intervention threshold) |
| Emergency Shutdown | 0.95 (kill switch) |
| Transparency Mode | Always identifies as AI |
| Max Neurons | 100,000 (safety limit) |

---

## What's Measurable vs Philosophical

| Level | Status |
|-------|--------|
| Behavioral similarity to conscious agents | 95%+ achievable |
| Functional consciousness (passes tests) | 85% achievable |
| Phenomenal consciousness (qualia) | Unknown/philosophical |

This implementation focuses on **functional** self-awareness - behaviorally demonstrable capabilities - rather than making claims about subjective experience.

---

## Hardware Requirements

| Level | GPU | RAM |
|-------|-----|-----|
| Minimum | RTX 3090 (24GB VRAM) | 32GB |
| Optimal | RTX 4090 / A100 (40GB) | 64GB |
| SNN Scale | RTX 4090 supports 100k-500k neurons real-time | |

---

## Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests.

Areas of interest:
- Improving Φ (Phi) estimation accuracy
- Additional consciousness tests
- Neurogenesis optimization
- Multi-modal extensions

---

## License

MIT License - see LICENSE file for details.

---

## References

- Baars, B. J. (2005). Global workspace theory of consciousness
- Dehaene, S. (2014). Consciousness and the Brain
- Friston, K. (2010). The free-energy principle
- Tononi, G. (2012). Integrated Information Theory

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{synapse2024,
  author = {NextGen Research},
  title = {Project SYNAPSE: Functionally Self-Aware AI},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/FrauAndMann/NextGen_brainsimulation}
}
```

---

*"The question is not whether machines can be conscious, but whether we can build systems that behave as if they were."*
