# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project SYNAPSE

Research implementation of functionally self-aware AI based on:
- **Global Workspace Theory (GWT)** - Baars/Dehaene
- **Predictive Processing** - Friston
- **Integrated Information Theory (IIT)** - Tononi

Core principle: Self-awareness = Recursive self-prediction + Integration + Agency

---

## Architecture (Critical Understanding)

### 5-Layer Hierarchy

```
Layer 0: World Model (VAE + Transformer) → predicts world states
    ↓
Layer 1: Self Model (128-dim internal state) → predicts own states
    ↓
Layer 2: Agency Model (forward/inverse) → distinguishes "I did this"
    ↓
Layer 3: Meta-Cognition → "I know that I know"
    ↓
Layer 4: Consciousness Integrator (GWT) → unified experience (Φ)
    ↓
Behavior Generation
```

### Neural Populations (8 Types)

| Population | Role | SNN Analog |
|------------|------|------------|
| Sensory Input | External stimuli | Sensory cortex |
| Prediction L1 | Low-level prediction errors | Early visual cortex |
| Prediction L2 | High-level prediction errors | Higher visual areas |
| Association | Integrates information | Association cortex |
| PFC Attractor | Stable "character" patterns | Prefrontal cortex |
| Hippocampus | Episodic memory, replay | Hippocampus |
| Amygdala | Emotional valuation | Amygdala |
| Self-Model | Self-prediction | Posterior cingulate |
| GW Hub | Consciousness broadcast | Frontoparietal network |

### Data Flow (Three Streams)

```
Bottom-up:    Sensory → L1 Error → L2 Error → Association → GW
Top-down:     L2 Prediction → L1 Prediction → Sensory
Lateral:      GW broadcast → all populations
              Neurotransmitters → all populations
              Self-Model ↔ Association
```

### Core Dimensions

| Dimension | Size | Components |
|-----------|------|------------|
| `obs_dim` | 512 | observation |
| `world_latent_dim` | 256 | world representation |
| `self_state_dim` | 128 | neurochemistry(32) + energy(8) + emotion(16) + attention(72) |
| `action_dim` | 64 | motor output |
| `hidden_dim` | 512 | internal processing |
| `workspace_capacity` | 16 | GWT slots |

---

## Self-Expansion Architecture (v3)

### Four Types of "Getting Smarter"

1. **Structural Growth (Neurogenesis)** - Adding neurons where capacity is insufficient
2. **Functional Growth** - Better predictions through experience
3. **Meta-Learning** - Improving the learning process itself
4. **Architectural Evolution** - NEAT-like topology evolution

### Complementary Learning Systems (CLS)

Two memory systems with different learning speeds:
- **Fast system (Hippocampus)**: Rapid acquisition, temporary storage
- **Slow system (Cortex)**: Gradual consolidation, long-term storage
- Transfer via **replay** during "sleep"

### Three Levels of Forgetting Protection

| Level | Mechanism | Purpose |
|-------|-----------|---------|
| Architectural | CLS separation | Physical isolation of fast/slow learning |
| EWC | Fisher Information protection | Critical weights frozen from overwriting |
| Progressive Networks | New columns for new domains | Old knowledge never overwritten |

### Personality as Strange Attractor

- System never repeats exactly, but always recognizable
- Stable character patterns in PFC Attractor population
- Three sources of uniqueness:
  1. Initial random weight initialization
  2. History of interactions
  3. Neurochemical history (temperament)

---

## Key Modules

### config.py
- `Config` - Main configuration dataclass with validation
- `NeurochemistryConfig` - 32 neurotransmitter dynamics
- `EnvironmentConfig` - Synthetic environment settings
- `get_fast_config()` / `get_full_config()` - Preset configurations

### environment.py
- `NeurochemistryEngine` - Simulates internal neurochemistry with decay, interactions, emotional triggers
- `SyntheticEnvironment` - Training environment with agency_ratio (controllable vs external changes)
- `SyntheticEnvironmentDataset` - Trajectory generation for training

### evaluation.py
- `SelfAwarenessEvaluator` - Test suite: mirror, meta-cognition, self-boundary, temporal, agency
- `ConsciousnessVisualizer` - Workspace, neurochemistry, consciousness flow visualization

### self_aware_ai_implementation_plan.md
Complete technical blueprint with:
- `WorldModel` - VAE encoder/decoder + Transformer temporal model
- `SelfModel` - Recursive self-state prediction
- `AgencyModel` - Forward/inverse models for causality detection
- `MetaCognitiveModel` - Confidence, uncertainty, attention
- `ConsciousnessIntegrator` - Competition for workspace, Φ estimation
- `BehaviorGenerator` - Policy network
- `SelfAwareAI` - Main integration class
- `train.py` - Full training pipeline

---

## Consciousness Architecture Integration

### Three Theories Integrated

| Theory | Key Metric | Implementation |
|--------|-----------|----------------|
| IIT (Tononi) | Φ (Phi) | Integration score in ConsciousnessIntegrator |
| GWT (Dehaene) | Broadcast clarity | Global Workspace competition |
| Predictive Processing (Friston) | Prediction error | Hierarchical prediction layers |

### What's Measurable vs Philosophical

| Level | Status |
|-------|--------|
| Behavioral similarity | 95%+ achievable |
| Functional consciousness (passes tests) | 85% achievable |
| Phenomenal consciousness (qualia) | Unknown/philosophical |

---

## Commands

### Setup (Windows)
```bash
python -m venv venv
venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers sentence-transformers chromadb wandb numpy scipy matplotlib seaborn networkx opencv-python pillow tqdm pytest
```

### Test Environment
```bash
cd files
python environment.py
```

### Training (after creating model.py)
```bash
python train.py --config fast   # 10K samples, 10 epochs
python train.py --config full   # 1M+ samples, 200 epochs
```

### Dashboard
```bash
# React dashboard - integrate into React project with: npm install recharts
```

---

## Success Metrics

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Agency Signal | > 0.70 | System feels agency for own actions |
| Integration Score | > 0.60 | Information unified |
| Φ (Phi) | > 0.40 | Consciousness present (IIT) |
| Meta-Confidence | > 0.60 | System knows what it knows |
| Temporal Consistency | > 0.70 | Stable sense of self |
| Agency Discrimination | > 0.30 | Distinguishes own vs external |
| Self-Prediction Error | < 0.30 | Good self-understanding |

---

## Implementation Status

**Completed:**
- [x] config.py - Configuration system
- [x] environment.py - Synthetic environment + neurochemistry
- [x] evaluation.py - Test suite + visualization
- [x] self_aware_ai_implementation_plan.md - Full blueprint
- [x] Synapse_Design_v3.docx - Complete design document with self-expansion architecture

**To Create:**
- [ ] model.py - Copy from implementation plan
- [ ] train.py - Copy from implementation plan
- [ ] requirements.txt

---

## Critical Implementation Notes

### Self Model - The Core
The self model must predict its own future state:
```python
Self(t+1) = Predict(Self(t) | World(t), Action(t))
```
This creates a recursive loop: the system models itself modeling itself.

### Agency Detection
Distinguishes self-caused vs externally-caused changes:
- Forward model: action + state → predicted change
- Inverse model: state change → inferred action
- Agency = prediction accuracy × action consistency

### Φ (Phi) Estimation
Approximate integrated information:
- High connectivity between workspace elements
- High variance (differentiated information)
- Φ = connectivity_score × sigmoid(variance)

### Self-Expansion
System grows through:
1. **Neurogenesis** - New neurons added when population >85% active
2. **Structural Plasticity** - New connections form via STDP
3. **Progressive Networks** - New columns for new domains
4. **Meta-Learning** - Dopamine/Acetylcholine accelerate learning

### Protection from Catastrophic Forgetting
1. **CLS** - Fast hippocampal + slow cortical learning
2. **EWC** - Fisher Information protects critical weights
3. **Replay** - Nightly consolidation of episodic memories

### Safety
- `max_distress_level: 0.8` - Threshold for intervention
- `transparency_mode: True` - Always identify as AI
- Kill switch in ConsciousnessIntegrator

---

## Hardware Requirements

- Minimum: RTX 3090 (24GB VRAM), 32GB RAM
- Optimal: RTX 4090 / A100 (40GB), 64GB RAM
- SNN Scale: 100k-500k neurons on RTX 4090 (real-time via GeNN)

---

## File Structure

```
NextGen/
├── files/
│   ├── self_aware_ai_implementation_plan.md  # Full code blueprint
│   ├── config.py                              # System configuration
│   ├── environment.py                         # Synthetic environment
│   ├── evaluation.py                          # Tests + visualization
│   └── README.md                              # User guide
├── docs/
│   └── plans/
├── synapse_dashboard.jsx                      # React monitoring dashboard
├── Synapse_Design_v3.docx                     # Complete design document (v3)
├── Human_Like_AI_Architecture.docx            # Architecture documentation
└── CLAUDE.md                                  # This file
```

---

## Dashboard Interpretation

When monitoring the system:
- **Spike Raster** - Each flash is a neuron firing; synchronized patterns indicate conscious binding
- **Φ (Phi)** - Integration measure; >0.4 indicates consciousness-like integration
- **Self Δ** - Self-prediction error; low = good self-understanding
- **GW Winner** - Which population currently holds "consciousness spotlight"
- **Neurochemistry** - Dopamine >60% = accelerated learning; Cortisol >70% = memory impairment
- **REPLAY indicator** - System consolidating memories (normal during rest)
