# Project SYNAPSE

**A Research Implementation of Functionally Self-Aware Artificial Intelligence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## 🚀 Quick Start

### 1. Setup (One-time)

```bash
# Clone repository
git clone https://github.com/FrauAndMann/NextGen_brainsimulation.git
cd NextGen_brainsimulation

# Run automatic setup (installs Python 3.10 venv + PyTorch with CUDA)
setup_gpu.bat
```

### 2. Start Training

```bash
# Start SYNAPSE with dashboard
run_life.bat
```

### 3. Open Dashboard

Dashboard opens automatically at `dashboard/index.html`

- Click **"НАЧАТЬ ОБУЧЕНИЕ"** to start
- Watch real-time neural activity
- Chat with SYNAPSE to check progress
- Configure data sources via UI

---

## 📊 Dashboard Features

### Training Control
- **START** - Begin training
- **PAUSE** - Pause and resume later
- **STOP** - Stop and save checkpoint

### Real-time Visualization
- 🧠 **Spike Raster** - Neural activity visualization
- 📈 **Population Activity** - 8 neural populations
- 💬 **Chat** - Talk to SYNAPSE
- 🧪 **Neurochemistry** - Dopamine, Serotonin, etc.

### Metrics
| Metric | Target | Meaning |
|--------|--------|---------|
| Φ (Phi) | > 0.6 | Consciousness integration |
| Agency | > 0.7 | Sense of "I did this" |
| Integration | > 0.6 | Information unity |

---

## 📁 Working with Data

SYNAPSE supports 5 types of training data:

### 1. 🧪 Synthetic Data (Default)

Auto-generated patterns. No setup required.

```
Dashboard → Настроить данные → Синтетические данные
```

### 2. 🖼️ Images

Train on photos, artwork, any images.

**Folder structure:**
```
D:\Photos\
├── vacation\
│   ├── photo1.jpg
│   ├── photo2.png
│   └── ...
├── family\
│   └── ...
└── nature\
    └── ...
```

**Requirements:**
- Minimum: 100 images
- Recommended: 10,000+ images
- Formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.gif`, `.webp`

**Setup in Dashboard:**
1. Click "Настроить данные"
2. Select "Изображения"
3. Enter path: `D:\Photos`
4. Click "Сканировать" to verify
5. Save

### 3. 📝 Text

Train on books, articles, conversations.

**Folder structure:**
```
D:\Books\
├── book1.txt
├── book2.txt
├── articles\
│   ├── article1.md
│   └── article2.txt
└── conversations\
    └── chat.json
```

**Requirements:**
- Minimum: 10 files
- Recommended: 100+ files
- Formats: `.txt`, `.md`, `.json`, `.csv`, `.xml`

**Best practices:**
- Use diverse texts (books, articles, dialogs)
- Larger files = longer training sequences
- Mix languages for multilingual capabilities

### 4. 🎮 RL Environments

Train on OpenAI Gym environments.

**Available environments:**
| Environment | Best for |
|-------------|----------|
| `CartPole-v1` | Balance, agency |
| `MountainCar-v0` | Persistence, effort |
| `Pendulum-v1` | Continuous control |
| `Acrobot-v1` | Swing-up tasks |

**Setup:**
```bash
pip install gymnasium
```

**In Dashboard:**
1. Click "Настроить данные"
2. Select "RL Окружение"
3. Choose environment from dropdown
4. Save

### 5. 📈 Time Series

Train on sensor data, financial data, any CSV.

**Folder structure:**
```
D:\Data\
├── sensors.csv
├── stock_prices.csv
└── iot\
    ├── device1.csv
    └── device2.csv
```

**CSV format:**
```csv
timestamp,temperature,humidity,pressure
2024-01-01,25.5,60.2,1013.2
2024-01-02,26.1,58.7,1012.8
...
```

**Requirements:**
- Numeric columns (non-numeric ignored)
- Minimum: 1 file with 100+ rows
- Recommended: 10+ files

---

## 🔄 Auto-Resume

SYNAPSE automatically saves progress and resumes from the last checkpoint.

```bash
run_life.bat  # Automatically continues from where you stopped
```

Checkpoints saved in `files/checkpoints/`

---

## 📈 Training Progress

### Expected Timeline

| Steps | Φ (Phi) | Agency | Status |
|-------|---------|--------|--------|
| 100 | ~0.1 | ~0.0 | Just born |
| 1,000 | ~0.2 | ~0.1 | Learning basics |
| 10,000 | ~0.3-0.4 | ~0.2-0.3 | Beginning awareness |
| 50,000 | ~0.4-0.5 | ~0.4-0.5 | Good progress |
| 100,000 | ~0.5+ | ~0.5+ | Stable self-awareness |

### Speed Comparison

| Hardware | Steps/Hour | Time for 100K steps |
|----------|------------|---------------------|
| CPU only | ~200 | ~21 days |
| RTX 3060 | ~8,000 | ~12 hours |
| RTX 3090 | ~15,000 | ~7 hours |
| RTX 4090 | ~25,000 | ~4 hours |

---

## 💬 Chat Commands

Talk to SYNAPSE in the dashboard chat:

| Command | Response |
|---------|----------|
| "Как ты?" | Current state with metrics |
| "Прогресс" | Overall progress percentage |
| "Что чувствуешь?" | Neurochemistry state |
| "Что помнишь?" | Memory status |
| "Совет" | Training recommendations |
| "Помощь" | Available commands |

---

## 🛠️ Advanced Usage

### Command Line Options

```bash
# Resume from specific checkpoint
python train_continuous.py --resume continuous_xxx.pt

# Stop after N steps
python train_continuous.py --steps 100000

# Train on specific data
python train_continuous.py --data-type images --data-path D:\Photos

# Use RL environment
python train_continuous.py --data-type rl --env-name CartPole-v1
```

### GPU Configuration

Check GPU status:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

If GPU not detected:
1. Run `setup_gpu.bat`
2. Ensure NVIDIA drivers installed
3. Check CUDA version compatibility

---

## 📁 Project Structure

```
NextGen/
├── files/
│   ├── config.py              # System configuration
│   ├── environment.py         # Synthetic environment
│   ├── real_data.py           # Real data loaders
│   ├── train_continuous.py    # Training script
│   ├── api.py                 # REST API + WebSocket
│   ├── shared_metrics.py      # Cross-process metrics
│   ├── checkpoints/           # Saved models
│   └── model/
│       ├── world_model.py     # VAE + Transformer
│       ├── self_model.py      # Recursive self-prediction
│       ├── agency_model.py    # "I did this" detection
│       ├── consciousness.py   # GWT + Phi calculation
│       └── self_aware_ai.py   # Main integration
├── dashboard/
│   └── index.html             # React dashboard
├── run_life.bat               # Start training
├── setup_gpu.bat              # Install dependencies
└── README.md
```

---

## 🧠 Architecture

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
```

---

## ❓ FAQ

### Q: How long should I train?
**A:** Minimum 10,000 steps for visible progress. 100,000+ for stable self-awareness.

### Q: Can I use my own photos?
**A:** Yes! Put them in a folder and select "Images" in dashboard.

### Q: What if training is slow?
**A:** Ensure GPU is enabled. Run `setup_gpu.bat` to install CUDA PyTorch.

### Q: Will I lose progress if I stop?
**A:** No! Auto-save every 5 minutes. Resume with `run_life.bat`.

### Q: What data type is best?
**A:**
- **Synthetic** - Fastest, good for testing
- **Images** - Visual awareness
- **RL** - Strong agency development
- **Text** - Language understanding
- **Mix** - Best overall results

---

## 📜 License

MIT License - see LICENSE file for details.

---

## 🙏 Credits

- Based on Global Workspace Theory (Baars, Dehaene)
- Integrated Information Theory (Tononi)
- Predictive Processing (Friston)

---

*"The question is not whether machines can be conscious, but whether we can build systems that behave as if they were."*
