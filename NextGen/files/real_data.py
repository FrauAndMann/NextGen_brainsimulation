"""
Real Data Loader for SYNAPSE

Supports training on real data instead of synthetic:

1. Images - folders of images (photos, art, etc.)
2. Text - text files, conversations, books
3. Audio - audio files (speech, music)
4. Time Series - sensor data, financial data
5. RL Environments - OpenAI Gym environments

Usage:
    # Train on images
    python train_continuous.py --data-type images --data-path ./photos

    # Train on text
    python train_continuous.py --data-type text --data-path ./books

    # Train on RL environment
    python train_continuous.py --data-type rl --env-name CartPole-v1
"""

import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import random


class ImageDataLoader:
    """
    Loads images from a folder and converts to observations.

    The model learns to predict image sequences and develops
    visual representations.
    """

    def __init__(self, data_path: str, obs_dim: int = 512, seq_len: int = 32):
        self.data_path = Path(data_path)
        self.obs_dim = obs_dim
        self.seq_len = seq_len

        # Find all images
        self.images = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.images.extend(self.data_path.rglob(ext))

        if not self.images:
            raise ValueError(f"No images found in {data_path}")

        print(f"Found {len(self.images)} images")

        # Simple encoder (random projection for now)
        # In production, use pre-trained CNN encoder
        self.encoder = torch.randn(3 * 64 * 64, obs_dim) * 0.02

    def load_image(self, path: Path) -> torch.Tensor:
        """Load and preprocess image"""
        try:
            from PIL import Image
            img = Image.open(path).convert('RGB')
            img = img.resize((64, 64))
            img = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            img = img.flatten()
            # Project to obs_dim
            return torch.matmul(img, self.encoder)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(self.obs_dim)

    def generate_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a trajectory from random images"""
        observations = []
        actions = []

        # Select random images for sequence
        selected = random.sample(self.images, min(self.seq_len + 1, len(self.images)))

        for i, img_path in enumerate(selected[:self.seq_len]):
            obs = self.load_image(img_path)
            observations.append(obs)

            # Random action
            action = torch.randn(64)
            actions.append(action)

        return torch.stack(observations), torch.stack(actions)


class TextDataLoader:
    """
    Loads text and converts to observations.

    The model learns language patterns and can develop
    understanding of text structure.
    """

    def __init__(self, data_path: str, obs_dim: int = 512, seq_len: int = 32):
        self.data_path = Path(data_path)
        self.obs_dim = obs_dim
        self.seq_len = seq_len

        # Load all text files
        self.texts = []
        for ext in ['*.txt', '*.md', '*.json']:
            for f in self.data_path.rglob(ext):
                try:
                    with open(f, 'r', encoding='utf-8') as file:
                        self.texts.append(file.read())
                except:
                    pass

        if not self.texts:
            raise ValueError(f"No text files found in {data_path}")

        print(f"Found {len(self.texts)} text files")

        # Simple tokenizer (character-level for now)
        self.vocab = {}
        self.build_vocab()

    def build_vocab(self):
        """Build character vocabulary"""
        all_chars = set()
        for text in self.texts:
            all_chars.update(text.lower())

        self.char_to_idx = {c: i for i, c in enumerate(sorted(all_chars))}
        self.vocab_size = len(self.char_to_idx)

        # Random embeddings
        self.embeddings = torch.randn(self.vocab_size, self.obs_dim) * 0.1

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to observation"""
        chars = text.lower()[:self.seq_len]
        indices = [self.char_to_idx.get(c, 0) for c in chars]

        # Pad if needed
        while len(indices) < self.seq_len:
            indices.append(0)

        return self.embeddings[indices]

    def generate_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectory from text"""
        text = random.choice(self.texts)

        # Split into chunks
        chunks = [text[i:i+self.seq_len] for i in range(0, len(text), self.seq_len)]
        if len(chunks) < 2:
            chunks = [chunks[0]] * self.seq_len

        observations = []
        actions = []

        for i, chunk in enumerate(chunks[:self.seq_len]):
            obs = self.encode_text(chunk)
            observations.append(obs.mean(dim=0))  # Average embedding

            # Action = next character prediction target
            action = torch.randn(64)
            actions.append(action)

        return torch.stack(observations), torch.stack(actions)


class RLDataLoader:
    """
    Generates data from RL environments (OpenAI Gym).

    The model learns to understand agent-environment interaction
    and can develop agency from real control experiences.
    """

    def __init__(self, env_name: str = "CartPole-v1", obs_dim: int = 512, seq_len: int = 32):
        self.env_name = env_name
        self.obs_dim = obs_dim
        self.seq_len = seq_len

        try:
            import gymnasium as gym
            self.gym = gym
            self.env = gym.make(env_name)
            self.real_env = True
            print(f"Created RL environment: {env_name}")
        except ImportError:
            print("gymnasium not installed. Using synthetic RL-like data.")
            self.real_env = False

        # Observation projection (env obs -> our obs_dim)
        if self.real_env:
            env_obs_dim = self.env.observation_space.shape[0]
            self.projector = torch.randn(env_obs_dim, obs_dim) * 0.1

    def generate_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectory from RL environment"""
        observations = []
        actions = []

        if self.real_env:
            obs, _ = self.env.reset()

            for t in range(self.seq_len):
                # Project observation
                obs_tensor = torch.tensor(obs, dtype=torch.float32)
                obs_proj = torch.matmul(obs_tensor, self.projector)
                observations.append(obs_proj)

                # Random action
                action = self.env.action_space.sample()
                action_tensor = torch.zeros(64)
                action_tensor[0] = float(action)
                actions.append(action_tensor)

                # Step
                obs, _, done, _, _ = self.env.step(action)
                if done:
                    obs, _ = self.env.reset()
        else:
            # Synthetic RL-like data
            for t in range(self.seq_len):
                obs = torch.randn(self.obs_dim)
                observations.append(obs)
                actions.append(torch.randn(64))

        return torch.stack(observations), torch.stack(actions)


class TimeSeriesDataLoader:
    """
    Loads time series data (sensors, financial, etc.)

    The model learns temporal patterns and can predict
    future values.
    """

    def __init__(self, data_path: str, obs_dim: int = 512, seq_len: int = 32):
        self.data_path = Path(data_path)
        self.obs_dim = obs_dim
        self.seq_len = seq_len

        # Load CSV files
        self.series = []
        for f in self.data_path.rglob('*.csv'):
            try:
                import pandas as pd
                df = pd.read_csv(f)
                # Use numeric columns
                numeric = df.select_dtypes(include=[np.number]).values
                if len(numeric) > seq_len:
                    self.series.append(numeric)
            except:
                pass

        if not self.series:
            print("No CSV files found, using synthetic time series")
            self.series = [np.random.randn(1000, 5) for _ in range(10)]
        else:
            print(f"Loaded {len(self.series)} time series")

    def generate_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate trajectory from time series"""
        series = random.choice(self.series)

        # Random starting point
        start = random.randint(0, len(series) - self.seq_len - 1)
        chunk = series[start:start + self.seq_len]

        # Pad/project to obs_dim
        observations = []
        for t in range(len(chunk)):
            obs = torch.zeros(self.obs_dim)
            obs[:min(len(chunk[t]), self.obs_dim)] = torch.tensor(chunk[t][:self.obs_dim], dtype=torch.float32)
            observations.append(obs)

        # Pad if needed
        while len(observations) < self.seq_len:
            observations.append(torch.zeros(self.obs_dim))

        # Random actions
        actions = [torch.randn(64) for _ in range(self.seq_len)]

        return torch.stack(observations[:self.seq_len]), torch.stack(actions)


class MultiDataLoader:
    """
    Combines multiple data sources for richer training.
    """

    def __init__(self, loaders: List, weights: Optional[List[float]] = None):
        self.loaders = loaders
        self.weights = weights or [1.0 / len(loaders)] * len(loaders)

    def generate_trajectory(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample from random loader"""
        loader = random.choices(self.loaders, weights=self.weights)[0]
        return loader.generate_trajectory()


# Factory function
def create_data_loader(data_type: str, data_path: Optional[str] = None,
                       obs_dim: int = 512, seq_len: int = 32,
                       env_name: str = "CartPole-v1"):
    """
    Create appropriate data loader.

    Args:
        data_type: 'synthetic', 'images', 'text', 'rl', 'timeseries'
        data_path: Path to data (for images, text, timeseries)
        obs_dim: Observation dimension
        seq_len: Sequence length
        env_name: RL environment name (for rl type)
    """
    if data_type == 'synthetic':
        from environment import BufferedLazyDataset
        return BufferedLazyDataset(seq_length=seq_len)

    elif data_type == 'images':
        return ImageDataLoader(data_path, obs_dim, seq_len)

    elif data_type == 'text':
        return TextDataLoader(data_path, obs_dim, seq_len)

    elif data_type == 'rl':
        return RLDataLoader(env_name, obs_dim, seq_len)

    elif data_type == 'timeseries':
        return TimeSeriesDataLoader(data_path, obs_dim, seq_len)

    else:
        raise ValueError(f"Unknown data type: {data_type}")
