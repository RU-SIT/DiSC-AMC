# Discrete LLM AMC - Repository Summary & Developer Guide

## 🎯 Project Purpose

This repository implements a **DINO-based embedding pipeline** for Automatic Modulation Classification (AMC). The primary goal is to:

1. **Generate embeddings** from signal spectrogram/constellation images
2. **Reduce dimensionality** using PCA
3. **Discretize** the reduced embeddings into bins
4. **Output** the results as `{noise_mode}_{n_bins}_{top_k}_data.pkl` files (currently outputs JSON format)

### Key Output Files
- **Format**: `{noise_mode}_{n_bins}_{top_k}_data.pkl` (mentioned but currently outputs JSON as `dino_features.json`)
- **Contents**: Discretized feature vectors for each image, organized by feature type (noisy/noiseless)

---

## 📁 Current Repository Structure

```
discrete-llm-amc/
└── src/
    └── data/
        └── embedding_pipeline.py  # Main pipeline implementation
```

**Note**: The repository appears incomplete. Expected but missing directories:
- `src/evaluation/` - Should contain evaluation scripts (e.g., `gemini_googleai.py`)
- `src/spectrogram/` - Should contain model definitions (e.g., `autoencoder_vit.py`)
- `data/own/` - Data storage directory
- `exp/` - Model weights directory
- Tests directory
- Configuration files (requirements.txt, setup.py, etc.)

---

## 🔧 Core Components

### 1. `embedding_pipeline.py` - Main Pipeline

**Purpose**: End-to-end pipeline for extracting and discretizing DINO embeddings

**Key Functions**:

| Function | Purpose | Usage |
|----------|---------|-------|
| `extract_label()` | Extract modulation class from filename | Called on image filenames |
| `load_encoder()` | Load pre-trained DINO model | Initialize encoder from checkpoint |
| `extract_embeddings()` | Process images through encoder | Batch process all images in directory |
| `_reduce_and_discretize()` | Apply PCA + discretization | Transform embeddings to discrete bins |
| `process_split()` | Process train/test/val split | Main processing loop |
| `run_pipeline()` | End-to-end execution | CLI entry point |
| `save_results()` / `load_results()` | I/O operations | Persist/load JSON data |

**Modulation Classes Supported**:
```python
["OOK", "4ASK", "8ASK", "OQPSK", "CPFSK", "GFSK", "4PAM", "DQPSK", "16PAM", "GMSK"]
```

**Feature Types**:
- `noisyImg` - Images with noise
- `noiseLessImg` - Clean images

---

## 🚀 How to Use (Quick Start)

### Prerequisites
```bash
pip install torch torchvision scikit-learn pillow numpy tqdm
```

### Running the Pipeline

```bash
python src/data/embedding_pipeline.py \
    --dataset_folder unlabeled_10k \
    --weights ../exp/dino_classifier.pth \
    --n_components 10 \
    --n_bins 10 \
    --strategy uniform \
    --batch_size 32
```

### Parameters Explained

- `--dataset_folder`: Name of dataset subfolder (default: `unlabeled_10k`)
- `--weights`: Path to pre-trained model weights (`.pth` file)
- `--data_root`: Root directory for data (default: `../data/own`)
- `--n_components`: Number of PCA components (default: 10)
- `--n_bins`: Number of discretization bins (default: 10)
- `--strategy`: Binning strategy - `uniform`, `quantile`, or `kmeans` (default: `uniform`)
- `--batch_size`: Images per batch (default: 32)

### Expected Directory Structure

```
data/own/unlabeled_10k/
├── train/
│   ├── noisyImg/
│   │   └── *.png
│   └── noiseLessImg/
│       └── *.png
└── test/
    ├── noisyImg/
    │   └── *.png
    └── noiseLessImg/
        └── *.png
```

### Output Format

JSON file: `data/own/unlabeled_10k/dino_features.json`

```json
{
  "16PAM_0.01dB__0701_20250627_153146.png": {
    "noisyImg": [3, 7, 2, 9, 1, 5, 8, 4, 6, 0],
    "noiseLessImg": [4, 6, 3, 8, 2, 4, 7, 5, 3, 1]
  },
  ...
}
```

---

## 🎨 Modifying Prompting Strategies

### Current Workflow
1. Images → DINO Encoder → Embeddings
2. Embeddings → PCA → Reduced vectors
3. Reduced vectors → KBinsDiscretizer → Discrete bins

### Where to Modify

#### **Option 1: Change Discretization Strategy**
**File**: `embedding_pipeline.py`
**Function**: `_reduce_and_discretize()` (line 144)

```python
# Current strategies: 'uniform', 'quantile', 'kmeans'
discretizer = KBinsDiscretizer(
    n_bins=n_bins, 
    encode="ordinal",  # Change to "onehot" for one-hot encoding
    strategy=strategy,  # Change strategy here
)
```

**To add custom binning**:
```python
# Add after line 173
if strategy == "custom":
    # Your custom binning logic
    bins = your_custom_function(reduced)
    discretized = bins.astype(int)
else:
    discretized = discretizer.transform(reduced).astype(int)
```

#### **Option 2: Modify Image Preprocessing**
**File**: `embedding_pipeline.py`
**Constant**: `_DEFAULT_TRANSFORM` (line 53)

```python
# Add data augmentation or different preprocessing
_DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Add augmentation
    transforms.ColorJitter(brightness=0.2),  # Add jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),  # Add normalization
])
```

#### **Option 3: Add New Feature Types**
**File**: `embedding_pipeline.py`
**Constant**: `FEATURE_TYPES` (line 51)

```python
# Add new feature types
FEATURE_TYPES: List[str] = [
    "noisyImg", 
    "noiseLessImg",
    "augmentedImg",  # New type
    "filteredImg",   # New type
]
```

Then create corresponding directories in your dataset folder.

#### **Option 4: Change PCA to Other Dimensionality Reduction**
**File**: `embedding_pipeline.py`
**Function**: `_reduce_and_discretize()` (line 160)

```python
# Replace PCA with UMAP, t-SNE, or Autoencoder
from umap import UMAP

if pca is None:
    reducer = UMAP(n_components=n_components, random_state=42)
    reduced = reducer.fit_transform(embeddings)
else:
    reduced = reducer.transform(embeddings)
```

#### **Option 5: Output PKL Instead of JSON**
**File**: `embedding_pipeline.py`
**Function**: `save_results()` (line 254)

```python
import pickle

def save_results(results, output_path, noise_mode="uniform", n_bins=10, top_k=10):
    # Change file extension
    output_path = output_path.replace(".json", ".pkl")
    # Use naming convention from requirements
    output_path = f"{noise_mode}_{n_bins}_{top_k}_data.pkl"
    
    with open(output_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"Saved {len(results)} entries → {output_path}")
```

---

## 🔍 Code Review & Quality Analysis

### ✅ Strengths

1. **Well-Documented**: Comprehensive docstrings and inline comments
2. **SOLID Principles**: 
   - Single Responsibility: Each function does one thing
   - Open/Closed: Parameterized for extension
3. **DRY Compliance**: Reusable helper functions
4. **Type Hints**: Good use of type annotations
5. **Error Handling**: Basic file existence checks

### ⚠️ Code Smells & Issues

#### **1. Hard-Coded Constants**
**Location**: Lines 46-51
```python
CLASSES: List[str] = ["OOK", "4ASK", ...]  # Hard-coded
FEATURE_TYPES: List[str] = ["noisyImg", "noiseLessImg"]  # Hard-coded
```

**Issue**: Cannot easily extend without code modification

**Fix**: Move to configuration file
```python
# config.yaml
classes:
  - OOK
  - 4ASK
  ...
feature_types:
  - noisyImg
  - noiseLessImg
```

#### **2. Missing Error Handling**
**Location**: Multiple functions

**Issues**:
- `load_encoder()` (line 77): No validation of weights file existence
- `extract_embeddings()` (line 106): No handling of corrupted images
- `_load_images_as_batch()` (line 100): No try-catch for PIL.Image.open()

**Fix Example**:
```python
def _load_images_as_batch(paths, transform):
    images = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(transform(img))
        except (IOError, OSError) as e:
            print(f"Warning: Failed to load {p}: {e}")
            # Use a black image as placeholder or skip
            images.append(torch.zeros(3, 224, 224))
    return torch.stack(images)
```

#### **3. Resource Management**
**Location**: `extract_embeddings()` (line 110)

**Issue**: No explicit cleanup of loaded images (potential memory leak for large datasets)

**Fix**:
```python
def _load_images_as_batch(paths, transform):
    images = []
    for p in paths:
        with Image.open(p) as img:  # Use context manager
            images.append(transform(img.convert("RGB")))
    return torch.stack(images)
```

#### **4. Magic Numbers**
**Location**: Throughout

**Issues**:
- Line 54: `transforms.Resize((224, 224))` - hard-coded image size
- Line 360: `nn.LayerNorm(768)` - hard-coded embedding dimension

**Fix**: Use configuration constants
```python
IMAGE_SIZE = 224
EMBEDDING_DIM = 768
```

#### **5. Circular Dependency Risk**
**Location**: Lines 292-296

**Issue**: Requires `model_class` parameter but raises error if None. The CLI script imports DinoClassifier at runtime (line 346).

**Fix**: Move model definition to separate file
```python
# models/dino_classifier.py
class DinoClassifier(nn.Module):
    ...
```

#### **6. No Input Validation**
**Location**: `run_pipeline()` (line 272)

**Issues**:
- No check if `n_bins` > 0
- No check if `n_components` > 0
- No check if `batch_size` > 0

**Fix**:
```python
def run_pipeline(...):
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")
    if n_components <= 0:
        raise ValueError(f"n_components must be positive, got {n_components}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    ...
```

#### **7. No Logging**
**Issue**: Uses `print()` statements instead of proper logging

**Fix**:
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Replace print() with logger.info()
logger.info(f"Saved {len(results)} entries → {output_path}")
```

#### **8. No Unit Tests**
**Issue**: No test coverage

**Fix**: Add tests
```python
# tests/test_embedding_pipeline.py
import pytest
from src.data.embedding_pipeline import extract_label

def test_extract_label():
    filename = "16PAM_0.01dB__0701_20250627_153146.png"
    assert extract_label(filename) == "16PAM"
```

#### **9. No Progress Persistence**
**Issue**: If pipeline crashes, must restart from beginning

**Fix**: Add checkpointing
```python
import os
import pickle

def save_checkpoint(state, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, "latest.pkl"), "wb") as f:
        pickle.dump(state, f)

def load_checkpoint(checkpoint_dir="checkpoints"):
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pkl")
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)
    return None
```

#### **10. Inconsistent Output Format**
**Issue**: Documentation mentions `.pkl` files but code outputs `.json`

**Location**: Line 289, function `run_pipeline()`

---

## 🐛 Potential Bugs

### **Bug 1: Out of Memory with Large Datasets**
**Location**: `extract_embeddings()` line 136-138
```python
with torch.no_grad():
    emb = encoder(batch.to(device))  # (B, latent_dim)
parts.append(emb.cpu().numpy())  # Accumulates in memory
```

**Impact**: For large datasets, `parts` list can exhaust memory

**Fix**:
```python
# Stream to disk instead
import h5py

with h5py.File("temp_embeddings.h5", "w") as f:
    embeddings_dataset = f.create_dataset(
        "embeddings", 
        shape=(len(paths), latent_dim),
        dtype="float32"
    )
    idx = 0
    for start in range(0, len(paths), batch_size):
        batch = _load_images_as_batch(paths[start:start+batch_size])
        with torch.no_grad():
            emb = encoder(batch.to(device))
        embeddings_dataset[idx:idx+len(emb)] = emb.cpu().numpy()
        idx += len(emb)
```

### **Bug 2: Race Condition in Directory Creation**
**Location**: `save_results()` line 259
```python
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
```

**Impact**: In multi-process scenarios, could fail

**Fix**: Add exception handling
```python
try:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise
```

### **Bug 3: Silent Failure on Empty Directory**
**Location**: `extract_embeddings()` line 126
```python
if not paths:
    raise FileNotFoundError(f"No .png images in {image_dir}")
```

**Good!** This is actually handled correctly.

### **Bug 4: No Validation of PCA Components**
**Location**: `_reduce_and_discretize()` line 161

**Issue**: If `n_components` > number of features, PCA will fail

**Fix**:
```python
def _reduce_and_discretize(embeddings, n_components, ...):
    n_features = embeddings.shape[1]
    actual_components = min(n_components, n_features)
    if actual_components != n_components:
        logger.warning(
            f"Requested {n_components} components but only {n_features} features available. "
            f"Using {actual_components} components."
        )
    pca = PCA(n_components=actual_components)
    ...
```

---

## 🔒 Security Analysis

### **Security Issue 1: Path Traversal Vulnerability**
**Location**: Multiple file operations
```python
image_dir = os.path.join(split_path, ft)  # ft is user-controlled
```

**Risk**: If `ft` contains `../`, could access files outside intended directory

**Fix**:
```python
import os.path

def safe_join(base, *paths):
    """Join paths and ensure result is within base directory."""
    final_path = os.path.abspath(os.path.join(base, *paths))
    if not final_path.startswith(os.path.abspath(base)):
        raise ValueError("Path traversal detected")
    return final_path

image_dir = safe_join(split_path, ft)
```

### **Security Issue 2: Pickle Deserialization**
**Location**: Line 91 (torch.load)
```python
state = torch.load(weights_path, map_location="cpu")
```

**Risk**: Loading untrusted `.pth` files can execute arbitrary code

**Fix**:
```python
# Use weights_only parameter (PyTorch 1.13+)
state = torch.load(weights_path, map_location="cpu", weights_only=True)

# Or validate source
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")
if not weights_path.endswith((".pth", ".pt")):
    raise ValueError("Invalid weights file extension")
```

### **Security Issue 3: No Input Sanitization**
**Location**: CLI arguments (line 334-341)

**Risk**: Command-line injection if paths contain special characters

**Fix**:
```python
import re

def sanitize_path(path):
    """Remove potentially dangerous characters from path."""
    if not re.match(r'^[\w\-./]+$', path):
        raise ValueError(f"Invalid characters in path: {path}")
    return path

args.dataset_folder = sanitize_path(args.dataset_folder)
```

---

## ⚡ Performance Optimizations

### **Optimization 1: Batch Size Tuning**
**Current**: Fixed batch size of 32

**Improvement**: Auto-tune based on available GPU memory
```python
def get_optimal_batch_size(encoder, device, image_size=224):
    """Determine optimal batch size based on available memory."""
    if device.type == "cpu":
        return 32
    
    # Get available GPU memory
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        total_memory = props.total_memory
        # Estimate: each 224x224 image uses ~0.6MB
        safe_batch_size = int((total_memory * 0.5) / (0.6 * 1024 * 1024))
        return min(safe_batch_size, 256)  # Cap at 256
    return 32
```

### **Optimization 2: Mixed Precision Training**
**Location**: `extract_embeddings()` line 136

**Improvement**: Use automatic mixed precision for faster inference
```python
from torch.cuda.amp import autocast

with torch.no_grad(), autocast():
    emb = encoder(batch.to(device))
```

### **Optimization 3: Multi-GPU Support**
**Improvement**: Distribute across multiple GPUs
```python
if torch.cuda.device_count() > 1:
    encoder = nn.DataParallel(encoder)
```

### **Optimization 4: Data Loading Parallelism**
**Current**: Sequential image loading

**Improvement**: Use DataLoader with multiple workers
```python
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.paths[idx]

def extract_embeddings_fast(encoder, device, image_dir, batch_size=32):
    paths = sorted(glob(os.path.join(image_dir, "*.png")))
    dataset = ImageDataset(paths, _DEFAULT_TRANSFORM)
    loader = DataLoader(dataset, batch_size=batch_size, 
                       num_workers=4, pin_memory=True)
    
    embeddings = []
    filenames = []
    for batch_imgs, batch_paths in tqdm(loader):
        with torch.no_grad():
            emb = encoder(batch_imgs.to(device))
        embeddings.append(emb.cpu().numpy())
        filenames.extend([os.path.basename(p) for p in batch_paths])
    
    return filenames, np.concatenate(embeddings)
```

### **Optimization 5: Caching PCA Results**
**Improvement**: Cache PCA transformations to avoid recomputation
```python
import hashlib
import pickle

def get_cache_key(embeddings, n_components):
    """Generate cache key from embeddings hash."""
    emb_hash = hashlib.md5(embeddings.tobytes()).hexdigest()
    return f"pca_{emb_hash}_{n_components}.pkl"

def _reduce_and_discretize_cached(embeddings, n_components, ...):
    cache_key = get_cache_key(embeddings, n_components)
    cache_path = os.path.join("cache", cache_key)
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    
    result = _reduce_and_discretize(embeddings, n_components, ...)
    
    os.makedirs("cache", exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    
    return result
```

---

## 🧹 Cleanup Recommendations (SOLID, DRY, KISS)

### **Priority 1: High Impact**

#### 1. **Separate Concerns** (SOLID - Single Responsibility)
**Current**: All functionality in one file

**Recommendation**: Split into modules
```
src/
├── models/
│   ├── __init__.py
│   ├── dino_classifier.py      # Model definitions
│   └── encoder_factory.py      # Model loading logic
├── data/
│   ├── __init__.py
│   ├── image_loader.py         # Image loading & preprocessing
│   ├── embedding_extractor.py  # Embedding extraction
│   └── transformers.py         # PCA & discretization
├── utils/
│   ├── __init__.py
│   ├── io.py                   # File I/O operations
│   ├── config.py               # Configuration management
│   └── logging.py              # Logging setup
└── pipeline.py                 # Main orchestration
```

#### 2. **Configuration Management** (Open/Closed Principle)
**Current**: Hard-coded constants

**Recommendation**: Use configuration files
```yaml
# config/default.yaml
data:
  root: ../data/own
  dataset: unlabeled_10k
  feature_types:
    - noisyImg
    - noiseLessImg

model:
  weights: ../exp/dino_classifier.pth
  image_size: 224
  embedding_dim: 768

pipeline:
  n_components: 10
  n_bins: 10
  strategy: uniform
  batch_size: 32

classes:
  - OOK
  - 4ASK
  - 8ASK
  - OQPSK
  - CPFSK
  - GFSK
  - 4PAM
  - DQPSK
  - 16PAM
  - GMSK
```

Load with:
```python
import yaml

class Config:
    def __init__(self, config_path="config/default.yaml"):
        with open(config_path) as f:
            self._config = yaml.safe_load(f)
    
    def __getattr__(self, key):
        return self._config.get(key)

config = Config()
```

#### 3. **Add Dependency Injection** (Dependency Inversion Principle)
**Current**: Hard dependencies on specific classes

**Recommendation**: Use interfaces
```python
from abc import ABC, abstractmethod

class Reducer(ABC):
    @abstractmethod
    def fit_transform(self, X): pass
    
    @abstractmethod
    def transform(self, X): pass

class PCAReducer(Reducer):
    def __init__(self, n_components):
        self.pca = PCA(n_components=n_components)
    
    def fit_transform(self, X):
        return self.pca.fit_transform(X)
    
    def transform(self, X):
        return self.pca.transform(X)

class UAMPReducer(Reducer):
    def __init__(self, n_components):
        self.umap = UMAP(n_components=n_components)
    
    def fit_transform(self, X):
        return self.umap.fit_transform(X)
    
    def transform(self, X):
        return self.umap.transform(X)

# Use
def process_split(encoder, device, reducer: Reducer, ...):
    reduced = reducer.fit_transform(embeddings)
    ...
```

#### 4. **Add Comprehensive Tests**
```
tests/
├── __init__.py
├── conftest.py                 # Pytest fixtures
├── test_image_loader.py
├── test_embedding_extractor.py
├── test_transformers.py
├── test_pipeline.py
└── test_integration.py         # End-to-end tests
```

Example test:
```python
# tests/test_transformers.py
import pytest
import numpy as np
from src.data.transformers import _reduce_and_discretize

def test_reduce_and_discretize():
    # Arrange
    embeddings = np.random.randn(100, 768)
    n_components = 10
    n_bins = 5
    
    # Act
    discretized, pca, disc = _reduce_and_discretize(
        embeddings, n_components, n_bins, "uniform"
    )
    
    # Assert
    assert discretized.shape == (100, 10)
    assert discretized.min() >= 0
    assert discretized.max() < n_bins
    assert isinstance(discretized, np.ndarray)
    assert discretized.dtype == int
```

#### 5. **Add Proper Error Handling**
**Pattern**: Use custom exceptions
```python
# utils/exceptions.py
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass

class DataNotFoundError(PipelineError):
    """Raised when required data is not found."""
    pass

class ModelLoadError(PipelineError):
    """Raised when model fails to load."""
    pass

class TransformError(PipelineError):
    """Raised when transformation fails."""
    pass

# Usage
def load_encoder(model_class, weights_path, device=None):
    if not os.path.exists(weights_path):
        raise ModelLoadError(f"Weights not found: {weights_path}")
    
    try:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model = model_class()
        model.load_state_dict(state, strict=False)
        return model.encoder.to(device).eval(), device
    except Exception as e:
        raise ModelLoadError(f"Failed to load model: {e}") from e
```

### **Priority 2: Medium Impact**

#### 6. **Add Progress Tracking & Resumability**
```python
# utils/checkpoint.py
class CheckpointManager:
    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save(self, state, name="latest"):
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(state, f)
    
    def load(self, name="latest"):
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        if not os.path.exists(path):
            return None
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def exists(self, name="latest"):
        path = os.path.join(self.checkpoint_dir, f"{name}.pkl")
        return os.path.exists(path)
```

#### 7. **Add Logging Infrastructure**
```python
# utils/logging.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging(log_file="pipeline.log", level=logging.INFO):
    """Setup logging with both file and console handlers."""
    logger = logging.getLogger("discrete_llm_amc")
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler (rotating)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5
    )
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger
```

#### 8. **Add Data Validation**
```python
# utils/validators.py
def validate_image_directory(path, feature_types):
    """Validate that directory structure is correct."""
    errors = []
    
    if not os.path.isdir(path):
        errors.append(f"Directory not found: {path}")
        return errors
    
    for ft in feature_types:
        ft_path = os.path.join(path, ft)
        if not os.path.isdir(ft_path):
            errors.append(f"Feature type directory not found: {ft_path}")
            continue
        
        images = glob(os.path.join(ft_path, "*.png"))
        if not images:
            errors.append(f"No PNG images in: {ft_path}")
    
    return errors

def validate_config(config):
    """Validate configuration parameters."""
    errors = []
    
    if config.n_bins <= 0:
        errors.append(f"n_bins must be positive, got {config.n_bins}")
    
    if config.n_components <= 0:
        errors.append(f"n_components must be positive, got {config.n_components}")
    
    if config.batch_size <= 0:
        errors.append(f"batch_size must be positive, got {config.batch_size}")
    
    if config.strategy not in ["uniform", "quantile", "kmeans"]:
        errors.append(f"Invalid strategy: {config.strategy}")
    
    return errors
```

#### 9. **Add Metrics & Monitoring**
```python
# utils/metrics.py
from dataclasses import dataclass, field
from time import time
from typing import Dict

@dataclass
class PipelineMetrics:
    """Track pipeline performance metrics."""
    start_time: float = field(default_factory=time)
    images_processed: int = 0
    batches_processed: int = 0
    errors: int = 0
    warnings: int = 0
    
    def __post_init__(self):
        self.stage_times: Dict[str, float] = {}
    
    def record_stage(self, stage_name: str, duration: float):
        self.stage_times[stage_name] = duration
    
    def get_throughput(self):
        """Images per second."""
        elapsed = time() - self.start_time
        return self.images_processed / elapsed if elapsed > 0 else 0
    
    def get_summary(self):
        """Return human-readable summary."""
        elapsed = time() - self.start_time
        return {
            "total_time": f"{elapsed:.2f}s",
            "images_processed": self.images_processed,
            "throughput": f"{self.get_throughput():.2f} images/s",
            "errors": self.errors,
            "warnings": self.warnings,
            "stage_times": {k: f"{v:.2f}s" for k, v in self.stage_times.items()},
        }
```

### **Priority 3: Nice to Have**

#### 10. **Add Documentation Generation**
```bash
pip install sphinx sphinx-rtd-theme
sphinx-quickstart docs
```

Add docstrings in NumPy style:
```python
def extract_embeddings(encoder, device, image_dir, batch_size=32):
    """Extract embeddings from images in a directory.
    
    Parameters
    ----------
    encoder : nn.Module
        Pre-trained encoder model
    device : torch.device
        Device to run inference on
    image_dir : str
        Path to directory containing PNG images
    batch_size : int, optional
        Number of images per batch, by default 32
    
    Returns
    -------
    filenames : List[str]
        List of base filenames in sorted order
    embeddings : np.ndarray
        Array of shape (N, latent_dim) containing embeddings
    
    Raises
    ------
    FileNotFoundError
        If no PNG images found in directory
    
    Examples
    --------
    >>> encoder, device = load_encoder(DinoClassifier, "model.pth")
    >>> filenames, embeddings = extract_embeddings(encoder, device, "data/train")
    >>> print(embeddings.shape)
    (1000, 768)
    """
```

#### 11. **Add CLI Improvements**
```python
# Use Click for better CLI
import click

@click.command()
@click.option('--config', type=click.Path(exists=True), help='Config file path')
@click.option('--dataset', default='unlabeled_10k', help='Dataset folder name')
@click.option('--weights', type=click.Path(exists=True), required=True, help='Model weights')
@click.option('--n-components', default=10, help='PCA components')
@click.option('--n-bins', default=10, help='Discretization bins')
@click.option('--verbose', is_flag=True, help='Enable verbose logging')
def main(config, dataset, weights, n_components, n_bins, verbose):
    """Run the DINO embedding pipeline."""
    if verbose:
        setup_logging(level=logging.DEBUG)
    else:
        setup_logging(level=logging.INFO)
    
    run_pipeline(
        dataset_folder=dataset,
        weights_path=weights,
        n_components=n_components,
        n_bins=n_bins,
    )

if __name__ == "__main__":
    main()
```

#### 12. **Add Docker Support**
```dockerfile
# Dockerfile
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY config/ ./config/

ENTRYPOINT ["python", "-m", "src.pipeline"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  pipeline:
    build: .
    volumes:
      - ./data:/app/data
      - ./exp:/app/exp
      - ./output:/app/output
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: --dataset unlabeled_10k --weights /app/exp/dino_classifier.pth
```

---

## 📊 Recommended New Structure

```
discrete-llm-amc/
├── README.md
├── REPOSITORY_SUMMARY.md (this file)
├── requirements.txt
├── setup.py
├── .gitignore
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── config/
│   ├── default.yaml
│   ├── dev.yaml
│   └── prod.yaml
│
├── src/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract base classes
│   │   ├── dino_classifier.py     # DINO model
│   │   ├── autoencoder_vit.py     # Autoencoder
│   │   └── encoder_factory.py     # Model loading
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── image_dataset.py       # Dataset classes
│   │   ├── image_loader.py        # Image loading
│   │   ├── embedding_extractor.py # Embedding extraction
│   │   └── transformers.py        # PCA, discretization
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── gemini_googleai.py     # LLM evaluation
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── prompts.py             # Prompt templates
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py                  # File I/O
│   │   ├── config.py              # Config management
│   │   ├── logging.py             # Logging setup
│   │   ├── validators.py          # Input validation
│   │   ├── exceptions.py          # Custom exceptions
│   │   ├── checkpoint.py          # Checkpointing
│   │   └── metrics.py             # Performance tracking
│   │
│   ├── pipeline.py                # Main orchestration
│   └── cli.py                     # CLI interface
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── fixtures/
│   │   └── sample_images/
│   ├── unit/
│   │   ├── test_image_loader.py
│   │   ├── test_embedding_extractor.py
│   │   ├── test_transformers.py
│   │   └── test_models.py
│   ├── integration/
│   │   ├── test_pipeline.py
│   │   └── test_cli.py
│   └── performance/
│       └── test_benchmark.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_analysis.ipynb
│   └── 03_results_visualization.ipynb
│
├── scripts/
│   ├── download_data.sh
│   ├── train_model.sh
│   └── run_evaluation.sh
│
├── docs/
│   ├── conf.py
│   ├── index.rst
│   ├── installation.rst
│   ├── quickstart.rst
│   └── api/
│
├── data/                          # Gitignored
│   └── own/
│       └── unlabeled_10k/
│
├── exp/                           # Gitignored
│   └── dino_classifier.pth
│
├── output/                        # Gitignored
│   ├── logs/
│   ├── checkpoints/
│   └── results/
│
└── cache/                         # Gitignored
```

---

## 🚀 Migration Path

### Phase 1: Critical Fixes (Week 1)
1. Add error handling to image loading
2. Add input validation
3. Fix security issues (path traversal, pickle)
4. Add proper logging
5. Create requirements.txt

### Phase 2: Restructuring (Week 2-3)
1. Split code into modules
2. Create configuration system
3. Add checkpoint support
4. Implement proper exception hierarchy
5. Add basic tests

### Phase 3: Enhancement (Week 4-6)
1. Add full test coverage
2. Implement dependency injection
3. Add performance optimizations
4. Create documentation
5. Add Docker support

### Phase 4: Advanced Features (Week 7-8)
1. Add monitoring & metrics
2. Implement multiple prompting strategies
3. Add evaluation pipeline
4. Create visualization tools
5. Performance tuning

---

## 🎯 Quick Wins (Start Here)

These changes provide maximum benefit with minimal effort:

1. **Create `requirements.txt`**
   ```txt
   torch>=2.0.0
   torchvision>=0.15.0
   scikit-learn>=1.3.0
   pillow>=10.0.0
   numpy>=1.24.0
   tqdm>=4.65.0
   pyyaml>=6.0
   ```

2. **Add `.gitignore`**
   ```
   __pycache__/
   *.py[cod]
   *$py.class
   *.so
   .Python
   env/
   venv/
   *.pth
   *.pkl
   *.h5
   data/
   exp/
   output/
   cache/
   logs/
   .DS_Store
   .vscode/
   .idea/
   ```

3. **Add input validation** (5 minutes)
   ```python
   # Add at start of run_pipeline()
   if n_bins <= 0:
       raise ValueError(f"n_bins must be positive, got {n_bins}")
   if n_components <= 0:
       raise ValueError(f"n_components must be positive, got {n_components}")
   ```

4. **Add proper logging** (10 minutes)
   ```python
   # Add at top of file
   import logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(levelname)s - %(message)s'
   )
   logger = logging.getLogger(__name__)
   
   # Replace all print() with logger.info()
   ```

5. **Add error handling to image loading** (15 minutes)
   ```python
   def _load_images_as_batch(paths, transform):
       images = []
       for p in paths:
           try:
               img = Image.open(p).convert("RGB")
               images.append(transform(img))
           except Exception as e:
               logger.warning(f"Failed to load {p}: {e}")
               # Skip or use placeholder
       return torch.stack(images) if images else torch.empty(0)
   ```

---

## 🤝 Contributing Guidelines

### Code Style
- Follow PEP 8
- Use type hints
- Write docstrings (NumPy style)
- Maximum line length: 88 characters (Black formatter)

### Commit Messages
```
<type>(<scope>): <subject>

<body>

<footer>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`

Example:
```
feat(data): add UMAP dimensionality reduction option

- Add UAMPReducer class implementing Reducer interface
- Update config to support reducer selection
- Add tests for UMAP reducer

Closes #42
```

### Pull Request Process
1. Create feature branch from `main`
2. Write tests for new functionality
3. Update documentation
4. Run tests and linting
5. Submit PR with clear description

---

## 📚 Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **scikit-learn User Guide**: https://scikit-learn.org/stable/user_guide.html
- **SOLID Principles**: https://en.wikipedia.org/wiki/SOLID
- **Python Testing with pytest**: https://docs.pytest.org/
- **Clean Code**: Robert C. Martin

---

## 🆘 FAQ

### Q: How do I change the output format to PKL?
**A**: Modify `save_results()` function (see "Modifying Prompting Strategies" section)

### Q: How do I add a new discretization strategy?
**A**: Modify `_reduce_and_discretize()` to support custom strategy parameter

### Q: How do I use a different model architecture?
**A**: Pass different `model_class` to `run_pipeline()`, ensure it has `.encoder` attribute

### Q: How do I resume a failed pipeline run?
**A**: Currently not supported - implement checkpointing (see recommendations)

### Q: How do I process multiple datasets?
**A**: Run pipeline multiple times with different `--dataset_folder` arguments, or wrap in shell script

---

## ✅ Checklist for New Contributors

- [ ] Read this document fully
- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Download sample data and model weights
- [ ] Run pipeline once to verify setup
- [ ] Read through `embedding_pipeline.py` code
- [ ] Try modifying a parameter (e.g., `n_bins`)
- [ ] Run tests (when available): `pytest tests/`
- [ ] Check code style: `black src/ && flake8 src/`

---

**Document Version**: 1.0
**Last Updated**: 2026-02-06
**Maintainer**: [Your Name]
**Contact**: [Your Email]
