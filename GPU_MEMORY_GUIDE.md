# GPU Memory Optimization Guide

## ⚠️ Common Issue: CUDA Out of Memory

If you encounter `torch.OutOfMemoryError: CUDA out of memory`, here are solutions:

## 🔧 Quick Fixes

### 1. Reduce Batch Size
```python
# In training scripts, change:
batch_size = 40  # → 20 or 10
```

### 2. Reduce Number of Points
```python
# In dataset config (datasets/dataset_config.yaml):
num_points: 4096  # → 2048 or 1024
```

### 3. Use Smaller Model
```bash
# Instead of Set Transformer, use Deep Sets:
python train_set_transformer.py --model_type deepsets
```

## 📊 Memory Requirements by Configuration

### For 6GB GPU (like yours):
```bash
# Safe configuration
python train_set_transformer.py  # batch_size will auto-adjust
python compare_models.py --num_points 2048 --batch_size 4
```

### For 8GB GPU:
```bash
python train_set_transformer.py  # Default settings OK
python compare_models.py --num_points 4096 --batch_size 8
```

### For 12GB+ GPU:
```bash
# Can use larger configurations
python train_set_transformer.py  # All settings work
python compare_models.py --num_points 8192 --batch_size 16
```

## 🎛️ Model-Specific Memory Usage

| Model | Memory (2048 pts, bs=4) | Memory (4096 pts, bs=8) |
|-------|-------------------------|-------------------------|
| Deep Sets | ~800 MB | ~1.5 GB |
| PointNet2 | ~1.5 GB | ~3.0 GB |
| Set Transformer | ~2.5 GB | ~5.5 GB |

## 💡 Optimization Strategies

### 1. Modify Set Transformer Architecture

In `models/SetTransformer.py`, reduce model size:

```python
# Change these in __init__:
self.d_model = 128           # Down from 256
self.num_heads = 4           # Down from 8
self.num_inducing_points = 16 # Down from 32
self.num_layers = 3           # Down from 4
```

This will reduce parameters from 6.7M to ~1.5M and memory usage by ~60%.

### 2. Use Gradient Checkpointing

Add this to your training script:

```python
from torch.utils.checkpoint import checkpoint

# In model forward pass, wrap expensive operations:
def forward(self, x):
    for layer in self.encoder_layers:
        x = checkpoint(layer, x, use_reentrant=False)
    return x
```

This trades compute for memory (slower but uses less memory).

### 3. Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

This can reduce memory by ~40%.

### 4. Clear Cache Regularly

```python
import torch

# After each epoch or batch:
torch.cuda.empty_cache()
```

### 5. Use CPU for Some Operations

```python
# Move large operations to CPU if needed:
with torch.no_grad():
    corner_labels = create_corner_labels_improved(
        point_clouds.cpu(), 
        wf_vertices.cpu()
    ).cuda()
```

## 🔍 Monitoring Memory Usage

### Check Available Memory
```python
import torch

if torch.cuda.is_available():
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Total GPU Memory: {gpu_mem:.2f} GB")
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"Allocated: {allocated:.2f} GB")
    print(f"Reserved: {reserved:.2f} GB")
```

### During Training
```python
# Add to training loop:
if batch_idx % 10 == 0:
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    mem_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU Memory: {mem_allocated:.2f}GB allocated, {mem_reserved:.2f}GB reserved")
```

## 🚀 Recommended Configurations

### For Your 6GB GPU:

#### Training PointNet2 (Recommended):
```yaml
# datasets/dataset_config.yaml
Building3D:
  num_points: 4096
  
# train.py
batch_size: 40  # Should work fine
```

#### Training Set Transformer (Reduced):
```bash
# Option 1: Modify SetTransformer.py as shown above
# Option 2: Use smaller inputs
python train_set_transformer.py  # Will auto-adjust

# In train_set_transformer.py, change:
batch_size = 20  # Down from 40
```

#### Training Deep Sets (Best for 6GB):
```bash
python train_set_transformer.py --model_type deepsets
# Can use full batch_size=40, num_points=4096
```

## 📝 Configuration File Template

Create `config_6gb.yaml`:
```yaml
Building3D:
  data_path: "./datasets/demo_dataset"
  num_points: 2048  # Reduced for 6GB GPU
  use_color: true
  use_intensity: true
  use_group_ids: true
  use_border_weights: true
  normalize: true
  
training:
  batch_size: 20  # Reduced for 6GB GPU
  num_epochs: 300
  learning_rate: 0.001
  
model:
  type: "deepsets"  # or "pointnet2" or "transformer"
```

## ⚡ Quick Memory Test

Run this to test your GPU memory:

```python
import torch

device = torch.device('cuda')

# Test different configurations
configs = [
    (1024, 4, "Small"),
    (2048, 4, "Medium"),
    (4096, 4, "Large"),
    (4096, 8, "Very Large"),
]

for num_points, batch_size, name in configs:
    try:
        torch.cuda.empty_cache()
        x = torch.randn(batch_size, num_points, 10).to(device)
        print(f"✓ {name}: {num_points} points, batch {batch_size} - OK")
        del x
    except RuntimeError:
        print(f"✗ {name}: {num_points} points, batch {batch_size} - OOM")
        torch.cuda.empty_cache()
```

## 🎯 Summary for Your GPU (6GB)

**Best Choice: Deep Sets**
- Fast training
- Low memory usage
- Good baseline performance

**Alternative: PointNet2**
- batch_size=40, num_points=4096 ✓
- Proven architecture
- Moderate memory

**Use Set Transformer only if:**
- You reduce d_model to 128
- Use batch_size=20 or less
- Or use num_points=2048

**For Comparison Script:**
```bash
python compare_models.py --num_points 2048 --batch_size 4
```

## 📞 Still Having Issues?

1. Check actual memory usage: `nvidia-smi`
2. Close other applications using GPU
3. Restart Python kernel
4. Set environment variable:
   ```bash
   $env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
   ```
5. Update PyTorch to latest version

---

**Remember**: Lower batch_size = slower training, but same final results!
