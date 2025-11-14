# Memory Optimization Updates - Set Transformer

## 🔧 Changes Made for 6GB GPU Support

### 1. Auto-Adaptive Architecture

The Set Transformer now automatically adjusts its size based on GPU memory:

**For 6GB GPU (< 8GB):**
- `d_model`: 128 (was 256) - **50% reduction**
- `num_heads`: 4 (was 8) - **50% reduction**
- `num_inducing_points`: 16 (was 32) - **50% reduction**
- `num_layers`: 3 (was 4) - **25% reduction**
- `decoder_layers`: 1 (was 2) - **50% reduction**
- `batch_size`: 12 (was 40) - **70% reduction**

**Result:** Parameters reduced from ~6.7M to ~1.5M (~78% reduction!)

**For 8-12GB GPU:**
- Medium configuration (192d, 6 heads, 24 inducing points)

**For 12GB+ GPU:**
- Full configuration (256d, 8 heads, 32 inducing points)

### 2. Batch Size Auto-Adjustment

Training script now automatically sets optimal batch sizes:
- **6GB GPU + Set Transformer**: batch_size = 12
- **6GB GPU + Deep Sets**: batch_size = 40 (no change, works great!)
- **8GB+ GPU**: Uses larger batch sizes

### 3. Memory Management

Added automatic cache clearing:
- Before training starts
- Every 20 batches during training
- Between epochs

### 4. User Warnings

The training script now warns you if:
- You're using Set Transformer on a 6GB GPU
- Recommends Deep Sets as a faster alternative
- Gives you 5 seconds to cancel and switch

## 📊 New Performance Characteristics (6GB GPU)

### Set Transformer (Reduced):
- **Parameters**: ~1.5M (was ~6.7M)
- **Memory**: ~2.5 GB (was ~5.5 GB)
- **Batch Size**: 12
- **Speed**: Slower than Deep Sets, but works!

### Deep Sets (Unchanged):
- **Parameters**: ~400K
- **Memory**: ~1.5 GB
- **Batch Size**: 40
- **Speed**: Fast ⚡

### PointNet2 (Unchanged):
- **Parameters**: ~1.5M
- **Memory**: ~3.0 GB
- **Batch Size**: 40
- **Speed**: Fast ⚡

## 🚀 Recommended Workflow for 6GB GPU

### Option 1: Use Deep Sets (RECOMMENDED ⭐)
```bash
python train_set_transformer.py --model_type deepsets
```
- Fastest training
- Full batch size (40)
- Good baseline performance
- ~400K parameters

### Option 2: Use PointNet2 (PROVEN ⭐)
```bash
python train.py
```
- Proven architecture
- Full batch size (40)
- Geometric inductive bias
- ~1.5M parameters

### Option 3: Use Reduced Set Transformer (EXPERIMENTAL)
```bash
python train_set_transformer.py
```
- Automatically uses reduced architecture
- Smaller batch size (12)
- Slower training (~3x longer than Deep Sets)
- Still expressive with ~1.5M parameters

## 📈 Training Time Estimates (300 epochs, 113 samples)

| Model | Batch Size | Time per Epoch | Total Time |
|-------|-----------|----------------|------------|
| Deep Sets | 40 | ~15s | ~75 min |
| PointNet2 | 40 | ~20s | ~100 min |
| Set Transformer (6GB) | 12 | ~60s | ~300 min |

**Conclusion: For 6GB GPU, Deep Sets is 4x faster than reduced Set Transformer!**

## 💡 When to Use Each Model

### Use Deep Sets If:
- ✅ You have limited GPU memory (6GB)
- ✅ You want fast experimentation
- ✅ You need a solid baseline quickly
- ✅ Training time is a concern

### Use PointNet2 If:
- ✅ You want proven performance
- ✅ You value geometric inductive bias
- ✅ You have 6-8GB GPU
- ✅ You need balance of speed and accuracy

### Use Set Transformer If:
- ✅ You have 8GB+ GPU
- ✅ You want maximum expressiveness
- ✅ You're willing to wait longer for training
- ✅ You're doing research/experimentation
- ⚠️ With 6GB: Only if you've tried others first

## 🔍 Verifying Memory Usage

Before training, check your GPU:
```bash
python test_gpu_memory.py
```

During training, monitor with:
```bash
nvidia-smi -l 1
```

Or in Python:
```python
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
```

## ⚙️ Manual Override (Advanced)

If you want to manually tune the Set Transformer, edit `models/SetTransformer.py`:

```python
# Line ~235-260, force specific configuration:
self.d_model = 96          # Even smaller
self.num_heads = 4
self.num_inducing_points = 12
self.num_layers = 2
```

And in `train_set_transformer.py`:
```python
# Line ~275, force smaller batch:
batch_size = 8  # or even 4
```

## 🎯 Summary

**The Set Transformer now works on 6GB GPU, but it's slower.**

**For best results on 6GB GPU: Use Deep Sets or PointNet2!**

They're both faster, use memory efficiently, and give you great baselines to compare against.

Save the Set Transformer for when you have more GPU memory or when you're doing final model comparisons.

---

**All changes are automatic - just run the training script as normal! 🎉**
