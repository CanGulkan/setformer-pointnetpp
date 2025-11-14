# Set Transformer for Corner Detection

This implementation provides a **Set Transformer** architecture based on the **Deep Sets** paper for permutation-invariant corner detection in 3D point clouds.

## Overview

### Deep Sets Theory

The core principle from the Deep Sets paper is that any permutation-invariant function on sets can be decomposed as:

$$f(X) = \rho\left(\sum_{x \in X} \phi(x)\right)$$

Where:
- $\phi$ : element-wise transformation (applied to each point independently)
- $\sum$ : permutation-invariant aggregation
- $\rho$ : final transformation on aggregated features

### Set Transformer Architecture

The Set Transformer extends this with **attention mechanisms** for more expressive set operations:

1. **Multi-Head Attention (MHA)**: Computes relationships between set elements
2. **Set Attention Block (SAB)**: Self-attention on all set elements (permutation equivariant)
3. **Induced Set Attention Block (ISAB)**: Efficient attention using inducing points (reduces O(N²) to O(MN))
4. **Pooling by Multihead Attention (PMA)**: Learned pooling using seed vectors

## Architecture Components

### 1. Input Embedding (φ in Deep Sets)
```
Input [B, N, C] → Linear(128) → ReLU → Linear(256) → [B, N, d_model]
```

### 2. Encoder (4 × ISAB)
Each ISAB layer:
- Uses 32 inducing points for efficiency
- Multi-head attention (8 heads)
- Layer normalization + residual connections
- Feed-forward network (4× expansion)

### 3. Decoder (2 × SAB)
- Refines encoded features
- Maintains permutation equivariance for point-wise outputs

### 4. Corner Detection Head (ρ in Deep Sets)
```
Features [B, N, 256] → MLP → Corner Logits [B, N]
```

## Models Provided

### 1. **SetTransformer** (Full Model)
- Complexity: ~2M parameters
- Uses Induced Set Attention Blocks for efficiency
- Multi-head attention with 8 heads
- Best for large point clouds

### 2. **DeepSetsMLP** (Baseline)
- Complexity: ~500K parameters
- Simple φ-Σ-ρ architecture
- Faster training
- Good baseline for comparison

## Usage

### Training with Set Transformer
```bash
# Train from scratch
python train_set_transformer.py

# Train with Deep Sets baseline
python train_set_transformer.py --model_type deepsets

# Resume training
python train_set_transformer.py --resume output/set_transformer_checkpoint_epoch_100.pth --wandb_id YOUR_RUN_ID
```

### Using in Your Code
```python
from models import SetTransformer, DeepSetsMLP, build_set_transformer

# Option 1: Use factory function
model = build_set_transformer(config, model_type='transformer')

# Option 2: Direct instantiation
model = SetTransformer(config)

# Forward pass
corner_logits, global_features = model(point_clouds)
```

### Configuration
```python
config = {
    'input_channels': 10,  # xyz + rgba + intensity + group_id + border_weight
    'use_color': True,
    'use_intensity': True,
    'use_group_ids': True,
    'use_border_weights': True,
    'normalize': True,
    'num_points': 4096
}
```

## Key Features

### 1. Permutation Invariance
The architecture is theoretically guaranteed to be permutation invariant:
- Input order doesn't matter
- Attention mechanism computes relationships dynamically
- No dependence on point cloud structure

### 2. Efficiency
- **ISAB** reduces complexity from O(N²) to O(MN) where M=32
- Suitable for point clouds with thousands of points
- GPU-efficient implementation with batched operations

### 3. Expressiveness
- Multi-head attention captures complex relationships
- Multiple layers build hierarchical representations
- Global and local features combined

### 4. Compatibility
- Drop-in replacement for PointNet2
- Same input/output format
- Works with existing training pipeline

## Comparison: Set Transformer vs PointNet2

| Feature | Set Transformer | PointNet2 |
|---------|----------------|-----------|
| **Permutation Invariance** | Guaranteed by design | Via max pooling |
| **Local Context** | Attention-based | FPS + ball query |
| **Complexity** | O(MN) with ISAB | O(N log N) |
| **Parameters** | ~2M | ~1.5M |
| **Training Speed** | Medium | Fast |
| **Expressiveness** | High (attention) | Medium (geometric) |

## Hyperparameters

```python
# Set Transformer
d_model = 256              # Hidden dimension
num_heads = 8              # Attention heads
num_inducing_points = 32   # ISAB efficiency
num_layers = 4             # Encoder depth
dropout = 0.1              # Regularization
```

## Theoretical Properties

### Permutation Equivariance
For point-wise outputs (corner detection):
$$f(\pi(X)) = \pi(f(X))$$
Where π is any permutation of the input points.

### Permutation Invariance
For global outputs:
$$f(\pi(X)) = f(X)$$

### Universal Approximation
Set Transformers can approximate any continuous permutation-equivariant function given sufficient capacity (from Deep Sets theorem).

## Performance Tips

1. **Batch Size**: Use 32-40 for best GPU utilization
2. **Learning Rate**: Start with 0.001, decay if needed
3. **Gradient Clipping**: Use max_norm=1.0 for stability
4. **Warmup**: Consider learning rate warmup for first few epochs
5. **Data Augmentation**: Rotation and jittering are permutation-preserving

## Evaluation

Compare both models:
```bash
# Train PointNet2
python train.py

# Train Set Transformer
python train_set_transformer.py

# Train Deep Sets baseline
python train_set_transformer.py --model_type deepsets
```

Monitor:
- Training loss convergence
- Corner detection precision/recall
- Inference time per sample
- Model size

## References

1. **Deep Sets** (Zaheer et al., 2017)
   - Introduces φ-Σ-ρ decomposition
   - Proves universal approximation theorem for sets

2. **Set Transformer** (Lee et al., 2019)
   - Extends Deep Sets with attention
   - Introduces ISAB and PMA for efficiency

3. **Attention Is All You Need** (Vaswani et al., 2017)
   - Multi-head attention mechanism
   - Layer normalization and residual connections

## File Structure

```
models/
  ├── SetTransformer.py          # Set Transformer implementation
  ├── PointNet2.py               # Original PointNet2 model
  └── __init__.py                # Model exports

train_set_transformer.py         # Training script for Set Transformer
train.py                         # Training script for PointNet2
```

## Troubleshooting

**OOM (Out of Memory)**
- Reduce batch size
- Reduce num_inducing_points (32 → 16)
- Reduce d_model (256 → 128)

**Slow Training**
- Use ISAB instead of SAB (already default)
- Reduce num_layers (4 → 3)
- Increase num_workers in DataLoader

**Poor Convergence**
- Add learning rate scheduler
- Increase dropout for regularization
- Check data normalization
- Verify corner labels are balanced

## Future Improvements

1. **Pre-training**: Self-supervised learning on unlabeled point clouds
2. **Multi-scale**: Different inducing points at different layers
3. **Graph Attention**: Incorporate explicit edges between nearby points
4. **Memory Compression**: Compress long-range dependencies
5. **Architecture Search**: NAS for optimal hyperparameters
