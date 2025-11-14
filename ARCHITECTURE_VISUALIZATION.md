# Set Transformer Architecture Visualization

## High-Level Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DEEP SETS PRINCIPLE                                  │
│                                                                              │
│                      f(X) = ρ( Σ φ(xᵢ) )                                    │
│                                                                              │
│  φ: element-wise transform  |  Σ: aggregation  |  ρ: final transform       │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 1. Deep Sets Baseline (Simple)

```
Input Point Cloud [B, N, 10]
         │
         ▼
    ┌─────────┐
    │   φ     │  MLP: 10 → 256 → 256 → 256
    │  (MLP)  │  Applied to each point independently
    └─────────┘
         │
         ▼  [B, N, 256]
    ┌─────────┐
    │    Σ    │  Sum over N dimension
    │  (Sum)  │  Permutation invariant
    └─────────┘
         │
         ▼  [B, 256] (global)
    ┌─────────┐
    │  Expand │  Broadcast to all points
    └─────────┘
         │
         ▼  [B, N, 256]
    ┌─────────┐
    │  Concat │  Element + Global features
    └─────────┘
         │
         ▼  [B, N, 512]
    ┌─────────┐
    │   ρ     │  MLP: 512 → 256 → 128 → 1
    │  (MLP)  │  Per-point predictions
    └─────────┘
         │
         ▼
Corner Logits [B, N]

Parameters: ~400K
Speed: Fastest
Complexity: O(N)
```

## 2. Set Transformer (Advanced)

```
Input Point Cloud [B, N, 10]
         │
         ▼
    ┌──────────────┐
    │   Embedding  │  Linear: 10 → 256
    │     (φ)      │  Initial transformation
    └──────────────┘
         │
         ▼  [B, N, 256]
    ╔══════════════╗
    ║   ISAB 1     ║  Induced Set Attention Block
    ║              ║  • 32 inducing points (learnable)
    ║   I ← X      ║  • Multi-head attention (8 heads)
    ║   X ← I      ║  • Complexity: O(MN) where M=32
    ╚══════════════╝
         │
         ▼  [B, N, 256]
    ╔══════════════╗
    ║   ISAB 2     ║  Same structure
    ╚══════════════╝
         │
         ▼  [B, N, 256]
    ╔══════════════╗
    ║   ISAB 3     ║  Building hierarchical
    ╚══════════════╝  representations
         │
         ▼  [B, N, 256]
    ╔══════════════╗
    ║   ISAB 4     ║  Final encoding
    ╚══════════════╝
         │
         ▼  [B, N, 256]
    ┌──────────────┐
    │    SAB 1     │  Set Attention Block (decoder)
    │              │  • Full self-attention
    │   X ↔ X      │  • Refines features
    └──────────────┘
         │
         ▼  [B, N, 256]
    ┌──────────────┐
    │    SAB 2     │  Final refinement
    └──────────────┘
         │
         ▼  [B, N, 256]
    ┌──────────────┐
    │ Corner Head  │  MLP: 256 → 128 → 64 → 1
    │     (ρ)      │  Per-point corner scores
    └──────────────┘
         │
         ▼
Corner Logits [B, N]

Parameters: ~6.7M
Speed: Medium
Complexity: O(MN), M=32
```

## 3. PointNet2 (Geometric)

```
Input Point Cloud [B, N, 10]
         │
         ▼
    ┌──────────────┐
    │     FPS      │  Farthest Point Sampling
    │              │  Select representative points
    └──────────────┘
         │
         ▼  [B, N', d]
    ┌──────────────┐
    │ Ball Query   │  Group nearby points
    │   + MLP      │  Local feature extraction
    └──────────────┘
         │
         ▼  [B, N', d']
    ┌──────────────┐
    │  MaxPool     │  Aggregate local features
    └──────────────┘
         │
         ▼  [B, N'', d'']
    ┌──────────────┐
    │   Repeat     │  Multiple Set Abstraction layers
    └──────────────┘
         │
         ▼  [B, d_global]
    ┌──────────────┐
    │  Upsample    │  Feature Propagation
    │   + MLP      │  Back to original resolution
    └──────────────┘
         │
         ▼  [B, N, d_final]
    ┌──────────────┐
    │   MLP Head   │  Corner prediction
    └──────────────┘
         │
         ▼
Corner Logits [B, N]

Parameters: ~1.5M
Speed: Fast
Complexity: O(N log N)
```

## Attention Mechanism Detail (Set Transformer)

```
┌─────────────────────────────────────────────────────────────┐
│            INDUCED SET ATTENTION BLOCK (ISAB)               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input X [B, N, d]                                          │
│     │                                                        │
│     │  ┌────────────────────┐                              │
│     │  │ Inducing Points I  │ [B, M, d]  (M=32)           │
│     │  │   (Learnable)      │                              │
│     │  └────────────────────┘                              │
│     │           │                                           │
│     ▼           ▼                                           │
│  ┌─────────────────────┐                                   │
│  │  Multi-Head Attn    │  I attends to X                   │
│  │   I ← Attn(I, X)    │  [B, M, d]                       │
│  └─────────────────────┘                                   │
│           │                                                 │
│           ▼  H [B, M, d]                                   │
│  ┌─────────────────────┐                                   │
│  │  Multi-Head Attn    │  X attends to H                   │
│  │   X' ← Attn(X, H)   │  [B, N, d]                       │
│  └─────────────────────┘                                   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────┐                                   │
│  │   Layer Norm        │  Normalization                    │
│  │   + Residual        │  X' = X + Dropout(Attn(X, H))    │
│  └─────────────────────┘                                   │
│           │                                                 │
│           ▼                                                 │
│  ┌─────────────────────┐                                   │
│  │   Feed Forward      │  MLP: d → 4d → d                 │
│  │   + Residual        │                                   │
│  └─────────────────────┘                                   │
│           │                                                 │
│           ▼                                                 │
│  Output X'' [B, N, d]                                      │
│                                                              │
│  Complexity: O(MN + M²) ≈ O(MN) when M << N               │
└─────────────────────────────────────────────────────────────┘
```

## Multi-Head Attention Detail

```
┌──────────────────────────────────────────────────────┐
│         MULTI-HEAD ATTENTION (8 heads)               │
├──────────────────────────────────────────────────────┤
│                                                       │
│  Query Q [B, N_q, d]                                 │
│  Key   K [B, N_k, d]                                 │
│  Value V [B, N_v, d]                                 │
│     │      │      │                                   │
│     ▼      ▼      ▼                                   │
│  ┌─────────────────┐                                │
│  │  Linear Proj    │  W_q, W_k, W_v                 │
│  └─────────────────┘                                │
│     │      │      │                                   │
│     ▼      ▼      ▼                                   │
│  ┌─────────────────┐                                │
│  │ Split into      │  [B, h, N, d/h]                │
│  │ h=8 heads       │  h heads, d_k = d/h = 32       │
│  └─────────────────┘                                │
│     │      │      │                                   │
│     ▼      ▼      ▼                                   │
│        Q_h  K_h  V_h   (for each head h)            │
│          │   │   │                                    │
│          ▼   ▼   ▼                                    │
│  ┌────────────────────┐                             │
│  │  Attention Score   │  QK^T / √d_k                │
│  │  Score = QK^T/√d_k │  [B, h, N_q, N_k]           │
│  └────────────────────┘                             │
│          │                                            │
│          ▼                                            │
│  ┌────────────────────┐                             │
│  │   Softmax          │  Normalize scores            │
│  └────────────────────┘                             │
│          │                                            │
│          ▼  [B, h, N_q, N_k]                        │
│  ┌────────────────────┐                             │
│  │  Weighted Sum      │  Attention × V              │
│  │  Output = Attn(V)  │  [B, h, N_q, d_k]           │
│  └────────────────────┘                             │
│          │                                            │
│          ▼                                            │
│  ┌────────────────────┐                             │
│  │  Concatenate       │  [B, N_q, d]                │
│  │  heads             │                              │
│  └────────────────────┘                             │
│          │                                            │
│          ▼                                            │
│  ┌────────────────────┐                             │
│  │  Linear W_o        │  Final projection            │
│  └────────────────────┘                             │
│          │                                            │
│          ▼                                            │
│  Output [B, N_q, d]                                  │
│                                                       │
└──────────────────────────────────────────────────────┘
```

## Training Flow Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘

DATA LOADING (same for all)
    │
    ▼
Point Cloud [B, N, 10]  (xyz + rgba + intensity + group + border)
    │
    ├──────────────┬──────────────┬──────────────┐
    │              │              │              │
    ▼              ▼              ▼              ▼
PointNet2      Set          Deep Sets    Ensemble
              Transformer

FORWARD PASS
    │              │              │              │
    ▼              ▼              ▼              ▼
Corner         Corner        Corner       Combined
Logits [B,N]   Logits [B,N]  Logits [B,N] Logits [B,N]

LOSS COMPUTATION (same for all)
    │
    ▼
AdaptiveCornerLoss
    • Focal Loss (class imbalance)
    • Distance Loss (spatial accuracy)
    │
    ▼
Backward + Optimizer Step
    │
    ▼
Update Weights

CHECKPOINTS (every 10 epochs)
    • Model state dict
    • Optimizer state
    • Loss
    • Config
```

## Memory & Speed Comparison

```
┌────────────────────────────────────────────────────────────┐
│                  RESOURCE COMPARISON                        │
│                (batch_size=8, num_points=4096)             │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Model          │ Parameters │ Memory  │ Speed    │ FLOPs │
│  ──────────────┼────────────┼─────────┼──────────┼───────│
│  Deep Sets     │    400K    │  ~1 GB  │  8 ms    │  Low  │
│  PointNet2     │   1.5M     │  ~2 GB  │  12 ms   │  Med  │
│  Set Transform │   6.7M     │  ~4 GB  │  18 ms   │  High │
│                                                             │
│  Recommendation:                                            │
│  • Prototyping → Deep Sets (fastest)                       │
│  • Production  → PointNet2 (balanced)                      │
│  • Research    → Set Transformer (most expressive)         │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

## Key Advantages of Each Model

```
┌─────────────────────────────────────────────────────────────┐
│                    MODEL STRENGTHS                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  DEEP SETS                                                   │
│  ✓ Theoretically grounded                                   │
│  ✓ Fastest training & inference                             │
│  ✓ Lowest memory footprint                                  │
│  ✓ Good baseline for comparisons                            │
│  ✓ Mathematically proven permutation invariance             │
│                                                              │
│  SET TRANSFORMER                                             │
│  ✓ Most expressive (attention mechanism)                    │
│  ✓ Captures long-range dependencies                         │
│  ✓ Dynamic feature weighting                                │
│  ✓ State-of-the-art for set-structured data                │
│  ✓ Scalable with ISAB (O(MN) instead of O(N²))            │
│                                                              │
│  POINTNET2                                                   │
│  ✓ Geometric inductive bias                                 │
│  ✓ Proven for point clouds                                  │
│  ✓ Good balance of speed and accuracy                       │
│  ✓ Hierarchical feature learning                            │
│  ✓ Explicit spatial locality                                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```
