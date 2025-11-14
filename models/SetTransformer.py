"""
Set Transformer for Corner Detection
Based on "Deep Sets" paper - permutation invariant deep learning on sets
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for set elements"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [B, N_q, d_model]
            key: [B, N_k, d_model]
            value: [B, N_v, d_model]
            mask: [B, N_q, N_k] or [B, 1, N_k]
        Returns:
            output: [B, N_q, d_model]
            attention: [B, num_heads, N_q, N_k]
        """
        batch_size = query.size(0)
        
        # Linear projections and split into multiple heads
        # [B, N, d_model] -> [B, N, num_heads, d_k] -> [B, num_heads, N, d_k]
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # [B, num_heads, N_q, d_k] x [B, num_heads, d_k, N_k] = [B, num_heads, N_q, N_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            # Expand mask to match attention dimensions
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, N_q, N_k]
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        # [B, num_heads, N_q, N_k] x [B, num_heads, N_k, d_k] = [B, num_heads, N_q, d_k]
        output = torch.matmul(attention, V)
        
        # Concatenate heads and apply final linear layer
        # [B, num_heads, N_q, d_k] -> [B, N_q, num_heads, d_k] -> [B, N_q, d_model]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(output)
        
        return output, attention


class SetAttentionBlock(nn.Module):
    """
    Set Attention Block (SAB) - applies self-attention to set elements
    Permutation invariant operation
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, X, mask=None):
        """
        Args:
            X: [B, N, d_model] - set of N elements
            mask: [B, N] - optional mask for padded elements
        Returns:
            output: [B, N, d_model]
        """
        # Self-attention with residual connection
        attn_output, _ = self.mha(X, X, X, mask)
        X = self.norm1(X + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(X)
        X = self.norm2(X + ffn_output)
        
        return X


class InducedSetAttentionBlock(nn.Module):
    """
    Induced Set Attention Block (ISAB) - uses inducing points for efficiency
    Reduces complexity from O(N^2) to O(MN) where M << N
    """
    
    def __init__(self, d_model, num_heads, num_inducing_points, dropout=0.1):
        super().__init__()
        self.num_inducing_points = num_inducing_points
        
        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, d_model))
        
        self.mha1 = MultiHeadAttention(d_model, num_heads, dropout)
        self.mha2 = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, X, mask=None):
        """
        Args:
            X: [B, N, d_model]
            mask: [B, N]
        Returns:
            output: [B, N, d_model]
        """
        batch_size = X.size(0)
        
        # Expand inducing points for batch
        I = self.inducing_points.expand(batch_size, -1, -1)  # [B, M, d_model]
        
        # Attention from inducing points to input
        H, _ = self.mha1(I, X, X, mask)
        H = self.norm1(I + H)
        
        # Attention from input to inducing points
        attn_output, _ = self.mha2(X, H, H)
        X = self.norm2(X + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(X)
        X = self.norm3(X + ffn_output)
        
        return X


class PoolingByMultiHeadAttention(nn.Module):
    """
    Pooling by Multihead Attention (PMA) - learns to pool set elements
    Uses learnable seed vectors to aggregate information
    """
    
    def __init__(self, d_model, num_heads, num_seeds, dropout=0.1):
        super().__init__()
        self.num_seeds = num_seeds
        
        # Learnable seed vectors for pooling
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seeds, d_model))
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, X, mask=None):
        """
        Args:
            X: [B, N, d_model]
            mask: [B, N]
        Returns:
            output: [B, num_seeds, d_model]
        """
        batch_size = X.size(0)
        
        # Expand seed vectors for batch
        S = self.seed_vectors.expand(batch_size, -1, -1)  # [B, K, d_model]
        
        # Attention from seeds to input
        attn_output, _ = self.mha(S, X, X, mask)
        S = self.norm(S + attn_output)
        
        # Feed-forward
        ffn_output = self.ffn(S)
        S = self.norm2(S + ffn_output)
        
        return S


class SetTransformer(nn.Module):
    """
    Set Transformer for Corner Detection
    Permutation invariant architecture based on Deep Sets principles
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Configuration
        self.use_color = config.get('use_color', False)
        self.use_intensity = config.get('use_intensity', False)
        self.use_group_ids = config.get('use_group_ids', False)
        self.use_border_weights = config.get('use_border_weights', False)
        
        # Calculate input channels (xyz + features)
        input_channels = config.get('input_channels', 3)
        
        # Model hyperparameters - auto-adjust for GPU memory
        import torch
        
        # Check GPU memory and adjust model size
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            
            if gpu_mem < 8:
                # For 6GB GPUs: Use smaller model
                self.d_model = 128  # Reduced from 256
                self.num_heads = 4  # Reduced from 8
                self.num_inducing_points = 16  # Reduced from 32
                self.num_layers = 3  # Reduced from 4
                print(f"⚠ GPU memory < 8GB detected ({gpu_mem:.1f}GB)")
                print(f"⚠ Using reduced Set Transformer architecture for memory efficiency")
            elif gpu_mem < 12:
                # For 8-12GB GPUs: Use medium model
                self.d_model = 192
                self.num_heads = 6
                self.num_inducing_points = 24
                self.num_layers = 3
            else:
                # For 12GB+ GPUs: Use full model
                self.d_model = 256
                self.num_heads = 8
                self.num_inducing_points = 32
                self.num_layers = 4
        else:
            # CPU fallback
            self.d_model = 128
            self.num_heads = 4
            self.num_inducing_points = 16
            self.num_layers = 3
        
        self.dropout = 0.1
        
        # Input embedding: φ(x) in Deep Sets notation
        # Projects raw features to d_model dimensions
        self.input_embedding = nn.Sequential(
            nn.Linear(input_channels, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, self.d_model),
            nn.LayerNorm(self.d_model)
        )
        
        # Stack of Set Attention Blocks (can use SAB or ISAB)
        self.encoder_layers = nn.ModuleList([
            InducedSetAttentionBlock(
                self.d_model, 
                self.num_heads, 
                self.num_inducing_points,
                self.dropout
            ) for _ in range(self.num_layers)
        ])
        
        # Decoder: processes encoded features for per-point predictions
        # This maintains permutation equivariance for point-wise outputs
        # Reduced to 1 layer for memory efficiency
        self.decoder_layers = nn.ModuleList([
            SetAttentionBlock(self.d_model, self.num_heads, self.dropout)
            for _ in range(1)  # Reduced from 2 to 1
        ])
        
        # Corner detection head: ρ(Σ φ(x)) in Deep Sets notation
        # Outputs per-point corner probability
        self.corner_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 1)  # Binary corner classification
        )
        
        # Optional: Global pooling for set-level features
        self.global_pool = PoolingByMultiHeadAttention(
            self.d_model,
            self.num_heads,
            num_seeds=1,
            dropout=self.dropout
        )
        
        # Border attention weight (learnable parameter)
        if self.use_border_weights:
            self.border_attention_weight = nn.Parameter(torch.tensor(0.0))
        else:
            self.border_attention_weight = None
        
        print(f"Set Transformer initialized:")
        print(f"  - Input channels: {input_channels}")
        print(f"  - d_model: {self.d_model}")
        print(f"  - num_heads: {self.num_heads}")
        print(f"  - num_layers: {self.num_layers}")
        print(f"  - num_inducing_points: {self.num_inducing_points}")
        
    def forward(self, point_clouds):
        """
        Args:
            point_clouds: [B, N, C] where C = input_channels
        Returns:
            corner_logits: [B, N] - corner detection scores for each point
            global_features: [B, d_model] - optional global set features
        """
        B, N, C = point_clouds.shape
        
        # Input embedding: φ(x)
        X = self.input_embedding(point_clouds)  # [B, N, d_model]
        
        # Create mask for valid points (assuming -1e9 is padding sentinel)
        # You can adjust this based on your padding strategy
        mask = None
        if torch.any(point_clouds[:, :, 0] < -1e8):
            mask = (point_clouds[:, :, 0] > -1e8).unsqueeze(-1)  # [B, N, 1]
        
        # Encoder: process set elements with attention
        for layer in self.encoder_layers:
            X = layer(X, mask)  # [B, N, d_model]
        
        # Decoder: refine features for per-point predictions
        for layer in self.decoder_layers:
            X = layer(X, mask)  # [B, N, d_model]
        
        # Corner detection: ρ(features)
        corner_logits = self.corner_head(X).squeeze(-1)  # [B, N]
        
        # Optional: Extract global set features
        global_features = self.global_pool(X, mask).squeeze(1)  # [B, d_model]
        
        return corner_logits, global_features


class DeepSetsMLP(nn.Module):
    """
    Simple Deep Sets baseline: f(X) = ρ(Σ φ(x))
    Simpler alternative to Set Transformer
    """
    
    def __init__(self, config):
        super().__init__()
        
        input_channels = config.get('input_channels', 3)
        hidden_dim = 256
        
        # φ: element-wise transformation
        self.phi = nn.Sequential(
            nn.Linear(input_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # ρ: aggregation and final transformation
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # For per-point predictions, we need element-wise features
        self.point_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # concat element + global
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, point_clouds):
        """
        Args:
            point_clouds: [B, N, C]
        Returns:
            corner_logits: [B, N]
            global_features: [B, hidden_dim]
        """
        # Element-wise transformation: φ(x)
        element_features = self.phi(point_clouds)  # [B, N, hidden_dim]
        
        # Permutation invariant aggregation: Σ φ(x)
        global_features = torch.sum(element_features, dim=1)  # [B, hidden_dim]
        
        # For per-point prediction, combine element and global features
        # Expand global features to all points
        global_expanded = global_features.unsqueeze(1).expand(-1, point_clouds.size(1), -1)
        
        # Concatenate element-wise and global features
        combined = torch.cat([element_features, global_expanded], dim=-1)  # [B, N, hidden_dim*2]
        
        # Per-point corner prediction
        corner_logits = self.point_decoder(combined).squeeze(-1)  # [B, N]
        
        return corner_logits, global_features


# Factory function for easy model creation
def build_set_transformer(config, model_type='transformer'):
    """
    Build Set Transformer or Deep Sets model
    
    Args:
        config: Model configuration dict
        model_type: 'transformer' or 'deepsets'
    """
    if model_type == 'transformer':
        return SetTransformer(config)
    elif model_type == 'deepsets':
        return DeepSetsMLP(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
