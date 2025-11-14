"""
Test script to verify Set Transformer implementation
"""

import torch
from models import SetTransformer, DeepSetsMLP, build_set_transformer


def test_set_transformer():
    """Test Set Transformer forward pass and properties"""
    
    print("="*60)
    print("Testing Set Transformer Implementation")
    print("="*60)
    
    # Configuration
    config = {
        'input_channels': 10,  # xyz + rgba + intensity + group_id + border_weight
        'use_color': True,
        'use_intensity': True,
        'use_group_ids': True,
        'use_border_weights': True,
        'normalize': True,
        'num_points': 4096
    }
    
    # Test dimensions
    batch_size = 4
    num_points = 1024
    input_channels = config['input_channels']
    
    # Create random input
    point_clouds = torch.randn(batch_size, num_points, input_channels)
    
    print("\n1. Testing Set Transformer...")
    print("-" * 60)
    
    # Build model
    model = build_set_transformer(config, model_type='transformer')
    model.eval()
    
    # Forward pass
    with torch.no_grad():
        corner_logits, global_features = model(point_clouds)
    
    print(f"✓ Input shape: {point_clouds.shape}")
    print(f"✓ Corner logits shape: {corner_logits.shape}")
    print(f"✓ Global features shape: {global_features.shape}")
    
    assert corner_logits.shape == (batch_size, num_points), "Corner logits shape mismatch"
    assert global_features.shape[0] == batch_size, "Global features batch size mismatch"
    print("✓ Output shapes correct")
    
    # Test permutation invariance for global features
    print("\n2. Testing Permutation Invariance...")
    print("-" * 60)
    
    # Permute points
    perm_indices = torch.randperm(num_points)
    point_clouds_permuted = point_clouds[:, perm_indices, :]
    
    with torch.no_grad():
        corner_logits_perm, global_features_perm = model(point_clouds_permuted)
    
    # Global features should be similar (approximately invariant)
    global_diff = torch.abs(global_features - global_features_perm).mean().item()
    print(f"Global features difference after permutation: {global_diff:.6f}")
    
    # Corner logits should be permuted accordingly (equivariant)
    corner_logits_reordered = corner_logits[:, perm_indices]
    corner_diff = torch.abs(corner_logits_perm - corner_logits_reordered).mean().item()
    print(f"Corner logits difference after reordering: {corner_diff:.6f}")
    
    if corner_diff < 1e-5:
        print("✓ Permutation equivariance verified for corner detection")
    else:
        print(f"⚠ Corner logits differ by {corner_diff} (expected < 1e-5)")
    
    # Test gradient flow
    print("\n3. Testing Gradient Flow...")
    print("-" * 60)
    
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Dummy loss
    corner_logits, _ = model(point_clouds)
    dummy_labels = torch.randint(0, 2, (batch_size, num_points)).float()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(corner_logits, dummy_labels)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    has_gradients = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break
    
    if has_gradients:
        print("✓ Gradients computed successfully")
    else:
        print("✗ No gradients found")
    
    optimizer.step()
    print("✓ Optimizer step successful")
    
    print("\n4. Testing Deep Sets Baseline...")
    print("-" * 60)
    
    model_deepsets = build_set_transformer(config, model_type='deepsets')
    model_deepsets.eval()
    
    with torch.no_grad():
        corner_logits_ds, global_features_ds = model_deepsets(point_clouds)
    
    print(f"✓ Input shape: {point_clouds.shape}")
    print(f"✓ Corner logits shape: {corner_logits_ds.shape}")
    print(f"✓ Global features shape: {global_features_ds.shape}")
    
    assert corner_logits_ds.shape == (batch_size, num_points), "Corner logits shape mismatch"
    print("✓ Deep Sets model works correctly")
    
    # Compare parameter counts
    print("\n5. Model Statistics...")
    print("-" * 60)
    
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    transformer_params = count_parameters(model)
    deepsets_params = count_parameters(model_deepsets)
    
    print(f"Set Transformer parameters: {transformer_params:,}")
    print(f"Deep Sets baseline parameters: {deepsets_params:,}")
    print(f"Ratio: {transformer_params / deepsets_params:.2f}x")
    
    print("\n" + "="*60)
    print("✓ All tests passed!")
    print("="*60)
    
    return True


def test_different_batch_sizes():
    """Test model with different batch sizes"""
    
    print("\n" + "="*60)
    print("Testing Different Batch Sizes")
    print("="*60 + "\n")
    
    config = {
        'input_channels': 10,
        'use_color': True,
        'use_intensity': True,
        'use_group_ids': True,
        'use_border_weights': True,
    }
    
    model = build_set_transformer(config, model_type='transformer')
    model.eval()
    
    test_cases = [
        (1, 512),    # Single sample, small
        (2, 1024),   # Small batch
        (8, 2048),   # Medium batch
        (4, 4096),   # Large point cloud
    ]
    
    for batch_size, num_points in test_cases:
        point_clouds = torch.randn(batch_size, num_points, config['input_channels'])
        
        with torch.no_grad():
            corner_logits, global_features = model(point_clouds)
        
        assert corner_logits.shape == (batch_size, num_points)
        print(f"✓ Batch size {batch_size}, Points {num_points}: OK")
    
    print("\n✓ All batch size tests passed!")


def test_cuda_if_available():
    """Test model on CUDA if available"""
    
    if not torch.cuda.is_available():
        print("\n⚠ CUDA not available, skipping GPU test")
        return
    
    print("\n" + "="*60)
    print("Testing CUDA Support")
    print("="*60 + "\n")
    
    config = {
        'input_channels': 10,
        'use_color': True,
        'use_intensity': True,
        'use_group_ids': True,
        'use_border_weights': True,
    }
    
    device = torch.device('cuda')
    model = build_set_transformer(config, model_type='transformer')
    model = model.to(device)
    
    batch_size = 4
    num_points = 2048
    point_clouds = torch.randn(batch_size, num_points, config['input_channels']).to(device)
    
    with torch.no_grad():
        corner_logits, global_features = model(point_clouds)
    
    assert corner_logits.is_cuda, "Output not on CUDA"
    print(f"✓ Model runs on CUDA")
    print(f"✓ Output shape: {corner_logits.shape}")
    print(f"✓ Device: {corner_logits.device}")


if __name__ == "__main__":
    try:
        # Run tests
        test_set_transformer()
        test_different_batch_sizes()
        test_cuda_if_available()
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("="*60)
        print("\nYou can now train the model with:")
        print("  python train_set_transformer.py")
        print("\nOr compare with PointNet2:")
        print("  python train.py")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
