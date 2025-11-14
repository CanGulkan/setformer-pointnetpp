"""
Quick GPU Memory Test
Run this before training to find optimal batch size and num_points
"""

import torch
import sys


def test_gpu_memory():
    """Test GPU memory with different configurations"""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please check your PyTorch installation.")
        return
    
    device = torch.device('cuda')
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print("\n" + "="*70)
    print("GPU MEMORY TEST")
    print("="*70)
    print(f"GPU: {gpu_name}")
    print(f"Total Memory: {gpu_mem:.2f} GB")
    print()
    
    # Test configurations
    configs = [
        # (num_points, batch_size, model_type, description)
        (1024, 4, "deepsets", "Tiny - Deep Sets"),
        (2048, 4, "deepsets", "Small - Deep Sets"),
        (4096, 8, "deepsets", "Medium - Deep Sets"),
        (4096, 20, "deepsets", "Large - Deep Sets"),
        
        (1024, 4, "transformer", "Tiny - Set Transformer"),
        (2048, 4, "transformer", "Small - Set Transformer"),
        (4096, 4, "transformer", "Medium - Set Transformer"),
        (4096, 8, "transformer", "Large - Set Transformer"),
        (4096, 20, "transformer", "Very Large - Set Transformer"),
    ]
    
    from models import build_set_transformer
    
    results = []
    
    print(f"{'Config':<30s} {'Points':<8s} {'Batch':<8s} {'Status':<10s} {'Memory':<15s}")
    print("-" * 70)
    
    for num_points, batch_size, model_type, description in configs:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        try:
            # Create model
            config = {
                'input_channels': 10,
                'use_color': True,
                'use_intensity': True,
                'use_group_ids': True,
                'use_border_weights': True,
            }
            
            model = build_set_transformer(config, model_type=model_type)
            model = model.to(device)
            model.eval()
            
            # Test forward pass
            x = torch.randn(batch_size, num_points, 10).to(device)
            
            with torch.no_grad():
                _, _ = model(x)
            
            # Get memory usage
            mem_allocated = torch.cuda.max_memory_allocated() / 1024**2  # MB
            
            status = "✓ OK"
            mem_str = f"{mem_allocated:.0f} MB"
            results.append((description, num_points, batch_size, True, mem_allocated))
            
            print(f"{description:<30s} {num_points:<8d} {batch_size:<8d} {status:<10s} {mem_str:<15s}")
            
            # Clean up
            del model, x
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                status = "✗ OOM"
                mem_str = "N/A"
                results.append((description, num_points, batch_size, False, 0))
                print(f"{description:<30s} {num_points:<8d} {batch_size:<8d} {status:<10s} {mem_str:<15s}")
                torch.cuda.empty_cache()
            else:
                print(f"Error: {e}")
                break
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    # Find best configs
    best_deepsets = None
    best_transformer = None
    
    for desc, pts, bs, success, mem in results:
        if success:
            if "Deep Sets" in desc:
                if best_deepsets is None or (pts * bs) > (best_deepsets[1] * best_deepsets[2]):
                    best_deepsets = (desc, pts, bs, mem)
            elif "Transformer" in desc:
                if best_transformer is None or (pts * bs) > (best_transformer[1] * best_transformer[2]):
                    best_transformer = (desc, pts, bs, mem)
    
    print("\n📊 Optimal Configurations for Your GPU:\n")
    
    if best_deepsets:
        desc, pts, bs, mem = best_deepsets
        print(f"Deep Sets Baseline:")
        print(f"  python train_set_transformer.py --model_type deepsets")
        print(f"  • num_points: {pts}")
        print(f"  • batch_size: {bs}")
        print(f"  • Memory: {mem:.0f} MB")
        print()
    
    if best_transformer:
        desc, pts, bs, mem = best_transformer
        print(f"Set Transformer:")
        print(f"  python train_set_transformer.py")
        print(f"  • num_points: {pts}")
        print(f"  • batch_size: {bs}")
        print(f"  • Memory: {mem:.0f} MB")
        print()
    
    # Suggest config changes
    print("💡 To use these settings:")
    print()
    print("1. Edit datasets/dataset_config.yaml:")
    if best_deepsets:
        print(f"   num_points: {best_deepsets[1]}")
    print()
    print("2. Edit train_set_transformer.py:")
    if best_deepsets:
        print(f"   batch_size = {best_deepsets[2]}")
    print()
    
    print("="*70)
    
    # GPU utilization recommendation
    if gpu_mem < 8:
        print("\n⚠ Your GPU has limited memory (< 8GB)")
        print("  • Recommended: Use Deep Sets baseline")
        print("  • Alternative: Reduce Set Transformer size (see GPU_MEMORY_GUIDE.md)")
    elif gpu_mem < 12:
        print("\n✓ Your GPU should handle most configurations")
        print("  • Set Transformer: Use batch_size ≤ 20")
        print("  • Deep Sets: All configurations work")
    else:
        print("\n✓ Your GPU has plenty of memory")
        print("  • All configurations should work fine")
    
    print()


if __name__ == "__main__":
    try:
        test_gpu_memory()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
