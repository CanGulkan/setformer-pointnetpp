"""
Model Comparison Utility
Compare PointNet2 vs Set Transformer vs Deep Sets
"""

import torch
import time
import numpy as np
from models import PointNet2CornerDetection, build_set_transformer


def compare_models(input_channels=10, num_points=4096, batch_size=8, device='cuda'):
    """
    Compare different models on the same input
    
    Args:
        input_channels: Number of input features
        num_points: Number of points per sample
        batch_size: Batch size for testing
        device: 'cuda' or 'cpu'
    """
    
    print("\n" + "="*80)
    print("MODEL COMPARISON: PointNet2 vs Set Transformer vs Deep Sets")
    print("="*80)
    
    # Auto-adjust batch size based on GPU memory
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"GPU Memory: {gpu_mem:.2f} GB")
        
        # Adjust batch size and num_points for available memory
        if gpu_mem < 8:
            if num_points > 2048:
                print(f"⚠ Reducing num_points from {num_points} to 2048 for limited GPU memory")
                num_points = 2048
            if batch_size > 4:
                print(f"⚠ Reducing batch_size from {batch_size} to 4 for limited GPU memory")
                batch_size = 4
    
    config = {
        'input_channels': input_channels,
        'use_color': True,
        'use_intensity': True,
        'use_group_ids': True,
        'use_border_weights': True,
        'normalize': True,
        'num_points': num_points,
        'use_fps': True
    }
    
    # Initialize models
    print("\n1. Initializing Models...")
    print("-" * 80)
    
    print(f"Using device: {device}")
    print(f"Test configuration: batch_size={batch_size}, num_points={num_points}")
    
    models = {}
    
    # PointNet2
    print("\n[PointNet2]")
    models['PointNet2'] = PointNet2CornerDetection(config=config).to(device)
    
    # Set Transformer
    print("\n[Set Transformer]")
    models['SetTransformer'] = build_set_transformer(config, model_type='transformer').to(device)
    
    # Deep Sets
    print("\n[Deep Sets Baseline]")
    models['DeepSets'] = build_set_transformer(config, model_type='deepsets').to(device)
    
    # Count parameters
    print("\n2. Parameter Counts...")
    print("-" * 80)
    
    param_counts = {}
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        param_counts[name] = trainable_params
        print(f"{name:20s}: {trainable_params:>12,} parameters")
    
    # Create test input
    print("\n3. Creating Test Input...")
    print("-" * 80)
    print(f"Batch size: {batch_size}")
    print(f"Points per sample: {num_points}")
    print(f"Input channels: {input_channels}")
    
    point_clouds = torch.randn(batch_size, num_points, input_channels).to(device)
    
    # Inference speed test
    print("\n4. Inference Speed Test (100 iterations)...")
    print("-" * 80)
    
    num_iterations = 100
    inference_times = {}
    
    for name, model in models.items():
        model.eval()
        
        # Clear cache before each model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        try:
            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    if name == 'PointNet2':
                        _ = model(point_clouds)
                    else:
                        _, _ = model(point_clouds)
            
            # Timing
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            start_time = time.time()
            with torch.no_grad():
                for _ in range(num_iterations):
                    if name == 'PointNet2':
                        _ = model(point_clouds)
                    else:
                        _, _ = model(point_clouds)
            
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / num_iterations * 1000  # ms
            inference_times[name] = avg_time
            
            fps = 1000 / avg_time * batch_size  # samples per second
            print(f"{name:20s}: {avg_time:>8.2f} ms/batch  ({fps:>8.1f} samples/sec)")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{name:20s}: ⚠ OOM - Skipping speed test")
                inference_times[name] = float('inf')
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise
    
    # Memory usage
    print("\n5. GPU Memory Usage...")
    print("-" * 80)
    
    if device.type == 'cuda':
        memory_usage = {}
        
        for name, model in models.items():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            try:
                model.eval()
                with torch.no_grad():
                    if name == 'PointNet2':
                        _ = model(point_clouds)
                    else:
                        _, _ = model(point_clouds)
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
                memory_usage[name] = peak_memory
                
                print(f"{name:20s}: {peak_memory:>8.2f} MB")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"{name:20s}: ⚠ OOM - Memory test skipped")
                    memory_usage[name] = float('inf')
                    torch.cuda.empty_cache()
                else:
                    raise
    else:
        print("CUDA not available, skipping memory test")
        memory_usage = {}
    
    # Output comparison
    print("\n6. Output Comparison...")
    print("-" * 80)
    
    outputs = {}
    for name, model in models.items():
        try:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            model.eval()
            with torch.no_grad():
                if name == 'PointNet2':
                    corner_logits, features = model(point_clouds)
                    outputs[name] = {'corner_logits': corner_logits, 'features': features}
                else:
                    corner_logits, features = model(point_clouds)
                    outputs[name] = {'corner_logits': corner_logits, 'features': features}
            
            print(f"\n{name}:")
            print(f"  Corner logits: {outputs[name]['corner_logits'].shape}, "
                  f"range=[{outputs[name]['corner_logits'].min():.3f}, {outputs[name]['corner_logits'].max():.3f}]")
            print(f"  Features: {outputs[name]['features'].shape}, "
                  f"mean={outputs[name]['features'].mean():.3f}, std={outputs[name]['features'].std():.3f}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n{name}:")
                print(f"  ⚠ OOM - Output comparison skipped")
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            else:
                raise
    
    # Summary
    print("\n7. Summary...")
    print("-" * 80)
    
    # Normalize metrics for comparison (lower is better for most)
    max_params = max(param_counts.values())
    max_time = max(inference_times.values())
    
    print(f"\n{'Model':<20s} {'Parameters':<15s} {'Speed':<15s} {'Recommendation':<30s}")
    print("-" * 80)
    
    recommendations = {
        'PointNet2': 'Good balance, geometric',
        'SetTransformer': 'Most expressive, slower',
        'DeepSets': 'Fastest, lightweight baseline'
    }
    
    for name in ['PointNet2', 'SetTransformer', 'DeepSets']:
        params_pct = param_counts[name] / max_params * 100
        time_pct = inference_times[name] / max_time * 100
        
        print(f"{name:<20s} {param_counts[name]:>8,} ({params_pct:>5.1f}%)  "
              f"{inference_times[name]:>6.2f}ms ({time_pct:>5.1f}%)  {recommendations[name]:<30s}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS:")
    print("="*80)
    print("• Start with Deep Sets: Fastest training, good baseline")
    print("• Use Set Transformer: Maximum expressiveness, complex patterns")
    print("• Use PointNet2: Geometric inductive bias, proven architecture")
    print("\nFor corner detection: Try all three and compare results!")
    print("="*80 + "\n")
    
    return {
        'param_counts': param_counts,
        'inference_times': inference_times,
        'outputs': outputs
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare corner detection models')
    parser.add_argument('--input_channels', type=int, default=10, help='Number of input channels')
    parser.add_argument('--num_points', type=int, default=2048, help='Number of points per sample (default: 2048 for GPU safety)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (default: 4 for GPU safety)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device')
    args = parser.parse_args()
    
    print("\n💡 TIP: For 6GB GPU, use: --num_points 2048 --batch_size 4")
    print("💡 TIP: For 8GB+ GPU, use: --num_points 4096 --batch_size 8\n")
    
    try:
        results = compare_models(
            input_channels=args.input_channels,
            num_points=args.num_points,
            batch_size=args.batch_size,
            device=args.device
        )
    except Exception as e:
        print(f"\n❌ Error during comparison: {e}")
        print("\n💡 Try reducing --num_points or --batch_size")
        print("   Example: python compare_models.py --num_points 1024 --batch_size 2")
        raise
