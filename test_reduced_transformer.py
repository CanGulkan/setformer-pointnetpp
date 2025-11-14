"""Quick test for reduced Set Transformer on 6GB GPU"""
import torch
from models import build_set_transformer

print("Testing Reduced Set Transformer...")
print("="*60)

config = {
    'input_channels': 8,
    'use_color': True,
    'use_intensity': True,
}

print("\nBuilding model...")
model = build_set_transformer(config, 'transformer')

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# Test on CPU first
print("\n1. Testing on CPU...")
x_cpu = torch.randn(12, 4096, 8)
model.eval()
with torch.no_grad():
    logits_cpu, feats_cpu = model(x_cpu)
print(f"✓ CPU test passed! Output shape: {logits_cpu.shape}")

# Test on GPU if available
if torch.cuda.is_available():
    print("\n2. Testing on GPU...")
    device = torch.device('cuda')
    model = model.to(device)
    x_gpu = torch.randn(12, 4096, 8).to(device)
    
    torch.cuda.empty_cache()
    
    with torch.no_grad():
        logits_gpu, feats_gpu = model(x_gpu)
    
    mem_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"✓ GPU test passed! Output shape: {logits_gpu.shape}")
    print(f"✓ GPU memory used: {mem_allocated:.2f} GB")
    
    print("\n" + "="*60)
    print("SUCCESS! Reduced Set Transformer works on your GPU!")
    print("="*60)
    print("\nYou can now train with:")
    print("  python train_set_transformer.py")
else:
    print("\n⚠ CUDA not available, skipping GPU test")

print("\n💡 Remember: Deep Sets is faster for 6GB GPU!")
print("   python train_set_transformer.py --model_type deepsets")
