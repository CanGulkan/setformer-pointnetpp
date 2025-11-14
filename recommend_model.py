"""
Model Selector - Helps you choose the right model for your GPU
"""

import torch


def recommend_model():
    """Recommend the best model based on available GPU"""
    
    print("\n" + "="*70)
    print("🤖 MODEL RECOMMENDATION SYSTEM")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("\n❌ No CUDA GPU detected!")
        print("\n💡 Recommendation: Deep Sets (CPU mode)")
        print("   Command: python train_set_transformer.py --model_type deepsets")
        return
    
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    print(f"\n📊 Your GPU: {gpu_name}")
    print(f"💾 Memory: {gpu_mem:.2f} GB")
    print()
    
    # Make recommendations
    if gpu_mem < 6:
        print("⚠️  WARNING: Very limited GPU memory (<6GB)")
        print("\n💡 ONLY OPTION: Deep Sets")
        print("   ✓ Will work with your GPU")
        print("   ✓ Fast training")
        print("   ✓ Good results")
        print("\n📝 Command:")
        print("   python train_set_transformer.py --model_type deepsets")
        
    elif gpu_mem < 8:
        print("⚠️  Limited GPU memory (6-8GB)")
        print("\n💡 RECOMMENDED OPTIONS (in order):")
        print("\n1. Deep Sets ⭐⭐⭐⭐⭐ (BEST CHOICE)")
        print("   ✓ Fast training (~100 min)")
        print("   ✓ Full batch size (40)")
        print("   ✓ Low memory (~1.5 GB)")
        print("   ✓ Excellent results")
        print("\n   📝 Command:")
        print("      python train_set_transformer.py --model_type deepsets")
        
        print("\n2. PointNet2 ⭐⭐⭐⭐ (GREAT CHOICE)")
        print("   ✓ Fast training (~120 min)")
        print("   ✓ Full batch size (40)")
        print("   ✓ Proven architecture")
        print("   ✓ Geometric features")
        print("\n   📝 Command:")
        print("      python train.py")
        
        print("\n3. Set Transformer ⭐ (NOT RECOMMENDED)")
        print("   ✗ Very slow (~400 min)")
        print("   ✗ Tiny batch size (6)")
        print("   ✗ High OOM risk")
        print("   ⚠  Only use if you really need attention mechanism")
        print("\n   📝 Command (not recommended):")
        print("      python train_set_transformer.py")
        
    elif gpu_mem < 12:
        print("✅ Good GPU memory (8-12GB)")
        print("\n💡 RECOMMENDED OPTIONS:")
        print("\n1. Deep Sets ⭐⭐⭐⭐⭐ (FASTEST)")
        print("   ✓ Very fast training")
        print("   ✓ Low memory usage")
        print("   ✓ Great baseline")
        print("\n   📝 Command:")
        print("      python train_set_transformer.py --model_type deepsets")
        
        print("\n2. PointNet2 ⭐⭐⭐⭐⭐ (PROVEN)")
        print("   ✓ Fast training")
        print("   ✓ Excellent results")
        print("   ✓ Geometric bias")
        print("\n   📝 Command:")
        print("      python train.py")
        
        print("\n3. Set Transformer ⭐⭐⭐⭐ (RESEARCH)")
        print("   ✓ Works well at batch size 24")
        print("   ✓ Most expressive")
        print("   ✓ Attention mechanism")
        print("   ~ Medium speed")
        print("\n   📝 Command:")
        print("      python train_set_transformer.py")
        
    else:
        print("🎉 Excellent GPU memory (12GB+)")
        print("\n💡 YOU CAN USE ANY MODEL:")
        print("\n1. Deep Sets ⭐⭐⭐⭐⭐ (FASTEST BASELINE)")
        print("   📝 python train_set_transformer.py --model_type deepsets")
        
        print("\n2. PointNet2 ⭐⭐⭐⭐⭐ (PROVEN PERFORMER)")
        print("   📝 python train.py")
        
        print("\n3. Set Transformer ⭐⭐⭐⭐⭐ (FULL POWER)")
        print("   📝 python train_set_transformer.py")
        
        print("\n💡 Suggestion: Train all three and compare!")
    
    print("\n" + "="*70)
    print("📚 For more details, see:")
    print("   - README_6GB_GPU.md (if you have 6GB GPU)")
    print("   - SET_TRANSFORMER_GUIDE.md (general guide)")
    print("   - GPU_MEMORY_GUIDE.md (memory optimization)")
    print("="*70 + "\n")


def interactive_choice():
    """Interactive model selection"""
    
    recommend_model()
    
    if not torch.cuda.is_available():
        return
    
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_mem >= 8:
        print("\n🤔 Which model do you want to train?")
        print("1. Deep Sets (fastest)")
        print("2. PointNet2 (proven)")
        print("3. Set Transformer (most expressive)")
        print("4. Show me all commands")
        print("5. Exit")
        
        try:
            choice = input("\nEnter your choice (1-5): ").strip()
            
            if choice == '1':
                print("\n✅ Great choice! Run this:")
                print("   python train_set_transformer.py --model_type deepsets")
            elif choice == '2':
                print("\n✅ Excellent choice! Run this:")
                print("   python train.py")
            elif choice == '3':
                print("\n✅ Good choice! Run this:")
                print("   python train_set_transformer.py")
            elif choice == '4':
                print("\n📋 All Training Commands:")
                print("\nDeep Sets:")
                print("   python train_set_transformer.py --model_type deepsets")
                print("\nPointNet2:")
                print("   python train.py")
                print("\nSet Transformer:")
                print("   python train_set_transformer.py")
            elif choice == '5':
                print("\n👋 Goodbye!")
            else:
                print("\n⚠️  Invalid choice. Run this script again.")
        except KeyboardInterrupt:
            print("\n\n👋 Cancelled.")


if __name__ == "__main__":
    try:
        interactive_choice()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        recommend_model()
