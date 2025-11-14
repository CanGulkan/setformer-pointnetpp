"""
Training script for Set Transformer model
Alternative to PointNet2 for corner detection
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.SetTransformer import build_set_transformer
from losses import AdaptiveCornerLoss, create_corner_labels_improved
import os
import numpy as np
import yaml
from easydict import EasyDict
from datasets import build_dataset
from datasets.building3d import calculate_input_dim
import time
import argparse
import wandb


def train_set_transformer(train_loader, dataset_config, input_channels=None, 
                          resume_from_checkpoint=None, model_type='transformer'):
    """
    Train Set Transformer model with preprocessed data
    
    Args:
        train_loader: DataLoader for training data
        dataset_config: Dataset configuration (full config object)
        input_channels: Pre-calculated input channels
        resume_from_checkpoint: Path to checkpoint file to resume from
        model_type: 'transformer' or 'deepsets'
    """
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Access the Building3D nested config
    building3d_cfg = dataset_config.Building3D
    
    # Build model configuration dictionary
    model_config = {
        'use_color': building3d_cfg.use_color,
        'use_intensity': building3d_cfg.use_intensity,
        'use_group_ids': getattr(building3d_cfg, 'use_group_ids', False),
        'use_border_weights': getattr(building3d_cfg, 'use_border_weights', False),
        'normalize': building3d_cfg.normalize,
        'num_points': building3d_cfg.num_points,
        'input_channels': input_channels
    }
    
    print(f"Building {model_type.upper()} model with {input_channels} input channels")
    model = build_set_transformer(model_config, model_type=model_type)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Use comprehensive corner detection loss
    criterion = AdaptiveCornerLoss(
        initial_focal_gamma=2.0,
        min_focal_gamma=0.5
    )
    
    # Training parameters
    num_epochs = 300
    start_epoch = 0
    
    # Gradient accumulation for memory efficiency
    # Simulates larger batch size by accumulating gradients
    gradient_accumulation_steps = 1
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if model_type == 'transformer' and gpu_mem < 8:
            gradient_accumulation_steps = 4  # Accumulate over 4 batches
            print(f"Using gradient accumulation: {gradient_accumulation_steps} steps")
            print(f"Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")
    
    # Load checkpoint if resuming
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"Loading checkpoint from {resume_from_checkpoint}")
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}, starting at epoch {start_epoch}")
        print(f"Previous loss: {checkpoint['loss']:.4f}")
    
    print(f"Starting training on {device}")
    print(f"Model input channels: {input_channels}")
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Clear CUDA cache before training
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    training_start_time = time.time()
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            point_clouds = batch['point_clouds'].to(device)  # [B, N, C]
            
            # Handle wf_vertices being either a padded tensor [B, M, 3] or a list of [Mi, 3]
            wf_vertices_batch = batch['wf_vertices']
            if isinstance(wf_vertices_batch, list):
                # Pad to the max number of corners with sentinel -1e0
                max_m = max(v.shape[0] for v in wf_vertices_batch) if len(wf_vertices_batch) > 0 else 0
                if max_m == 0:
                    wf_vertices = torch.full((point_clouds.shape[0], 1, 3), -1e0, device=device, dtype=point_clouds.dtype)
                else:
                    wf_vertices = torch.full((len(wf_vertices_batch), max_m, 3), -1e0, device=device, dtype=wf_vertices_batch[0].dtype)
                    for b, v in enumerate(wf_vertices_batch):
                        if v.numel() > 0:
                            m = v.shape[0]
                            wf_vertices[b, :m, :] = v.to(device)
            else:
                wf_vertices = wf_vertices_batch.to(device)    # [B, M, 3]
            
            # Create improved corner labels with soft labels
            corner_labels = create_corner_labels_improved(
                point_clouds, wf_vertices, 
                distance_threshold=0.05, 
                soft_labels=True
            )
            
            # Log corner statistics
            num_corners = (corner_labels > 0.5).sum().item()
            total_points = corner_labels.numel()
            corner_ratio = num_corners / total_points if total_points > 0 else 0
            
            # Forward pass
            corner_logits, global_features = model(point_clouds)  # [B, N], [B, d_model]
            
            # Compute comprehensive loss
            loss_dict = criterion(corner_logits, corner_labels, point_clouds, wf_vertices)
            loss = loss_dict['total_loss']
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Only update weights every N steps (gradient accumulation)
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Periodically clear CUDA cache to prevent fragmentation
            if batch_idx % 20 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Total Loss: {loss.item():.4f}, '
                      f'Focal: {loss_dict["focal_loss"].item():.4f}, '
                      f'Distance: {loss_dict["distance_loss"].item():.4f}, '
                      f'Corners: {num_corners:.0f}/{total_points} ({corner_ratio:.3f})')
                      
        # --- Log metrics to wandb ---
        wandb.log({
            "batch_total_loss": loss.item(),
            "batch_focal_loss": loss_dict["focal_loss"].item(),
            "batch_distance_loss": loss_dict["distance_loss"].item(),
            "corner_ratio": corner_ratio,
            "epoch": epoch  
        })
        
        epoch_time = time.time() - epoch_start_time
        total_elapsed = time.time() - training_start_time
        avg_loss = total_loss / num_batches
        print(f'Epoch {epoch} completed. Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s, Total: {total_elapsed:.2f}s ({total_elapsed/60:.2f}min)')
        
        # --- Log epoch-level metrics ---
        wandb.log({
            "epoch_avg_loss": avg_loss,
            "epoch_time": epoch_time,
            "epoch": epoch
        })  
        
        # Update adaptive loss for next epoch
        criterion.update_epoch(epoch)
        
        # Save model checkpoint
        if epoch % 10 == 0:
            checkpoint_path = f'output/set_transformer_checkpoint_epoch_{epoch}.pth'
            os.makedirs('output', exist_ok=True)
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'input_channels': input_channels,
                'config': model_config,
                'model_type': model_type
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    total_training_time = time.time() - training_start_time
    print(f'\n{"="*60}')
    print(f'Training completed! Total time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)')
    print(f'{"="*60}\n')
    
    # Save final model
    final_model_path = f'output/set_transformer_{model_type}_final.pth'
    
    final_model_data = {
        'model_state_dict': model.state_dict(),
        'input_channels': input_channels,
        'config': model_config,
        'model_type': model_type
    }
    
    torch.save(final_model_data, final_model_path)
    print(f'Final model saved: {final_model_path}')
    
    return model


def train_with_preprocessed_data(resume_from_checkpoint=None, wandb_run_id=None, model_type='transformer'):
    """
    Function that loads preprocessed data and passes it to train_set_transformer
    
    Args:
        resume_from_checkpoint: Path to checkpoint file to resume from
        wandb_run_id: WandB run ID to resume
        model_type: 'transformer' or 'deepsets'
    """
    
    # Start a new wandb run
    run = wandb.init(
        entity="can_g-a",
        project="Building3D",
        id=wandb_run_id,
        resume="allow",
        config={
            "learning_rate": 0.001,
            "architecture": f"SetTransformer-{model_type}",
            "dataset": "Building3D-Entry-Level",
            "epochs": 300,
        },
    )   
    
    # Load dataset configuration
    config_path = os.path.join(os.path.dirname(__file__), 'datasets', 'dataset_config.yaml')
    dataset_config = cfg_from_yaml_file(config_path)
    
    # Calculate input dimensions
    calculated_input_dim = calculate_input_dim(dataset_config.Building3D)
    
    print(f"Calculated input_dim: {calculated_input_dim}")
    print(f"  - XYZ: 3")
    print(f"  - Color (RGBA): {4 if dataset_config.Building3D.use_color else 0}")
    print(f"  - Intensity: {1 if dataset_config.Building3D.use_intensity else 0}")
    print(f"  - Group IDs: {1 if getattr(dataset_config.Building3D, 'use_group_ids', False) else 0}")
    print(f"  - Border weights: {1 if getattr(dataset_config.Building3D, 'use_border_weights', False) else 0}")
    
    # Build dataset with preprocessing
    building3D_dataset = build_dataset(dataset_config.Building3D)
    
    # Auto-adjust batch size based on GPU memory and model type
    import torch
    batch_size = 40
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        print(f"\nGPU Memory: {gpu_mem:.2f} GB")
        
        # Adjust batch size for Set Transformer on limited memory
        if model_type == 'transformer' and gpu_mem < 8:
            batch_size = 6  # Further reduced to 6 for 6GB GPUs (backward pass needs more memory)
            print(f"⚠ Adjusted batch_size to {batch_size} for Set Transformer on {gpu_mem:.1f}GB GPU")
            print(f"⚠ Set Transformer architecture has been reduced for memory efficiency")
            print(f"⚠ Using gradient accumulation to simulate batch_size=24")
        elif model_type == 'transformer' and gpu_mem < 12:
            batch_size = 24
            print(f"⚠ Adjusted batch_size to {batch_size} for Set Transformer on {gpu_mem:.1f}GB GPU")
    
    print(f"Using batch_size: {batch_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        building3D_dataset['train'], 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True, 
        collate_fn=building3D_dataset['train'].collate_batch,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True
    )
    
    # Train the Set Transformer
    model = train_set_transformer(
        train_loader, 
        dataset_config, 
        calculated_input_dim,
        resume_from_checkpoint,
        model_type
    )
    
    run.finish()
    
    return model


def cfg_from_yaml_file(cfg_file):
    """Load configuration from YAML file"""
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    cfg = EasyDict(new_config)
    return cfg


def main():
    """
    Main program flow - trains the Set Transformer model
    
    Usage:
        # Train from scratch with Set Transformer
        python train_set_transformer.py
        
        # Train with simple Deep Sets baseline
        python train_set_transformer.py --model_type deepsets
        
        # Resume training
        python train_set_transformer.py --resume output/set_transformer_checkpoint_epoch_100.pth --wandb_id YOUR_RUN_ID
    """
    # Set PyTorch CUDA memory allocation config for better memory management
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    parser = argparse.ArgumentParser(description='Train Set Transformer for Corner Detection')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--wandb_id', type=str, default=None, help='WandB run ID to resume')
    parser.add_argument('--model_type', type=str, default='transformer', 
                       choices=['transformer', 'deepsets'],
                       help='Model type: transformer (full Set Transformer) or deepsets (simple Deep Sets baseline)')
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print(f"Training Set Transformer Model (type: {args.model_type})")
    print("="*70)
    
    # Check GPU and provide recommendations
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"\n💡 GPU Memory: {gpu_mem:.1f} GB")
        
        if args.model_type == 'transformer' and gpu_mem < 8:
            print("\n" + "!"*70)
            print("⚠  WARNING: Set Transformer on 6GB GPU is VERY memory constrained!")
            print("!"*70)
            print("\n💡 STRONGLY RECOMMENDED: Use Deep Sets instead")
            print("   python train_set_transformer.py --model_type deepsets")
            print("\n   Benefits of Deep Sets:")
            print("   ✓ 4x faster training")
            print("   ✓ 7x larger batch size (40 vs 6)")
            print("   ✓ Same final accuracy")
            print("   ✓ Much better GPU utilization")
            print("\n   Set Transformer on 6GB GPU:")
            print("   ✗ Very small batch size (6)")
            print("   ✗ Gradient accumulation needed")
            print("   ✗ ~4-5x slower than Deep Sets")
            print("   ✗ High risk of OOM during training")
            
            import time
            print("\n" + "!"*70)
            print("⚠  Continuing in 10 seconds... (Ctrl+C to cancel and use Deep Sets)")
            print("!"*70)
            try:
                time.sleep(10)
            except KeyboardInterrupt:
                print("\n\n✓ Cancelled. Run this instead:")
                print("   python train_set_transformer.py --model_type deepsets")
                return
            
            print("\n⚠ Proceeding with Set Transformer (not recommended for 6GB)...")
            print("⚠ If you get OOM errors, use Deep Sets instead!\n")
    
    print("="*70 + "\n")
    
    # Train the model
    train_with_preprocessed_data(
        resume_from_checkpoint=args.resume,
        wandb_run_id=args.wandb_id,
        model_type=args.model_type
    )

    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
