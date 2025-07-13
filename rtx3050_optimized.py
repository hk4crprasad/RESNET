import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from torch.amp import autocast, GradScaler
import torch.nn.functional as F
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tqdm import tqdm
import gc

class MemoryEfficientGrainDataset(Dataset):
    """Memory-efficient dataset for RTX 3050 4GB"""
    
    def __init__(self, dataframe, img_dir, transform=None, load_in_memory=False):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.load_in_memory = load_in_memory  # Disabled for 4GB GPU
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']
        
        # Handle potential NaN or invalid image names
        if pd.isna(img_name) or not isinstance(img_name, str) or img_name.strip() == '':
            print(f"Warning: Invalid image name at index {idx}: {img_name}")
            # Create a dummy black image
            image = Image.new('RGB', (192, 192), color='black')
        else:
            img_path = os.path.join(self.img_dir, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except (FileNotFoundError, OSError, IOError) as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                # Create a dummy black image
                image = Image.new('RGB', (192, 192), color='black')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels with error handling
        try:
            count = float(self.dataframe.iloc[idx]['count'])
            good = float(self.dataframe.iloc[idx]['good'])
            bad = float(self.dataframe.iloc[idx]['bad'])
        except (ValueError, TypeError):
            print(f"Warning: Invalid numeric values at index {idx}")
            count = good = bad = 0.0
        
        # Create quality classification
        quality_label = 1 if good > bad else 0
        
        return image, {
            'count': torch.tensor(count, dtype=torch.float32),
            'good': torch.tensor(good, dtype=torch.float32),
            'bad': torch.tensor(bad, dtype=torch.float32),
            'quality': torch.tensor(quality_label, dtype=torch.long)
        }

class CompactGrainResNet(nn.Module):
    """Memory-efficient ResNet for RTX 3050 4GB"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(CompactGrainResNet, self).__init__()
        
        # Use ResNet-18 instead of ResNet-50 for memory efficiency
        self.backbone = models.resnet18(weights='IMAGENET1K_V1')
        
        # Remove the final layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get number of features (512 for ResNet-18)
        num_features = self.backbone.fc.in_features
        
        # Compact shared processing
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Compact task-specific heads
        self.count_head = nn.Linear(256, 1)
        self.good_head = nn.Linear(256, 1)
        self.bad_head = nn.Linear(256, 1)
        self.quality_head = nn.Linear(256, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in [self.shared_head, self.count_head, self.good_head, self.bad_head, self.quality_head]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.xavier_uniform_(layer.weight)
                        nn.init.constant_(layer.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Extract features
        features = self.features(x)
        
        # Shared processing
        shared_features = self.shared_head(features)
        
        # Task-specific outputs
        count = self.count_head(shared_features)
        good = self.good_head(shared_features)
        bad = self.bad_head(shared_features)
        quality = self.quality_head(shared_features)
        
        return {
            'count': count.squeeze(),
            'good': good.squeeze(),
            'bad': bad.squeeze(),
            'quality': quality
        }

def create_rtx3050_transforms():
    """Memory-optimized transforms for RTX 3050"""
    
    # Conservative image size for 4GB VRAM
    image_size = 192  # Smaller than usual to save memory
    
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 24, image_size + 24)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def gradient_accumulation_loss(outputs, targets, weights=None):
    """Memory-efficient loss with gradient accumulation"""
    if weights is None:
        weights = {'count': 1.0, 'good': 1.0, 'bad': 1.0, 'quality': 2.0}
    
    # Use Smooth L1 loss for regression (more memory efficient)
    count_loss = F.smooth_l1_loss(outputs['count'], targets['count'])
    good_loss = F.smooth_l1_loss(outputs['good'], targets['good'])
    bad_loss = F.smooth_l1_loss(outputs['bad'], targets['bad'])
    
    # Classification loss
    quality_loss = F.cross_entropy(outputs['quality'], targets['quality'])
    
    # Combined loss
    total_loss = (weights['count'] * count_loss + 
                  weights['good'] * good_loss + 
                  weights['bad'] * bad_loss + 
                  weights['quality'] * quality_loss)
    
    return total_loss, {
        'count_loss': count_loss.item(),
        'good_loss': good_loss.item(),
        'bad_loss': bad_loss.item(),
        'quality_loss': quality_loss.item(),
        'total_loss': total_loss.item()
    }

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def clean_dataset(df):
    """Clean the dataset by removing rows with invalid data"""
    
    print("🧹 Cleaning dataset...")
    initial_count = len(df)
    
    # Remove rows with NaN or invalid image names
    df = df.dropna(subset=['image_name'])
    df = df[df['image_name'].astype(str).str.strip() != '']
    df = df[df['image_name'] != 'nan']
    
    # Remove rows with invalid numeric values
    df = df.dropna(subset=['count', 'good', 'bad'])
    df = df[(df['count'] >= 0) & (df['good'] >= 0) & (df['bad'] >= 0)]
    
    # Ensure count = good + bad
    df = df[df['count'] == (df['good'] + df['bad'])]
    
    # Reset index
    df = df.reset_index(drop=True)
    
    final_count = len(df)
    removed_count = initial_count - final_count
    
    if removed_count > 0:
        print(f"⚠️  Removed {removed_count} problematic rows")
        print(f"✅ Clean dataset: {final_count} images")
    else:
        print("✅ Dataset is already clean")
    
    return df

def train_rtx3050_optimized(model, train_loader, val_loader, num_epochs=75, learning_rate=0.001):
    """RTX 3050 4GB optimized training with gradient accumulation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Clear memory before starting
        clear_gpu_memory()
        
        # Check initial memory
        print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    model.to(device)
    
    # Mixed precision with conservative settings for 4GB
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Gradient accumulation to simulate larger batch sizes
    accumulation_steps = 4  # Simulate batch size of 32 (8*4)
    
    # Conservative optimizer settings
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Simple learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 15
    patience_counter = 0
    
    print(f"🎯 Training Configuration for RTX 3050 4GB:")
    print(f"   📦 Effective batch size: {train_loader.batch_size * accumulation_steps}")
    print(f"   🔄 Gradient accumulation steps: {accumulation_steps}")
    print(f"   📸 Image size: 192x192")
    print(f"   🧠 Model: ResNet-18 (compact)")
    print(f"   ⚡ Mixed precision: {'Enabled' if scaler else 'Disabled'}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_components = {'count_loss': 0, 'good_loss': 0, 'bad_loss': 0, 'quality_loss': 0}
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            images = images.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            
            # Mixed precision forward pass
            if scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss, loss_components = gradient_accumulation_loss(outputs, targets)
                    loss = loss / accumulation_steps  # Scale loss for accumulation
                
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(images)
                loss, loss_components = gradient_accumulation_loss(outputs, targets)
                loss = loss / accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item() * accumulation_steps
            for key in train_loss_components:
                train_loss_components[key] += loss_components[key]
            
            # Memory monitoring
            if batch_idx % 20 == 0 and device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated() / 1e9
                train_pbar.set_postfix({
                    'Loss': f'{loss.item() * accumulation_steps:.4f}',
                    'GPU': f'{memory_used:.1f}GB'
                })
        
        # Clear gradients at epoch end
        optimizer.zero_grad(set_to_none=True)
        clear_gpu_memory()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components = {'count_loss': 0, 'good_loss': 0, 'bad_loss': 0, 'quality_loss': 0}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, targets in val_pbar:
                images = images.to(device, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                
                if scaler is not None:
                    with autocast(device_type='cuda'):
                        outputs = model(images)
                        loss, loss_components = gradient_accumulation_loss(outputs, targets)
                else:
                    outputs = model(images)
                    loss, loss_components = gradient_accumulation_loss(outputs, targets)
                
                val_loss += loss.item()
                for key in val_loss_components:
                    val_loss_components[key] += loss_components[key]
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Clear memory after validation
        clear_gpu_memory()
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step()
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, 'best_grain_model_rtx3050.pth')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary
        if device.type == 'cuda':
            max_memory = torch.cuda.max_memory_allocated() / 1e9
            print(f'\n📈 Epoch {epoch+1}/{num_epochs} Summary:')
            print(f'   ⏱️  Time: {epoch_time:.1f}s')
            print(f'   🏃 Train Loss: {avg_train_loss:.4f}')
            print(f'   🎯 Val Loss: {avg_val_loss:.4f}')
            print(f'   🔥 Best Val Loss: {best_val_loss:.4f}')
            print(f'   💾 Peak GPU Memory: {max_memory:.2f} GB')
            print(f'   📊 LR: {scheduler.get_last_lr()[0]:.6f}')
            
            # Reset peak memory counter
            torch.cuda.reset_peak_memory_stats()
        
        if patience_counter >= patience:
            print(f'⏹️  Early stopping triggered after {patience} epochs without improvement')
            break
    
    return train_losses, val_losses

def main_rtx3050():
    """Main function optimized for RTX 3050 4GB"""
    
    # RTX 3050 4GB Configuration
    CONFIG = {
        'IMG_DIR': 'images/',
        'CSV_FILE': 'grain_training_data.csv',
        'BATCH_SIZE': 8,          # Small batch size for 4GB VRAM
        'NUM_EPOCHS': 75,
        'LEARNING_RATE': 0.001,
        'IMAGE_SIZE': 192,        # Smaller image size
        'NUM_WORKERS': 2,         # Conservative worker count
        'PIN_MEMORY': False,      # Disable for limited memory
        'PREFETCH_FACTOR': 2      # Reduce prefetching
    }
    
    print("🌾 RTX 3050 4GB OPTIMIZED GRAIN TRAINING")
    print("=" * 50)
    
    # GPU check
    if not torch.cuda.is_available():
        print("❌ No GPU detected! This script is optimized for RTX 3050.")
        return
    
    gpu_name = torch.cuda.get_device_name()
    if "3050" not in gpu_name:
        print(f"⚠️  GPU detected: {gpu_name}")
        print("This script is optimized for RTX 3050 4GB. Proceeding anyway...")
    
    # Clear any existing GPU memory
    clear_gpu_memory()
    
    # Load and clean data
    print("\n📁 Loading dataset...")
    df = pd.read_csv(CONFIG['CSV_FILE'])
    
    # Clean the dataset
    df = clean_dataset(df)
    
    if len(df) == 0:
        print("❌ No valid data remaining after cleaning!")
        return
    
    # Stratified split to maintain balance
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, 
                                        stratify=df['good'] > df['bad'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42,
                                      stratify=temp_df['good'] > temp_df['bad'])
    
    # Create memory-efficient transforms
    train_transform, val_transform = create_rtx3050_transforms()
    
    # Create datasets (no memory caching)
    train_dataset = MemoryEfficientGrainDataset(train_df, CONFIG['IMG_DIR'], train_transform)
    val_dataset = MemoryEfficientGrainDataset(val_df, CONFIG['IMG_DIR'], val_transform)
    test_dataset = MemoryEfficientGrainDataset(test_df, CONFIG['IMG_DIR'], val_transform)
    
    # Create memory-efficient data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY'],
        prefetch_factor=CONFIG['PREFETCH_FACTOR'] if CONFIG['NUM_WORKERS'] > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY']
    )
    
    # Initialize compact model
    model = CompactGrainResNet(num_classes=2, dropout=0.3)
    
    print(f"\n🎯 RTX 3050 Training Configuration:")
    print(f"   📸 Training samples: {len(train_dataset)}")
    print(f"   🔍 Validation samples: {len(val_dataset)}")
    print(f"   🧪 Test samples: {len(test_dataset)}")
    print(f"   📦 Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"   🔄 Effective batch size: {CONFIG['BATCH_SIZE'] * 4} (with gradient accumulation)")
    print(f"   ⏳ Epochs: {CONFIG['NUM_EPOCHS']}")
    print(f"   📏 Image size: {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}")
    print(f"   🧠 Model: ResNet-18 (memory efficient)")
    
    # Train model
    print("\n🚀 Starting RTX 3050 optimized training...")
    train_losses, val_losses = train_rtx3050_optimized(
        model, train_loader, val_loader,
        num_epochs=CONFIG['NUM_EPOCHS'],
        learning_rate=CONFIG['LEARNING_RATE']
    )
    
    # Final memory cleanup
    clear_gpu_memory()
    
    print("\n🎉 RTX 3050 training completed!")
    print("📁 Model saved as: best_grain_model_rtx3050.pth")
    
    # Memory usage summary
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"💾 Total GPU Memory: {total_memory:.1f} GB")
        print(f"✅ Training completed successfully on 4GB GPU!")

if __name__ == "__main__":
    main_rtx3050()