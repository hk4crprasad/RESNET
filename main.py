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
import warnings
warnings.filterwarnings('ignore')

class MaxPerformanceTeslaT4GrainDataset(Dataset):
    """Maximum performance Tesla T4 optimized dataset"""
    
    def __init__(self, dataframe, img_dir, transform=None, cache_size=1000):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Use caching for frequently accessed images
        img_name = self.dataframe.iloc[idx]['image_name']
        
        if img_name in self.cache:
            image = self.cache[img_name]
        else:
            # Handle potential NaN or invalid image names
            if pd.isna(img_name) or not isinstance(img_name, str) or img_name.strip() == '':
                print(f"Warning: Invalid image name at index {idx}: {img_name}")
                image = Image.new('RGB', (320, 320), color='black')
            else:
                img_path = os.path.join(self.img_dir, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                    # Cache the image if cache not full
                    if len(self.cache) < self.cache_size:
                        self.cache[img_name] = image
                except (FileNotFoundError, OSError, IOError) as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    image = Image.new('RGB', (320, 320), color='black')
        
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
        
        # Create quality classification (0: bad dominant, 1: good dominant)
        quality_label = 1 if good > bad else 0
        
        return image, {
            'count': torch.tensor(count, dtype=torch.float32),
            'good': torch.tensor(good, dtype=torch.float32),
            'bad': torch.tensor(bad, dtype=torch.float32),
            'quality': torch.tensor(quality_label, dtype=torch.long)
        }

class MaxPerformanceTeslaT4GrainResNet(nn.Module):
    """Maximum performance Tesla T4 optimized ResNet"""
    
    def __init__(self, num_classes=2, dropout=0.3):
        super(MaxPerformanceTeslaT4GrainResNet, self).__init__()
        
        # Use ResNet-50 with optimizations
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get the number of features
        num_features = self.backbone.fc.in_features
        
        # More efficient feature processing
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.BatchNorm1d(num_features),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout/2)
        )
        
        # Streamlined task-specific heads
        self.count_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/3),
            nn.Linear(128, 1)
        )
        
        self.good_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/3),
            nn.Linear(128, 1)
        )
        
        self.bad_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/3),
            nn.Linear(128, 1)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize the new layers"""
        for m in [self.shared_head, self.count_head, self.good_head, self.bad_head, self.quality_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
                    nn.init.constant_(layer.bias, 0)
                elif isinstance(layer, nn.BatchNorm1d):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        # Extract features with ResNet backbone
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

def create_max_performance_transforms():
    """Create maximum performance transforms with higher resolution"""
    
    # Higher resolution for better performance
    image_size = 320  # Increased from 256
    
    # More aggressive augmentations
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 40, image_size + 40)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(20),  # Increased rotation
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2))
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def optimized_multi_task_loss(outputs, targets, weights=None):
    """Optimized multi-task loss function"""
    if weights is None:
        weights = {'count': 1.0, 'good': 1.0, 'bad': 1.0, 'quality': 2.5}
    
    # Use Huber loss for regression (more robust than smooth L1)
    count_loss = F.huber_loss(outputs['count'], targets['count'], delta=1.0)
    good_loss = F.huber_loss(outputs['good'], targets['good'], delta=1.0)
    bad_loss = F.huber_loss(outputs['bad'], targets['bad'], delta=1.0)
    
    # Classification loss with label smoothing
    quality_loss = F.cross_entropy(outputs['quality'], targets['quality'], label_smoothing=0.1)
    
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

def aggressive_clear_gpu_memory():
    """Aggressively clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()

def train_max_performance_tesla_t4(model, train_loader, val_loader, num_epochs=100, learning_rate=0.002):
    """Maximum performance Tesla T4 training with all optimizations"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Set memory growth to False for maximum performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable channels last memory format for better performance
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Clear memory before starting
        aggressive_clear_gpu_memory()
        
        # Check initial memory
        print(f"GPU Memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    model.to(device, memory_format=torch.channels_last)
    
    # Compile model for PyTorch 2.0+ (if available)
    try:
        model = torch.compile(model, mode='max-autotune')
        print("✅ Model compiled with torch.compile for maximum performance")
    except:
        print("⚠️  torch.compile not available, using standard model")
    
    # Full mixed precision for Tesla T4 with optimizations
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # More aggressive optimizer settings
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=2e-4,  # Increased weight decay
        betas=(0.9, 0.95),  # Optimized betas
        eps=1e-6,           # Smaller epsilon
        amsgrad=True        # Enable AMSGrad
    )
    
    # More aggressive learning rate schedule
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate * 15,  # Higher max LR
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.05,  # Shorter warmup
        anneal_strategy='cos',
        div_factor=25,    # More aggressive initial LR
        final_div_factor=1000  # Lower final LR
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20  # Increased patience
    patience_counter = 0
    
    print(f"🎯 Maximum Performance Tesla T4 Configuration:")
    print(f"   📦 Batch size: {train_loader.batch_size}")
    print(f"   📸 Image size: 320x320")
    print(f"   🧠 Model: ResNet-50 (compiled)")
    print(f"   ⚡ Mixed precision: {'Enabled' if scaler else 'Disabled'}")
    print(f"   🔥 Channels last: Enabled")
    print(f"   💾 Expected memory usage: ~13-15GB")
    print(f"   🚄 Data workers: {train_loader.num_workers}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_components = {'count_loss': 0, 'good_loss': 0, 'bad_loss': 0, 'quality_loss': 0}
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                         leave=False, ncols=120)
        
        for batch_idx, (images, targets) in enumerate(train_pbar):
            # Use channels last memory format
            images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            
            optimizer.zero_grad(set_to_none=True)
            
            # Mixed precision forward pass
            if scaler is not None:
                with autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss, loss_components = optimized_multi_task_loss(outputs, targets)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss, loss_components = optimized_multi_task_loss(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            for key in train_loss_components:
                train_loss_components[key] += loss_components[key]
            
            # Update progress bar every batch
            memory_used = torch.cuda.memory_allocated() / 1e9 if device.type == 'cuda' else 0
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.7f}',
                'GPU': f'{memory_used:.1f}GB',
                'Batch': f'{batch_idx+1}/{len(train_loader)}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components = {'count_loss': 0, 'good_loss': 0, 'bad_loss': 0, 'quality_loss': 0}
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]', 
                           leave=False, ncols=120)
            for images, targets in val_pbar:
                images = images.to(device, memory_format=torch.channels_last, non_blocking=True)
                targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
                
                if scaler is not None:
                    with autocast(device_type='cuda', dtype=torch.float16):
                        outputs = model(images)
                        loss, loss_components = optimized_multi_task_loss(outputs, targets)
                else:
                    outputs = model(images)
                    loss, loss_components = optimized_multi_task_loss(outputs, targets)
                
                val_loss += loss.item()
                for key in val_loss_components:
                    val_loss_components[key] += loss_components[key]
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }, 'best_grain_model_max_performance_tesla_t4.pth')
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start_time
        
        # Print epoch summary every 3 epochs
        if (epoch + 1) % 3 == 0 or epoch == 0:
            if device.type == 'cuda':
                max_memory = torch.cuda.max_memory_allocated() / 1e9
                current_memory = torch.cuda.memory_allocated() / 1e9
                print(f'\n📈 Epoch {epoch+1}/{num_epochs} Performance Summary:')
                print(f'   ⚡ Time: {epoch_time:.1f}s | Speed: {len(train_loader)*train_loader.batch_size/epoch_time:.0f} samples/s')
                print(f'   🏃 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
                print(f'   🔥 Best Val Loss: {best_val_loss:.4f} | Patience: {patience_counter}/{patience}')
                print(f'   💾 GPU Memory: {current_memory:.1f}GB | Peak: {max_memory:.1f}GB')
                print(f'   📊 Learning Rate: {scheduler.get_last_lr()[0]:.7f}')
                
                # Reset peak memory counter
                torch.cuda.reset_peak_memory_stats()
        
        if patience_counter >= patience:
            print(f'⏹️  Early stopping triggered after {patience} epochs without improvement')
            break
    
    return train_losses, val_losses

def main_max_performance_tesla_t4():
    """Main function with maximum performance optimizations for Tesla T4"""
    
    # Maximum Performance Tesla T4 Configuration
    CONFIG = {
        'IMG_DIR': 'images/',
        'CSV_FILE': 'grain_training_data.csv',
        'BATCH_SIZE': 64,         # Increased from 32 to maximize GPU usage
        'NUM_EPOCHS': 100,        # More epochs for better convergence
        'LEARNING_RATE': 0.002,   # Slightly higher learning rate
        'IMAGE_SIZE': 320,        # Higher resolution from 256
        'NUM_WORKERS': 8,         # Maximum workers for fast data loading
        'PIN_MEMORY': True,       # Enable for faster GPU transfer
        'PREFETCH_FACTOR': 4,     # Prefetch more batches
        'PERSISTENT_WORKERS': True # Keep workers alive between epochs
    }
    
    print("🌾 MAXIMUM PERFORMANCE TESLA T4 GRAIN TRAINING")
    print("=" * 60)
    
    # GPU check and optimization
    if not torch.cuda.is_available():
        print("❌ No GPU detected!")
        return
    
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 GPU: {gpu_name}")
    print(f"📊 Total Memory: {gpu_memory:.1f} GB")
    
    # Set aggressive CUDA settings
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # Clear any existing GPU memory
    aggressive_clear_gpu_memory()
    
    # Load and clean data
    print("\n📁 Loading and optimizing dataset...")
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
    
    # Create high-performance transforms
    train_transform, val_transform = create_max_performance_transforms()
    
    # Create datasets with caching
    train_dataset = MaxPerformanceTeslaT4GrainDataset(train_df, CONFIG['IMG_DIR'], train_transform, cache_size=2000)
    val_dataset = MaxPerformanceTeslaT4GrainDataset(val_df, CONFIG['IMG_DIR'], val_transform, cache_size=1000)
    test_dataset = MaxPerformanceTeslaT4GrainDataset(test_df, CONFIG['IMG_DIR'], val_transform, cache_size=500)
    
    # Create high-performance data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY'],
        prefetch_factor=CONFIG['PREFETCH_FACTOR'],
        persistent_workers=CONFIG['PERSISTENT_WORKERS'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']//2,  # Less workers for validation
        pin_memory=CONFIG['PIN_MEMORY'],
        prefetch_factor=CONFIG['PREFETCH_FACTOR']//2,
        persistent_workers=CONFIG['PERSISTENT_WORKERS']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False,
        num_workers=CONFIG['NUM_WORKERS']//2,
        pin_memory=CONFIG['PIN_MEMORY']
    )
    
    # Initialize optimized model
    model = MaxPerformanceTeslaT4GrainResNet(num_classes=2, dropout=0.3)
    
    # Calculate expected memory usage
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🎯 Maximum Performance Tesla T4 Configuration:")
    print(f"   📸 Training samples: {len(train_dataset):,}")
    print(f"   🔍 Validation samples: {len(val_dataset):,}")
    print(f"   🧪 Test samples: {len(test_dataset):,}")
    print(f"   📦 Batch size: {CONFIG['BATCH_SIZE']} (maximized)")
    print(f"   ⏳ Epochs: {CONFIG['NUM_EPOCHS']}")
    print(f"   📏 Image size: {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']} (high-res)")
    print(f"   🧠 Model: ResNet-50 (compiled + optimized)")
    print(f"   🔢 Total parameters: {total_params:,}")
    print(f"   🎯 Trainable parameters: {trainable_params:,}")
    print(f"   👥 Workers: {CONFIG['NUM_WORKERS']} (maximum)")
    print(f"   💾 Pin memory: {CONFIG['PIN_MEMORY']}")
    print(f"   🚄 Prefetch factor: {CONFIG['PREFETCH_FACTOR']}")
    print(f"   💪 Persistent workers: {CONFIG['PERSISTENT_WORKERS']}")
    print(f"   🔥 Expected GPU usage: ~13-15GB (95%+ utilization)")
    
    # Train model with maximum performance
    print("\n🚀 Starting MAXIMUM PERFORMANCE Tesla T4 training...")
    print("🔥 All optimizations enabled - using full GPU potential!")
    
    train_losses, val_losses = train_max_performance_tesla_t4(
        model, train_loader, val_loader,
        num_epochs=CONFIG['NUM_EPOCHS'],
        learning_rate=CONFIG['LEARNING_RATE']
    )
    
    # Final memory cleanup
    aggressive_clear_gpu_memory()
    
    print("\n🎉 MAXIMUM PERFORMANCE Tesla T4 training completed!")
    print("📁 Model saved as: best_grain_model_max_performance_tesla_t4.pth")
    
    # Final performance summary
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        utilization = (peak_memory / total_memory) * 100
        
        print(f"\n💪 PERFORMANCE SUMMARY:")
        print(f"   💾 Peak GPU Memory: {peak_memory:.2f} GB / {total_memory:.1f} GB")
        print(f"   📊 GPU Utilization: {utilization:.1f}%")
        print(f"   ⚡ Batch size: {CONFIG['BATCH_SIZE']} samples")
        print(f"   🖼️  Image resolution: {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}")
        print(f"   🚄 Data workers: {CONFIG['NUM_WORKERS']}")
        print(f"   🔥 All optimizations: ENABLED")
        print(f"   ✅ Training completed with MAXIMUM PERFORMANCE!")

if __name__ == "__main__":
    main_max_performance_tesla_t4()