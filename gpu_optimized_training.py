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

class GrainQualityDataset(Dataset):
    """GPU-optimized dataset for grain quality control"""
    
    def __init__(self, dataframe, img_dir, transform=None, cache_images=False):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        self.cache_images = cache_images
        self.image_cache = {} if cache_images else None
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['image_name']
        
        # Handle potential NaN or invalid image names
        if pd.isna(img_name) or not isinstance(img_name, str) or img_name.strip() == '':
            print(f"Warning: Invalid image name at index {idx}: {img_name}")
            # Create a dummy black image
            image = Image.new('RGB', (224, 224), color='black')
        else:
            # Use cached image if available
            if self.cache_images and img_name in self.image_cache:
                image = self.image_cache[img_name]
            else:
                img_path = os.path.join(self.img_dir, img_name)
                try:
                    image = Image.open(img_path).convert('RGB')
                except (FileNotFoundError, OSError, IOError) as e:
                    print(f"Warning: Could not load image {img_path}: {e}")
                    # Create a dummy black image
                    image = Image.new('RGB', (224, 224), color='black')
                
                if self.cache_images:
                    self.image_cache[img_name] = image
        
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

class GrainQualityResNetGPU(nn.Module):
    """GPU-optimized ResNet-50 for grain quality control"""
    
    def __init__(self, num_classes=2, dropout=0.5):
        super(GrainQualityResNetGPU, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(weights='IMAGENET1K_V2')  # Updated weights parameter
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get the number of features
        num_features = self.backbone.fc.in_features
        
        # Shared feature processing
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2)
        )
        
        # Task-specific heads
        self.count_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(256, 1)
        )
        
        self.good_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(256, 1)
        )
        
        self.bad_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(256, 1)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout/2),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize the new layers"""
        for m in [self.shared_head, self.count_head, self.good_head, self.bad_head, self.quality_head]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
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

def create_gpu_optimized_transforms(image_size=224):
    """Create GPU-optimized data transforms"""
    
    # Training transforms with aggressive augmentation
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1)  # Random erasing for better generalization
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def multi_task_loss_gpu(outputs, targets, weights=None, device='cuda'):
    """GPU-optimized multi-task loss function"""
    if weights is None:
        weights = {'count': 1.0, 'good': 1.0, 'bad': 1.0, 'quality': 2.0}
    
    # Regression losses (Smooth L1 for better stability)
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

def train_gpu_optimized(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """GPU-optimized training with mixed precision"""
    
    # GPU setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"🚀 Using GPU: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU not available, using CPU")
    
    model.to(device)
    
    # Enable mixed precision training for speed
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # Optimizer with better settings for GPU
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4, 
                           betas=(0.9, 0.999), eps=1e-8)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate * 10,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training tracking
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    print(f"🎯 Training on {len(train_loader.dataset)} samples")
    print(f"🔍 Validating on {len(val_loader.dataset)} samples")
    print(f"⚡ Batch size: {train_loader.batch_size}")
    
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
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Mixed precision forward pass
            if scaler is not None:
                with autocast(device_type='cuda'):
                    outputs = model(images)
                    loss, loss_components = multi_task_loss_gpu(outputs, targets, device=device)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss, loss_components = multi_task_loss_gpu(outputs, targets, device=device)
                loss.backward()
                optimizer.step()
            
            scheduler.step()
            
            train_loss += loss.item()
            for key in train_loss_components:
                train_loss_components[key] += loss_components[key]
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })
        
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
                        loss, loss_components = multi_task_loss_gpu(outputs, targets, device=device)
                else:
                    outputs = model(images)
                    loss, loss_components = multi_task_loss_gpu(outputs, targets, device=device)
                
                val_loss += loss.item()
                for key in val_loss_components:
                    val_loss_components[key] += loss_components[key]
                
                val_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        
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
            }, 'best_grain_model_gpu.pth')
        else:
            patience_counter += 1
        
        # Print epoch summary
        print(f'\n📈 Epoch {epoch+1}/{num_epochs} Summary:')
        print(f'   ⏱️  Time: {epoch_time:.1f}s')
        print(f'   🏃 Train Loss: {avg_train_loss:.4f}')
        print(f'   🎯 Val Loss: {avg_val_loss:.4f}')
        print(f'   🔥 Best Val Loss: {best_val_loss:.4f}')
        print(f'   📊 LR: {scheduler.get_last_lr()[0]:.6f}')
        
        if patience_counter >= patience:
            print(f'⏹️  Early stopping triggered after {patience} epochs without improvement')
            break
    
    return train_losses, val_losses

def main_gpu():
    """GPU-optimized main training function"""
    
    # GPU-optimized configuration
    CONFIG = {
        'IMG_DIR': 'images/',
        'CSV_FILE': 'grain_training_data.csv',
        'BATCH_SIZE': 64,  # Larger batch size for GPU
        'NUM_EPOCHS': 100,
        'LEARNING_RATE': 0.001,
        'IMAGE_SIZE': 256,  # Higher resolution
        'NUM_WORKERS': 4,   # For faster data loading
        'PIN_MEMORY': True,
        'CACHE_IMAGES': True  # Cache small images in memory
    }
    
    print("🌾 GPU-OPTIMIZED GRAIN QUALITY TRAINING")
    print("=" * 50)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"📊 CUDA Version: {torch.version.cuda}")
        print(f"🔥 PyTorch Version: {torch.__version__}")
    else:
        print("⚠️  No GPU detected!")
        CONFIG['BATCH_SIZE'] = 16  # Reduce batch size for CPU
    
    # Load and clean data
    print("\n📁 Loading dataset...")
    df = pd.read_csv(CONFIG['CSV_FILE'])
    
    # Clean the dataset
    df = clean_dataset(df)
    
    if len(df) == 0:
        print("❌ No valid data remaining after cleaning!")
        return
    
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['good'] > df['bad'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['good'] > temp_df['bad'])
    
    # Create transforms
    train_transform, val_transform = create_gpu_optimized_transforms(CONFIG['IMAGE_SIZE'])
    
    # Create datasets
    train_dataset = GrainQualityDataset(train_df, CONFIG['IMG_DIR'], train_transform, 
                                       cache_images=CONFIG['CACHE_IMAGES'])
    val_dataset = GrainQualityDataset(val_df, CONFIG['IMG_DIR'], val_transform)
    test_dataset = GrainQualityDataset(test_df, CONFIG['IMG_DIR'], val_transform)
    
    # Create data loaders with GPU optimization
    train_loader = DataLoader(
        train_dataset, 
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True,
        num_workers=CONFIG['NUM_WORKERS'],
        pin_memory=CONFIG['PIN_MEMORY'],
        persistent_workers=True if CONFIG['NUM_WORKERS'] > 0 else False
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
    
    # Initialize model
    model = GrainQualityResNetGPU(num_classes=2, dropout=0.5)
    
    print(f"\n🎯 Training Configuration:")
    print(f"   📸 Training samples: {len(train_dataset)}")
    print(f"   🔍 Validation samples: {len(val_dataset)}")
    print(f"   🧪 Test samples: {len(test_dataset)}")
    print(f"   📦 Batch size: {CONFIG['BATCH_SIZE']}")
    print(f"   🔄 Epochs: {CONFIG['NUM_EPOCHS']}")
    print(f"   📏 Image size: {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}")
    
    # Train model
    print("\n🚀 Starting GPU-optimized training...")
    train_losses, val_losses = train_gpu_optimized(
        model, train_loader, val_loader,
        num_epochs=CONFIG['NUM_EPOCHS'], 
        learning_rate=CONFIG['LEARNING_RATE']
    )
    
    # Load best model for evaluation
    checkpoint = torch.load('best_grain_model_gpu.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("\n🎉 GPU-optimized training completed!")
    print("📁 Model saved as: best_grain_model_gpu.pth")

if __name__ == "__main__":
    main_gpu()