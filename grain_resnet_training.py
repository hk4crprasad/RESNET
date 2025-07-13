import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class GrainQualityDataset(Dataset):
    """Custom dataset for grain quality control"""
    
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        
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
            img_path = os.path.join(self.img_dir, img_name)
            
            try:
                image = Image.open(img_path).convert('RGB')
            except (FileNotFoundError, OSError, IOError) as e:
                print(f"Warning: Could not load image {img_path}: {e}")
                # Create a dummy black image
                image = Image.new('RGB', (224, 224), color='black')
        
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

class GrainQualityResNet(nn.Module):
    """Modified ResNet-50 for grain quality control with multiple outputs"""
    
    def __init__(self, num_classes=2):
        super(GrainQualityResNet, self).__init__()
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=True)
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Get the number of features from ResNet-50
        num_features = self.backbone.fc.in_features
        
        # Custom heads for different outputs
        self.count_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.good_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.bad_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1)
        )
        
        self.quality_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Extract features
        features = self.features(x)
        features = torch.flatten(features, 1)
        
        # Multiple outputs
        count = self.count_head(features)
        good = self.good_head(features)
        bad = self.bad_head(features)
        quality = self.quality_head(features)
        
        return {
            'count': count.squeeze(),
            'good': good.squeeze(),
            'bad': bad.squeeze(),
            'quality': quality
        }

def create_data_transforms():
    """Create data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def multi_task_loss(outputs, targets, weights=None):
    """Multi-task loss function"""
    if weights is None:
        weights = {'count': 1.0, 'good': 1.0, 'bad': 1.0, 'quality': 2.0}
    
    # Regression losses (MSE)
    count_loss = nn.MSELoss()(outputs['count'], targets['count'])
    good_loss = nn.MSELoss()(outputs['good'], targets['good'])
    bad_loss = nn.MSELoss()(outputs['bad'], targets['bad'])
    
    # Classification loss (CrossEntropy)
    quality_loss = nn.CrossEntropyLoss()(outputs['quality'], targets['quality'])
    
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

def train_model(model, train_loader, val_loader, num_epochs=50, learning_rate=0.001):
    """Train the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_loss_components = {'count_loss': 0, 'good_loss': 0, 'bad_loss': 0, 'quality_loss': 0}
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_components = multi_task_loss(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for key in train_loss_components:
                train_loss_components[key] += loss_components[key]
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_loss_components = {'count_loss': 0, 'good_loss': 0, 'bad_loss': 0, 'quality_loss': 0}
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = {k: v.to(device) for k, v in targets.items()}
                
                outputs = model(images)
                loss, loss_components = multi_task_loss(outputs, targets)
                
                val_loss += loss.item()
                for key in val_loss_components:
                    val_loss_components[key] += loss_components[key]
        
        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_grain_model.pth')
        
        # Print progress
        if epoch % 5 == 0:
            print(f'Epoch [{epoch}/{num_epochs}]')
            print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'Train - Count: {train_loss_components["count_loss"]/len(train_loader):.4f}, '
                  f'Good: {train_loss_components["good_loss"]/len(train_loader):.4f}, '
                  f'Bad: {train_loss_components["bad_loss"]/len(train_loader):.4f}, '
                  f'Quality: {train_loss_components["quality_loss"]/len(train_loader):.4f}')
        
        scheduler.step()
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate the model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = {'count': [], 'good': [], 'bad': [], 'quality': []}
    all_targets = {'count': [], 'good': [], 'bad': [], 'quality': []}
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}
            
            outputs = model(images)
            
            # Store predictions and targets
            all_predictions['count'].extend(outputs['count'].cpu().numpy())
            all_predictions['good'].extend(outputs['good'].cpu().numpy())
            all_predictions['bad'].extend(outputs['bad'].cpu().numpy())
            all_predictions['quality'].extend(torch.argmax(outputs['quality'], dim=1).cpu().numpy())
            
            all_targets['count'].extend(targets['count'].cpu().numpy())
            all_targets['good'].extend(targets['good'].cpu().numpy())
            all_targets['bad'].extend(targets['bad'].cpu().numpy())
            all_targets['quality'].extend(targets['quality'].cpu().numpy())
    
    # Calculate metrics
    quality_accuracy = accuracy_score(all_targets['quality'], all_predictions['quality'])
    
    # Calculate MAE for regression tasks
    count_mae = np.mean(np.abs(np.array(all_predictions['count']) - np.array(all_targets['count'])))
    good_mae = np.mean(np.abs(np.array(all_predictions['good']) - np.array(all_targets['good'])))
    bad_mae = np.mean(np.abs(np.array(all_predictions['bad']) - np.array(all_targets['bad'])))
    
    print(f"Quality Classification Accuracy: {quality_accuracy:.4f}")
    print(f"Count MAE: {count_mae:.4f}")
    print(f"Good Count MAE: {good_mae:.4f}")
    print(f"Bad Count MAE: {bad_mae:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_targets['quality'], all_predictions['quality'], 
                              target_names=['Bad Quality', 'Good Quality']))
    
    return all_predictions, all_targets

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

def plot_training_curves(train_losses, val_losses):
    """Plot training and validation curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    plt.show()

def main():
    """Main training function"""
    # Configuration
    IMG_DIR = 'images/'  # Update this path
    CSV_FILE = 'grain_training_data.csv'      # Update this path
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Load data
    # Expected CSV format: image_name, count, good, bad
    df = pd.read_csv(CSV_FILE)
    
    # Clean the dataset
    df = clean_dataset(df)
    
    if len(df) == 0:
        print("❌ No valid data remaining after cleaning!")
        return
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    # Create transforms
    train_transform, val_transform = create_data_transforms()
    
    # Create datasets
    train_dataset = GrainQualityDataset(train_df, IMG_DIR, train_transform)
    val_dataset = GrainQualityDataset(val_df, IMG_DIR, val_transform)
    test_dataset = GrainQualityDataset(test_df, IMG_DIR, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = GrainQualityResNet(num_classes=2)
    
    print("Starting training...")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Train model
    train_losses, val_losses = train_model(model, train_loader, val_loader, 
                                         num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE)
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses)
    
    # Load best model for evaluation
    model.load_state_dict(torch.load('best_grain_model.pth'))
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    predictions, targets = evaluate_model(model, test_loader)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(targets['quality'], predictions['quality'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Bad Quality', 'Good Quality'],
                yticklabels=['Bad Quality', 'Good Quality'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    print("Training completed!")

if __name__ == "__main__":
    main()