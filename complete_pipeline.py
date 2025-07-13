"""
Complete pipeline for grain quality assessment using ResNet-50
This script combines dataset conversion and model training
"""

import os
import sys
import subprocess

def run_complete_pipeline(use_gpu_optimized=True, use_rtx3050_optimized=False):
    """
    Run the complete grain quality assessment pipeline
    """
    
    print("🌾 GRAIN QUALITY ASSESSMENT PIPELINE")
    print("=" * 50)
    
    # Configuration based on GPU type
    if use_rtx3050_optimized:
        config = {
            'input_annotations': '_annotations.csv.xlsx',
            'output_training_data': 'grain_training_data.csv',
            'image_directory': 'images/',
            'model_save_path': 'best_grain_model_rtx3050.pth',
            'batch_size': 8,  # Small batch for 4GB VRAM
            'epochs': 75,
            'learning_rate': 0.001,
            'image_size': 192,  # Reduced resolution
            'num_workers': 2,
            'training_type': 'RTX 3050 4GB Optimized'
        }
    elif use_gpu_optimized:
        config = {
            'input_annotations': '_annotations.csv.xlsx',
            'output_training_data': 'grain_training_data.csv',
            'image_directory': 'images/',
            'model_save_path': 'best_grain_model_gpu.pth',
            'batch_size': 64,  # Larger batch size for GPU
            'epochs': 100,
            'learning_rate': 0.001,
            'image_size': 256,  # Higher resolution for GPU
            'num_workers': 4,
            'training_type': 'High-End GPU Optimized'
        }
    else:
        config = {
            'input_annotations': '_annotations.csv.xlsx',
            'output_training_data': 'grain_training_data.csv',
            'image_directory': 'images/',
            'model_save_path': 'best_grain_model.pth',
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'image_size': 224,
            'num_workers': 2,
            'training_type': 'CPU/Standard'
        }
    
    # Step 1: Check if files exist
    print("\n📁 STEP 1: Checking input files...")
    
    if not os.path.exists(config['input_annotations']):
        print(f"❌ Error: Annotation file '{config['input_annotations']}' not found!")
        print("Please ensure the annotation file is in the current directory.")
        return False
    
    if not os.path.exists(config['image_directory']):
        print(f"❌ Error: Image directory '{config['image_directory']}' not found!")
        print("Please create the image directory and place your grain images there.")
        return False
    
    print("✅ Input files found!")
    
    # Step 2: Convert dataset
    print("\n🔄 STEP 2: Converting dataset...")
    
    try:
        # Import and run the converter
        from dataset_converter import convert_grain_annotations_to_training_format, create_image_quality_report
        
        # Convert annotations to training format
        training_df = convert_grain_annotations_to_training_format(
            config['input_annotations'], 
            config['output_training_data']
        )
        
        # Create quality report
        create_image_quality_report(training_df)
        
        print("✅ Dataset conversion completed!")
        
    except Exception as e:
        print(f"❌ Error during dataset conversion: {e}")
        return False
    
    # Step 3: Train model
    print(f"\n🚀 STEP 3: Training {config['training_type']} model...")
    
    try:
        if use_rtx3050_optimized:
            # Import RTX 3050 optimized training
            from rtx3050_optimized import main_rtx3050
            print(f"RTX 3050 4GB Training configuration:")
            print(f"  - Images directory: {config['image_directory']}")
            print(f"  - Training data: {config['output_training_data']}")
            print(f"  - Batch size: {config['batch_size']} (effective: {config['batch_size'] * 4} with gradient accumulation)")
            print(f"  - Epochs: {config['epochs']}")
            print(f"  - Learning rate: {config['learning_rate']}")
            print(f"  - Image size: {config['image_size']}x{config['image_size']}")
            print(f"  - Model: ResNet-18 (memory efficient)")
            
            # Run RTX 3050 optimized training
            main_rtx3050()
            
        elif use_gpu_optimized:
            # Import GPU-optimized training
            from gpu_optimized_training import main_gpu
            print(f"GPU Training configuration:")
            print(f"  - Images directory: {config['image_directory']}")
            print(f"  - Training data: {config['output_training_data']}")
            print(f"  - Batch size: {config['batch_size']}")
            print(f"  - Epochs: {config['epochs']}")
            print(f"  - Learning rate: {config['learning_rate']}")
            print(f"  - Image size: {config['image_size']}x{config['image_size']}")
            
            # Run GPU training
            main_gpu()
        else:
            # Import standard training modules
            from grain_resnet_training import main as train_main
            
            print(f"Training configuration:")
            print(f"  - Images directory: {config['image_directory']}")
            print(f"  - Training data: {config['output_training_data']}")
            print(f"  - Batch size: {config['batch_size']}")
            print(f"  - Epochs: {config['epochs']}")
            print(f"  - Learning rate: {config['learning_rate']}")
            
            # Run training
            train_main()
        
        print("✅ Model training completed!")
        
    except Exception as e:
        print(f"❌ Error during model training: {e}")
        print("Please check that all dependencies are installed:")
        if use_gpu_optimized or use_rtx3050_optimized:
            print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("pip install pandas scikit-learn matplotlib seaborn pillow tqdm")
        return False
    
    # Step 4: Summary
    print("\n🎉 PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print("Generated files:")
    print(f"  📊 {config['output_training_data']} - Training dataset")
    print(f"  📈 image_quality_report.csv - Quality analysis")
    print(f"  🤖 {config['model_save_path']} - Trained model")
    print(f"  📉 training_curves.png - Training progress")
    print(f"  🎯 confusion_matrix.png - Model performance")
    
    if use_rtx3050_optimized:
        print(f"\n🎯 RTX 3050 4GB Training completed with:")
        print(f"  💾 Memory-optimized ResNet-18 architecture")
        print(f"  ⚡ Gradient accumulation (effective batch size: {config['batch_size'] * 4})")
        print(f"  📏 Optimized 192x192 image resolution")
        print(f"  🧹 Automatic GPU memory management")
        print(f"  🔄 Conservative mixed precision training")
    elif use_gpu_optimized:
        print(f"\n🚀 GPU Training completed with:")
        print(f"  ⚡ Mixed precision training")
        print(f"  🔥 Advanced data augmentation")
        print(f"  📈 OneCycle learning rate scheduling")
        print(f"  🎯 Early stopping with patience")
    
    return True

def setup_environment():
    """
    Setup the environment and install required packages
    """
    print("🔧 Setting up environment...")
    
    # Check for GPU first
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU detected: {gpu_name} ({total_memory:.1f} GB)")
            
            if "3050" in gpu_name and total_memory < 5.0:
                print("🎯 RTX 3050 4GB optimization enabled")
        else:
            print("⚠️  No GPU detected, will use CPU")
    except ImportError:
        gpu_available = False
        print("📦 PyTorch not installed")
    
    # GPU-optimized packages
    if gpu_available:
        gpu_packages = ['tqdm']  # Additional packages for GPU training
        required_packages = [
            'torch', 'torchvision', 'torchaudio', 'pandas', 'scikit-learn', 
            'matplotlib', 'seaborn', 'pillow', 'openpyxl'
        ] + gpu_packages
    else:
        required_packages = [
            'torch', 'torchvision', 'pandas', 'scikit-learn', 
            'matplotlib', 'seaborn', 'pillow', 'openpyxl', 'tqdm'
        ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} not found. Installing...")
            if package in ['torch', 'torchvision', 'torchaudio'] and gpu_available:
                # Install GPU version of PyTorch
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio", 
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ])
                break  # Skip individual installation since we installed all torch packages
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    # Additional GPU setup recommendations
    if gpu_available:
        print("\n🚀 GPU Setup Recommendations:")
        print("- Ensure CUDA drivers are up to date")
        print("- Monitor GPU temperature during training")
        print("- Consider using nvidia-smi to monitor GPU usage")
        print("- Clear GPU cache if running multiple experiments: torch.cuda.empty_cache()")
    
    print("✅ Environment setup completed!")

def check_dataset_compatibility():
    """
    Check if the converted dataset is compatible with training
    """
    print("\n🔍 STEP 2.5: Checking dataset compatibility...")
    
    try:
        import pandas as pd
        
        # Load converted data
        df = pd.read_csv('grain_training_data.csv')
        
        # Basic checks
        required_columns = ['image_name', 'count', 'good', 'bad']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
            return False
        
        # Data quality checks
        if df.isnull().any().any():
            print("⚠️  Warning: Dataset contains null values")
        
        if (df['count'] != df['good'] + df['bad']).any():
            print("⚠️  Warning: Count doesn't match good + bad for some images")
        
        print(f"✅ Dataset compatibility check passed!")
        print(f"   - Total images: {len(df)}")
        print(f"   - Average grains per image: {df['count'].mean():.1f}")
        print(f"   - Good dominant images: {sum(df['good'] > df['bad'])}")
        print(f"   - Bad dominant images: {sum(df['bad'] > df['good'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset compatibility check failed: {e}")
        return False

def main():
    """
    Main function to run the complete pipeline
    """
    
    print("🌾 Welcome to the GPU-Optimized Grain Quality Assessment Pipeline!")
    print("This will convert your annotations and train a ResNet-50 model with GPU acceleration.")
    print()
    
    # Check GPU availability and type
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name()
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            
            print(f"🚀 GPU Detected: {gpu_name}")
            print(f"📊 GPU Memory: {total_memory:.1f} GB")
            
            # Detect RTX 3050 4GB specifically
            if "3050" in gpu_name and total_memory < 5.0:
                print("🎯 RTX 3050 4GB detected - using memory-optimized settings!")
                use_rtx3050_optimized = True
                use_gpu_optimized = False
            elif total_memory >= 8.0:
                print("🚀 High-end GPU detected - using full GPU optimization!")
                use_gpu_optimized = True
                use_rtx3050_optimized = False
            else:
                print("⚡ Mid-range GPU detected - using standard GPU optimization!")
                use_gpu_optimized = True
                use_rtx3050_optimized = False
        else:
            print("⚠️  No GPU detected. Will use CPU-optimized version.")
            use_gpu_optimized = False
            use_rtx3050_optimized = False
    except ImportError:
        print("⚠️  PyTorch not installed. Will setup environment first.")
        use_gpu_optimized = False
        use_rtx3050_optimized = False
    
    # Check if user wants to setup environment
    setup_env = input("\nDo you want to setup/check the environment? (y/n): ").lower() == 'y'
    if setup_env:
        setup_environment()
    
    print("\nStarting pipeline...")
    
    # Run the complete pipeline
    success = run_complete_pipeline(use_gpu_optimized, use_rtx3050_optimized)
    
    if success:
        # Additional compatibility check
        check_dataset_compatibility()
        
        print("\n🎯 NEXT STEPS:")
        print("1. Review the generated quality report to understand your dataset")
        print("2. Check training curves to ensure the model is learning properly")
        print("3. Evaluate the confusion matrix to assess classification performance")
        print("4. Fine-tune hyperparameters if needed")
        
        if use_rtx3050_optimized:
            print("\n🎯 RTX 3050 4GB SPECIFIC TIPS:")
            print("- Monitor GPU memory usage with: nvidia-smi")
            print("- Current settings use ~3.5GB peak (safe for 4GB)")
            print("- If memory issues occur, reduce batch_size to 4")
            print("- ResNet-18 model is more efficient for deployment")
            print("- Gradient accumulation simulates larger batch training")
        elif use_gpu_optimized:
            print("\n🚀 GPU-SPECIFIC TIPS:")
            print("- Monitor GPU memory usage with: nvidia-smi")
            print("- Increase batch size if you have more GPU memory")
            print("- Use mixed precision training for even faster speeds")
            print("- Consider using multiple GPUs if available")
        
        print("\n🔧 TROUBLESHOOTING:")
        print("- If classes are misclassified, update the class mappings in dataset_converter.py")
        if use_rtx3050_optimized:
            print("- If GPU runs out of memory, reduce batch_size to 4 in rtx3050_optimized.py")
            print("- If training is unstable, disable mixed precision by setting scaler=None")
            print("- Monitor GPU temperature: RTX 3050 should stay under 80°C")
        elif use_gpu_optimized:
            print("- If GPU runs out of memory, reduce batch_size or image resolution")
            print("- If training is unstable, disable mixed precision training")
        else:
            print("- If training is slow, reduce batch_size or image resolution")
        print("- If overfitting occurs, add more data augmentation or regularization")
    else:
        print("\n❌ Pipeline failed. Please check the error messages above.")

if __name__ == "__main__":
    main()