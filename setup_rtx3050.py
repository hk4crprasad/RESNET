"""
Automated setup script for RTX 3050 4GB Grain Quality Training
This script will set up everything you need to start training
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"🌾 {title}")
    print(f"{'='*60}")

def print_step(step_num, description):
    """Print a formatted step"""
    print(f"\n📋 STEP {step_num}: {description}")
    print("-" * 40)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required!")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def check_gpu():
    """Check GPU availability and specifications"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gpu_info = result.stdout.strip().split(',')
            gpu_name = gpu_info[0].strip()
            memory_mb = int(gpu_info[1].strip())
            memory_gb = memory_mb / 1024
            
            print(f"✅ GPU Found: {gpu_name}")
            print(f"📊 Memory: {memory_gb:.1f} GB")
            
            if "3050" in gpu_name and memory_gb < 5.0:
                print("🎯 RTX 3050 4GB detected - perfect for this project!")
                return True, "rtx3050"
            elif memory_gb >= 8.0:
                print("🚀 High-end GPU detected - will use advanced optimizations!")
                return True, "high_end"
            else:
                print("⚡ Mid-range GPU detected - will use standard GPU optimizations!")
                return True, "standard"
        else:
            print("⚠️  No NVIDIA GPU detected")
            return False, "cpu"
    except FileNotFoundError:
        print("⚠️  nvidia-smi not found - No GPU or drivers not installed")
        return False, "cpu"

def install_pytorch_cuda():
    """Install PyTorch with CUDA support"""
    print("🔧 Installing PyTorch with CUDA 12.1 support...")
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu121",
            "--upgrade"
        ])
        print("✅ PyTorch with CUDA installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install PyTorch with CUDA: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing required packages...")
    
    packages = [
        "pandas", "scikit-learn", "matplotlib", "seaborn", 
        "pillow", "openpyxl", "tqdm", "numpy"
    ]
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install"
        ] + packages + ["--upgrade"])
        print("✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_project_structure():
    """Create the project directory structure"""
    print("📁 Creating project structure...")
    
    # Create directories
    directories = ["images", "models", "outputs", "logs"]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}/")
    
    # Create a sample images check
    images_dir = Path("images")
    if not any(images_dir.iterdir()):
        print("⚠️  images/ directory is empty")
        print("   📝 You need to copy your 759 grain images here")
        
        # Create a placeholder file
        placeholder = images_dir / "README_PLACE_IMAGES_HERE.txt"
        placeholder.write_text(
            "Place your 759 grain images in this directory.\n"
            "Image names must match exactly with your annotation file.\n"
            "\nExample:\n"
            "- IMG_6517-JPG_jpg.rf.28a61011eae8e0272ca46e4ad3766bd0.jpg\n"
            "- 20220601_150138_jpg.rf.afda3954dc1638520d75fe7ed1981923.jpg\n"
        )
        print(f"   📄 Created: {placeholder}")
    else:
        image_count = len(list(images_dir.glob("*.jpg"))) + len(list(images_dir.glob("*.png")))
        print(f"✅ Found {image_count} images in images/ directory")
    
    return True

def check_required_files():
    """Check if required files are present"""
    print("🔍 Checking required files...")
    
    required_files = [
        "_annotations.csv.xlsx",
        "dataset_converter.py",
        "rtx3050_optimized.py",
        "complete_pipeline.py"
    ]
    
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("   Please ensure all Python files are in the current directory")
        return False
    
    return True

def test_installation():
    """Test if the installation works"""
    print("🧪 Testing installation...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Test CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA {torch.version.cuda}")
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
        else:
            print("⚠️  CUDA not available")
        
        # Test other packages
        import pandas as pd
        import sklearn
        import PIL
        import tqdm
        print("✅ All packages imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_quick_start_script():
    """Create a quick start script"""
    print("📝 Creating quick start script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Quick Start Script for RTX 3050 4GB Grain Quality Training
Run this after setup to start training immediately
"""

import os
import sys

def main():
    print("🌾 RTX 3050 4GB Grain Quality Training - Quick Start")
    print("=" * 60)
    
    # Check if images directory has files
    image_files = [f for f in os.listdir("images") if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print("❌ No images found in images/ directory!")
        print("   📝 Please copy your 759 grain images to the images/ folder")
        return
    
    print(f"✅ Found {len(image_files)} images")
    
    # Check if annotation file exists
    if not os.path.exists("_annotations.csv.xlsx"):
        print("❌ Annotation file '_annotations.csv.xlsx' not found!")
        return
    
    print("✅ Annotation file found")
    
    # Run the complete pipeline
    print("\\n🚀 Starting automated training pipeline...")
    print("   This will:")
    print("   1. Convert your annotations to training format")
    print("   2. Clean the dataset")
    print("   3. Train an RTX 3050 optimized model")
    print("   4. Generate performance reports")
    
    input("\\nPress Enter to continue or Ctrl+C to cancel...")
    
    try:
        os.system("python complete_pipeline.py")
    except KeyboardInterrupt:
        print("\\n⏹️  Training cancelled by user")

if __name__ == "__main__":
    main()
'''
    
    with open("quick_start.py", "w") as f:
        f.write(script_content)
    
    print("✅ Created: quick_start.py")
    return True

def main():
    """Main setup function"""
    print_header("RTX 3050 4GB Grain Quality Training Setup")
    print("This script will set up everything you need for grain quality assessment")
    
    # Step 1: Check Python version
    print_step(1, "Checking Python version")
    if not check_python_version():
        return False
    
    # Step 2: Check GPU
    print_step(2, "Detecting GPU")
    gpu_available, gpu_type = check_gpu()
    
    # Step 3: Check required files
    print_step(3, "Checking required files")
    if not check_required_files():
        return False
    
    # Step 4: Create project structure
    print_step(4, "Creating project structure")
    if not create_project_structure():
        return False
    
    # Step 5: Install PyTorch
    print_step(5, "Installing PyTorch with CUDA")
    if gpu_available:
        if not install_pytorch_cuda():
            return False
    else:
        print("⚠️  No GPU detected, will install CPU-only PyTorch")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
            print("✅ PyTorch (CPU) installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install PyTorch")
            return False
    
    # Step 6: Install dependencies
    print_step(6, "Installing dependencies")
    if not install_dependencies():
        return False
    
    # Step 7: Test installation
    print_step(7, "Testing installation")
    if not test_installation():
        return False
    
    # Step 8: Create quick start script
    print_step(8, "Creating helper scripts")
    if not create_quick_start_script():
        return False
    
    # Final summary
    print_header("SETUP COMPLETED SUCCESSFULLY!")
    
    print("✅ Your RTX 3050 4GB grain quality training environment is ready!")
    print("\n🎯 Next Steps:")
    print("1. Copy your 759 grain images to the 'images/' folder")
    print("2. Ensure '_annotations.csv.xlsx' is in the current directory")
    print("3. Run: python quick_start.py")
    print("\n📊 Expected Performance:")
    print(f"   - Training Speed: {'80-100 imgs/sec' if gpu_available else '10-20 imgs/sec'}")
    print(f"   - Training Time: {'~60-75 minutes' if gpu_available else '~4-6 hours'}")
    print(f"   - Memory Usage: {'~3.5GB GPU' if gpu_available else '~2GB RAM'}")
    
    if gpu_type == "rtx3050":
        print("\n🎯 RTX 3050 4GB Specific Tips:")
        print("   - Monitor GPU memory: nvidia-smi")
        print("   - Keep temperature under 80°C")
        print("   - Model will be optimized for 4GB VRAM")
    
    print("\n🚀 Ready to train your grain quality assessment model!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above and try again.")
        sys.exit(1)
    else:
        print("\n🎉 Setup completed successfully!")