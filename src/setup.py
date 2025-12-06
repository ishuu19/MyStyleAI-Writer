"""Environment setup and verification module."""

import sys
import subprocess
import os
from pathlib import Path


def verify_setup():
    """Verify that all dependencies are installed and Ollama is set up correctly."""
    print("="*60)
    print("Environment Setup Verification")
    print("="*60)
    
    all_ok = True
    
    # 1. Check Python version
    print("\n1. Checking Python version...")
    python_version = sys.version_info
    if python_version.major == 3 and python_version.minor >= 8:
        print(f"[OK] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"[WARNING] Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        print("  Recommended: Python 3.8+")
    
    # 2. Check dependencies
    print("\n2. Checking dependencies...")
    required_packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'peft': 'PEFT',
        'bitsandbytes': 'BitsAndBytes',
        'accelerate': 'Accelerate',
        'ollama': 'Ollama',
        'unsloth': 'Unsloth'
    }
    
    missing_packages = []
    for package, name in required_packages.items():
        try:
            # Special handling for unsloth which may fail on import without GPU
            if package == 'unsloth':
                # Check if package is installed without importing
                import importlib.util
                spec = importlib.util.find_spec('unsloth')
                if spec is None:
                    print(f"[X] {name} NOT installed")
                    missing_packages.append(package)
                else:
                    # Try to import but catch GPU-related errors
                    try:
                        import unsloth
                        print(f"[OK] {name} installed")
                    except (NotImplementedError, RuntimeError) as e:
                        if "GPU" in str(e) or "accelerator" in str(e).lower():
                            print(f"[X] {name} installed but requires GPU (not available)")
                            missing_packages.append(package)
                        else:
                            raise
            else:
                __import__(package)
                print(f"[OK] {name} installed")
        except (ImportError, NotImplementedError, RuntimeError) as e:
            if package == 'unsloth' and ("GPU" in str(e) or "accelerator" in str(e).lower()):
                print(f"[X] {name} installed but requires GPU (not available)")
                missing_packages.append(package)
            else:
                print(f"[X] {name} NOT installed")
                missing_packages.append(package)
                if package != 'unsloth':
                    all_ok = False
    
    if missing_packages:
        print(f"\n  Missing packages: {', '.join(missing_packages)}")
        print("  Install with: pip install -r requirements.txt")
    
    # 3. Check GPU
    print("\n3. Checking GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[OK] GPU available: {torch.cuda.get_device_name(0)}")
        else:
            print("[WARNING] No GPU available. Training will be slower on CPU.")
    except:
        print("[WARNING] Could not check GPU status")
    
    # 4. Check Ollama installation
    print("\n4. Checking Ollama installation...")
    try:
        result = subprocess.run(['ollama', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version = result.stdout.strip()
            print(f"[OK] Ollama installed: {version}")
        else:
            print("[X] Ollama not found")
            all_ok = False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[X] Ollama not found or not in PATH")
        all_ok = False
    
    # 5. Check Ollama service
    print("\n5. Checking Ollama service...")
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("[OK] Ollama service is running")
        else:
            print("[WARNING] Ollama service may not be running")
            print("  Start with: ollama serve")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[WARNING] Could not connect to Ollama service")
        print("  Start with: ollama serve")
    
    # 6. Check model availability
    print("\n6. Checking model availability...")
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            models = result.stdout.strip().split('\n')[1:] if result.stdout.strip() else []
            model_names = [line.split()[0] for line in models if line.strip()]
            if 'llama3.1:8b' in model_names or any('llama3.1' in m for m in model_names):
                print("[OK] Model 'llama3.1:8b' found")
            else:
                print("[X] Model 'llama3.1:8b' not found")
                if model_names:
                    print(f"  Available models: {', '.join(model_names)}")
                else:
                    print("  Available models: None")
                print("  To pull the model, run: ollama pull llama3.1:8b")
        else:
            print("[X] Could not check models")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[X] Could not check models")
    
    # 7. Check data file
    print("\n7. Checking data file...")
    data_file = Path("data/my-writing.txt")
    if data_file.exists():
        size_mb = data_file.stat().st_size / (1024 * 1024)
        print(f"[OK] Data file found: {data_file} ({size_mb:.2f} MB)")
    else:
        print(f"[X] Data file not found: {data_file}")
        all_ok = False
    
    # 8. Create directories
    print("\n8. Creating directories...")
    directories = [
        "checkpoints",
        "outputs",
        "data/datasets",
        "checkpoints/lora_adapter",
        "checkpoints/merged_model",
        "checkpoints/gguf_model"
    ]
    
    for dir_path in directories:
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Directory created/verified: {dir_path}")
    
    print("\n" + "="*60)
    if all_ok:
        print("[OK] All checks passed!")
    else:
        print("[WARNING] Some checks failed. Please fix the issues above.")
    print("="*60)
    
    return all_ok

