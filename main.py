"""
Main script for fine-tuning Llama 3.1 8B model with custom writing style.

This script handles:
1. Environment setup and dependency verification
2. Data processing and dataset creation
3. Model fine-tuning with LoRA/QLoRA
4. Model deployment to Ollama
5. Style transfer and text generation
"""

import sys
import subprocess
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import custom modules
try:
    from src.setup import verify_setup
    from src.data_processor import DataProcessor
    from src.finetune import FineTuner
    from src.deploy import ModelDeployer
    from src.style_transfer import StyleTransfer
    from src.generate import StyleGenerator, generate_text
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("\nThe 'src' directory or its modules are missing.")
    print("Please ensure the 'src' directory exists with the following modules:")
    print("  - src/setup.py")
    print("  - src/data_processor.py")
    print("  - src/finetune.py")
    print("  - src/deploy.py")
    print("  - src/style_transfer.py")
    print("  - src/generate.py")
    sys.exit(1)


def install_dependencies():
    """Install and fix dependencies with correct versions."""
    print("="*60)
    print("Installing Dependencies (Fixed Versions)")
    print("="*60)
    print(f"Python: {sys.executable}\n")

    # Step 1: Fix critical packages first (order matters!)
    print("Step 1: Fixing critical packages...")

    # Fix huggingface_hub (MUST be <1.0 for transformers 4.42.4)
    print("  - Fixing huggingface_hub (<1.0)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "huggingface_hub>=0.19.3,<1.0.0", "--force-reinstall", "--no-cache-dir"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("    [OK] huggingface_hub fixed")
    except:
        print("    [WARNING] huggingface_hub install had issues")

    # Fix torch (MUST be <2.4.0 to avoid int1 issue)
    print("  - Fixing torch (<2.4.0)...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch>=2.0.0,<2.4.0", "torchvision>=0.15.0,<0.19.0",
            "--force-reinstall", "--no-cache-dir"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("    [OK] torch fixed")
    except:
        print("    [WARNING] torch install had issues")

    print()

    # Step 2: Install from requirements.txt
    print("Step 2: Installing from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "requirements.txt", "--no-cache-dir"
        ])
        print("[OK] All packages installed\n")
    except Exception as e:
        print(f"[WARNING] Some packages may have issues: {e}\n")

    # Step 3: Verify critical versions
    print("Step 3: Verifying versions...")
    try:
        import huggingface_hub
        ver = huggingface_hub.__version__
        print(f"  huggingface_hub: {ver}", end="")
        if ver.startswith('1.'):
            print(" [ERROR - too new!]")
        else:
            print(" [OK]")
    except:
        print("  huggingface_hub: [ERROR - not installed]")

    try:
        import torch
        ver = torch.__version__
        print(f"  torch: {ver}", end="")
        if ver.startswith('2.5') or ver.startswith('2.6') or ver.startswith('2.9'):
            print(" [WARNING - too new, may cause issues]")
        else:
            print(" [OK]")
    except:
        print("  torch: [ERROR - not installed]")

    try:
        import transformers
        print(f"  transformers: {transformers.__version__} [OK]")
    except:
        print("  transformers: [ERROR - not installed]")

    try:
        import peft
        print(f"  peft: {peft.__version__} [OK]")
    except:
        print("  peft: [ERROR - not installed]")

    print("\n" + "="*60)
    print("IMPORTANT: RESTART SCRIPT NOW!")
    print("Then run: verify_setup()")
    print("="*60)


def remove_trl():
    """Remove TRL if installed (it's incompatible and not needed without GPU)."""
    print("="*60)
    print("Removing TRL (incompatible with transformers 4.42.4)")
    print("="*60)
    print(f"Python: {sys.executable}\n")

    # Force uninstall TRL
    print("Uninstalling TRL...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", 
            "-y", "trl"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("[OK] TRL uninstalled\n")
    except:
        print("[INFO] TRL uninstall attempted\n")

    # Verify it's gone
    print("Verifying TRL removal...")
    try:
        import trl
        print(f"[WARNING] TRL still found: {trl.__version__}")
        print("Trying force uninstall...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", 
            "-y", "trl", "--break-system-packages"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (ImportError, ModuleNotFoundError):
        print("[OK] TRL is not installed (good!)\n")

    print("="*60)
    print("RESTART SCRIPT and try importing again!")
    print("="*60)


def setup_environment():
    """Verify environment setup."""
    verify_setup()


def process_data():
    """Load writing, preprocess it, and create training datasets."""
    # Initialize processor
    processor = DataProcessor()

    # Process data
    train_dataset, val_dataset, stats = processor.process(save_datasets=True)

    # Display statistics
    print("\nData Statistics:")
    print(f"Total words: {stats['total_words']:,}")
    print(f"Total characters: {stats['total_characters']:,}")
    print(f"Train examples: {stats['train_examples']}")
    print(f"Val examples: {stats['val_examples']}")
    
    return train_dataset, val_dataset, stats


def fine_tune_model(train_dataset, val_dataset):
    """Fine-tune llama3.1:8b using Unsloth with LoRA/QLoRA for efficient training."""
    # Initialize fine-tuner
    fine_tuner = FineTuner()

    try:
        # Load model and setup LoRA
        fine_tuner.load_model()
        fine_tuner.setup_lora()

        # Train the model
        fine_tuner.train(train_dataset, val_dataset)

        # Save the model
        fine_tuner.save_model()
        fine_tuner.save_for_ollama()
        
        return fine_tuner
    except RuntimeError as e:
        if "GPU" in str(e) or "Unsloth" in str(e):
            print(f"\n[SKIPPED] Fine-tuning requires GPU: {e}")
            print("You can:")
            print("  1. Use a GPU-enabled machine or cloud service")
            print("  2. Use Ollama's built-in fine-tuning features")
            print("  3. Skip fine-tuning and use the base model for style transfer")
            return None
        else:
            raise


def deploy_model():
    """Convert the fine-tuned model to GGUF format and deploy to Ollama."""
    # Initialize deployer
    deployer = ModelDeployer()

    # Create Modelfile and deploy
    deployer.create_modelfile()
    # deployer.deploy()  # Uncomment after GGUF conversion
    
    return deployer


def test_style_transfer():
    """Test style transfer functionality."""
    # Initialize style transfer
    transfer = StyleTransfer()

    # Test with sample text
    sample_text = "Artificial intelligence has revolutionized technology. Machine learning processes vast data with remarkable accuracy."

    converted = transfer.transfer_style(sample_text)
    print("Original:", sample_text)
    print("\nConverted:", converted)
    
    return transfer


def initialize_generator():
    """Initialize the style generator (uses Ollama by default)."""
    # Initialize generator (uses Ollama by default)
    # This will connect to your Ollama service
    generator = StyleGenerator(use_ollama=True)

    print("Generator ready! You can now use:")
    print("  - generator.rewrite('your text here')")
    print("  - generator.write_about('your topic here')")
    
    return generator


def example_rewrite_text(generator):
    """Example: Rewrite text in your style."""
    text_to_rewrite = """Artificial intelligence has revolutionized the way we interact with technology. 
Machine learning algorithms can now process vast amounts of data and make predictions with remarkable accuracy. 
This has applications in various fields including healthcare, finance, and transportation."""

    rewritten = generator.rewrite(text_to_rewrite)
    print("Original text:")
    print(text_to_rewrite)
    print("\n" + "="*60)
    print("Rewritten in your style:")
    print("="*60)
    print(rewritten)
    
    return rewritten


def example_write_about_topic(generator):
    """Example: Write about a topic in your style."""
    topic = "The importance of education in personal development"

    generated = generator.write_about(topic, max_length=300, temperature=0.7)
    print(f"Topic: {topic}")
    print("\n" + "="*60)
    print("Generated text in your style:")
    print("="*60)
    print(generated)
    
    return generated


def example_quick_generation():
    """Quick function: One-line generation."""
    # Quick rewrite
    result = generate_text(prompt="Technology is changing our lives rapidly.", use_ollama=True)
    print(result)

    # Quick topic writing
    result = generate_text(topic="The future of artificial intelligence", use_ollama=True)
    print(result)
    
    return result


def example_custom_generation(generator):
    """Customize generation parameters."""
    # Customize generation parameters
    result = generator.generate(
        topic="Climate change and its impact",
        max_length=400,        # Maximum tokens to generate
        temperature=0.7,       # Lower = more consistent, Higher = more creative
    )

    print(result)
    return result


def main():
    """Main execution function."""
    print("="*60)
    print("Llama 3.1 8B Fine-tuning Pipeline")
    print("="*60)
    print("\nThis script will guide you through:")
    print("1. Environment setup")
    print("2. Data processing")
    print("3. Model fine-tuning")
    print("4. Model deployment")
    print("5. Style transfer and generation")
    print("\n" + "="*60)
    
    # Step 1: Environment Setup
    print("\n## Step 1: Environment Setup")
    print("Verifying environment...")
    setup_environment()
    
    # Step 2: Data Processing
    print("\n## Step 2: Data Processing")
    print("Processing data and creating datasets...")
    train_dataset, val_dataset, stats = process_data()
    
    # Step 3: Model Fine-tuning
    print("\n## Step 3: Model Fine-tuning")
    print("Fine-tuning model with LoRA/QLoRA...")
    fine_tuner = fine_tune_model(train_dataset, val_dataset)
    
    # Step 4: Deploy to Ollama
    print("\n## Step 4: Deploy to Ollama")
    if fine_tuner is not None:
        print("Creating Modelfile and preparing for deployment...")
        deployer = deploy_model()
    else:
        print("[SKIPPED] Deployment skipped (no fine-tuned model available)")
        deployer = None
    
    # Step 5: Style Transfer
    print("\n## Step 5: Style Transfer")
    print("Testing style transfer...")
    try:
        transfer = test_style_transfer()
    except Exception as e:
        print(f"[WARNING] Style transfer test failed: {e}")
        transfer = None
    
    # Step 6: Generate Text
    print("\n## Step 6: Generate Text in Your Style")
    print("Initializing generator...")
    try:
        generator = initialize_generator()
        
        # Example usage
        print("\n### Example: Rewrite Text")
        example_rewrite_text(generator)
        
        print("\n### Example: Write About Topic")
        example_write_about_topic(generator)
    except Exception as e:
        print(f"[WARNING] Text generation failed: {e}")
        print("Make sure Ollama is running: ollama serve")
        generator = None
    
    print("\n" + "="*60)
    if fine_tuner is not None:
        print("Pipeline completed successfully!")
    else:
        print("Pipeline completed (fine-tuning skipped - GPU required)")
    print("="*60)
    
    if generator is not None:
        print("\nYou can now use the generator to:")
        print("  - Rewrite text: generator.rewrite('your text')")
        print("  - Write about topics: generator.write_about('your topic')")
        print("  - Custom generation: generator.generate(topic='...', max_length=400, temperature=0.7)")
    else:
        print("\nNote: Text generation requires Ollama to be running.")
        print("Start Ollama with: ollama serve")
        print("Then use the generator functions from src.generate module")
    print("="*60)


if __name__ == "__main__":
    # Uncomment the following lines if you need to install dependencies first
    # install_dependencies()
    # remove_trl()
    
    # Run main pipeline
    main()

