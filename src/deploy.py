"""Model deployment module for Ollama."""

import subprocess
from pathlib import Path
from typing import Optional


class ModelDeployer:
    """Deploy fine-tuned model to Ollama."""
    
    def __init__(self, model_name: str = "mystyle-llama3.1:8b"):
        """
        Initialize the model deployer.
        
        Args:
            model_name: Name for the deployed model
        """
        self.model_name = model_name
        self.merged_model_dir = Path("checkpoints/merged_model")
        self.gguf_model_dir = Path("checkpoints/gguf_model")
        self.gguf_model_dir.mkdir(parents=True, exist_ok=True)
    
    def create_modelfile(self):
        """Create Modelfile for Ollama deployment."""
        print("Creating Modelfile...")
        
        modelfile_path = self.gguf_model_dir / "Modelfile"
        
        modelfile_content = f"""FROM {self.merged_model_dir.absolute()}

# Set system prompt
SYSTEM \"You are a writing style assistant. Rewrite text in the user's personal writing style.\"

# Set parameters
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER num_ctx 4096
"""
        
        with open(modelfile_path, 'w', encoding='utf-8') as f:
            f.write(modelfile_content)
        
        print(f"[OK] Modelfile created at {modelfile_path}")
        print("\nNote: Before deploying, you need to:")
        print("1. Convert the merged model to GGUF format using llama.cpp")
        print("2. Place the GGUF file in the model directory")
        print("3. Update the Modelfile to point to the GGUF file")
        print("4. Then run: ollama create {self.model_name} -f {modelfile_path}")
    
    def deploy(self):
        """
        Deploy model to Ollama.
        
        Note: This requires the model to be converted to GGUF format first.
        """
        print("Deploying model to Ollama...")
        
        modelfile_path = self.gguf_model_dir / "Modelfile"
        
        if not modelfile_path.exists():
            print("[ERROR] Modelfile not found. Run create_modelfile() first.")
            return
        
        try:
            # Create model in Ollama
            result = subprocess.run(
                ['ollama', 'create', self.model_name, '-f', str(modelfile_path)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"[OK] Model '{self.model_name}' deployed to Ollama")
                print(f"You can now use it with: ollama run {self.model_name}")
            else:
                print(f"[ERROR] Deployment failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("[ERROR] Deployment timed out")
        except FileNotFoundError:
            print("[ERROR] Ollama not found. Make sure Ollama is installed and in PATH.")
        except Exception as e:
            print(f"[ERROR] Deployment failed: {e}")

