"""Style transfer module for converting text to user's writing style."""

import subprocess
import json
from typing import Optional


class StyleTransfer:
    """Transfer writing style using the fine-tuned model."""
    
    def __init__(self, model_name: str = "mystyle-llama3.1:8b", 
                 base_model: str = "llama3.1:8b"):
        """
        Initialize style transfer.
        
        Args:
            model_name: Name of the fine-tuned model
            base_model: Base model to use if fine-tuned model is not available
        """
        self.model_name = model_name
        self.base_model = base_model
        self.current_model = None
        self._check_model_availability()
    
    def _check_model_availability(self):
        """Check which model is available."""
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                models = result.stdout.strip().split('\n')[1:] if result.stdout.strip() else []
                model_names = [line.split()[0] for line in models if line.strip()]
                
                if self.model_name in model_names:
                    self.current_model = self.model_name
                elif self.base_model in model_names:
                    self.current_model = self.base_model
                    print(f"Using base model '{self.base_model}' (fine-tuned model not found)")
                else:
                    self.current_model = self.base_model
                    print(f"Warning: Model not found. Will attempt to use '{self.base_model}'")
        except:
            self.current_model = self.base_model
    
    def transfer_style(self, text: str, temperature: float = 0.7) -> str:
        """
        Transfer text to user's writing style.
        
        Args:
            text: Text to convert
            temperature: Sampling temperature
            
        Returns:
            Text in user's writing style
        """
        if self.current_model is None:
            raise RuntimeError("No model available. Make sure Ollama is running and model is deployed.")
        
        prompt = f"Rewrite the following text in your writing style:\n\n{text}"
        
        try:
            result = subprocess.run(
                ['ollama', 'run', self.current_model, prompt],
                capture_output=True,
                text=True,
                timeout=60,
                input=prompt
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Request timed out"
        except FileNotFoundError:
            return "Error: Ollama not found. Make sure Ollama is installed and running."
        except Exception as e:
            return f"Error: {str(e)}"

