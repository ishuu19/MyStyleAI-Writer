"""Text generation module for creating content in user's writing style."""

import subprocess
import json
from typing import Optional


class StyleGenerator:
    """Generate text in user's writing style using Ollama."""
    
    def __init__(self, model_name: str = "mystyle-llama3.1:8b",
                 base_model: str = "llama3.1:8b",
                 use_ollama: bool = True):
        """
        Initialize the style generator.
        
        Args:
            model_name: Name of the fine-tuned model
            base_model: Base model to use if fine-tuned model is not available
            use_ollama: Whether to use Ollama for generation
        """
        self.model_name = model_name
        self.base_model = base_model
        self.use_ollama = use_ollama
        self.current_model = None
        
        if use_ollama:
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
                else:
                    self.current_model = self.base_model
        except:
            self.current_model = self.base_model
    
    def _call_ollama(self, prompt: str, temperature: float = 0.7, 
                    max_length: int = 500) -> str:
        """Call Ollama API to generate text."""
        if self.current_model is None:
            raise RuntimeError("No model available. Make sure Ollama is running.")
        
        try:
            # Use Ollama's generate API
            payload = {
                "model": self.current_model,
                "prompt": prompt,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_length
                }
            }
            
            result = subprocess.run(
                ['ollama', 'generate', self.current_model, prompt],
                capture_output=True,
                text=True,
                timeout=120,
                input=prompt
            )
            
            if result.returncode == 0:
                # Extract generated text (Ollama may include the prompt)
                output = result.stdout.strip()
                if prompt in output:
                    output = output.replace(prompt, "").strip()
                return output
            else:
                return f"Error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return "Error: Request timed out"
        except FileNotFoundError:
            return "Error: Ollama not found. Make sure Ollama is installed and running."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def rewrite(self, text: str, temperature: float = 0.7) -> str:
        """
        Rewrite text in user's writing style.
        
        Args:
            text: Text to rewrite
            temperature: Sampling temperature
            
        Returns:
            Rewritten text
        """
        if not self.use_ollama:
            raise NotImplementedError("Only Ollama generation is currently supported")
        
        prompt = f"Rewrite the following text in your writing style:\n\n{text}\n\nRewritten text:"
        return self._call_ollama(prompt, temperature=temperature)
    
    def write_about(self, topic: str, max_length: int = 300, 
                   temperature: float = 0.7) -> str:
        """
        Write about a topic in user's writing style.
        
        Args:
            topic: Topic to write about
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text about the topic
        """
        if not self.use_ollama:
            raise NotImplementedError("Only Ollama generation is currently supported")
        
        prompt = f"Write about the following topic in your writing style:\n\nTopic: {topic}\n\nText:"
        return self._call_ollama(prompt, temperature=temperature, max_length=max_length)
    
    def generate(self, topic: Optional[str] = None, 
                prompt: Optional[str] = None,
                max_length: int = 400,
                temperature: float = 0.7) -> str:
        """
        Generate text with custom parameters.
        
        Args:
            topic: Topic to write about (alternative to prompt)
            prompt: Custom prompt (alternative to topic)
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            
        Returns:
            Generated text
        """
        if not self.use_ollama:
            raise NotImplementedError("Only Ollama generation is currently supported")
        
        if prompt:
            full_prompt = f"{prompt}\n\nResponse:"
        elif topic:
            full_prompt = f"Write about the following topic in your writing style:\n\nTopic: {topic}\n\nText:"
        else:
            raise ValueError("Either 'topic' or 'prompt' must be provided")
        
        return self._call_ollama(full_prompt, temperature=temperature, max_length=max_length)


def generate_text(prompt: Optional[str] = None,
                 topic: Optional[str] = None,
                 use_ollama: bool = True,
                 max_length: int = 400,
                 temperature: float = 0.7) -> str:
    """
    Convenience function for quick text generation.
    
    Args:
        prompt: Text to rewrite or custom prompt
        topic: Topic to write about
        use_ollama: Whether to use Ollama
        max_length: Maximum length of generated text
        temperature: Sampling temperature
        
    Returns:
        Generated text
    """
    generator = StyleGenerator(use_ollama=use_ollama)
    
    if prompt:
        return generator.rewrite(prompt, temperature=temperature)
    elif topic:
        return generator.write_about(topic, max_length=max_length, temperature=temperature)
    else:
        raise ValueError("Either 'prompt' or 'topic' must be provided")

