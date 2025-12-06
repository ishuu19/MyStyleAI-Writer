"""API-based text generation module for use in Colab or without Ollama."""

import os
import requests
import json
from typing import Optional, Dict, Any


class APIStyleGenerator:
    """Generate text using various free API services."""
    
    def __init__(self, service: str = "mistral", api_key: Optional[str] = None):
        """
        Initialize API-based generator.
        
        Args:
            service: Service to use ('mistral', 'groq', 'hf', 'together', 'replicate')
            api_key: API key (if None, will try to get from environment or use default)
        """
        self.service = service.lower()
        self.api_key = api_key or self._get_api_key()
        self.model_name = self._get_model_name()
        
        if not self.api_key:
            raise ValueError(f"API key not found. Set {self._get_env_var_name()} environment variable or pass api_key parameter.")
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables or use default."""
        env_vars = {
            'groq': 'GROQ_API_KEY',
            'hf': 'HF_TOKEN',
            'huggingface': 'HF_TOKEN',
            'together': 'TOGETHER_API_KEY',
            'replicate': 'REPLICATE_API_TOKEN',
            'mistral': 'MISTRAL_API_KEY'
        }
        env_var = env_vars.get(self.service)
        api_key = os.getenv(env_var) if env_var else None
        
        # Default Mistral API key if not set in environment
        if not api_key and self.service == 'mistral':
            api_key = "Dz4Osbww2WlhcbsFYxUFShQThacvrU71"
        
        return api_key
    
    def _get_env_var_name(self) -> str:
        """Get environment variable name for the service."""
        env_vars = {
            'groq': 'GROQ_API_KEY',
            'hf': 'HF_TOKEN',
            'huggingface': 'HF_TOKEN',
            'together': 'TOGETHER_API_KEY',
            'replicate': 'REPLICATE_API_TOKEN',
            'mistral': 'MISTRAL_API_KEY'
        }
        return env_vars.get(self.service, 'API_KEY')
    
    def _get_model_name(self) -> str:
        """Get model name for the service."""
        models = {
            'mistral': 'mistral-small-latest',
            'groq': 'llama-3.1-8b-instant',
            'hf': 'meta-llama/Llama-3.1-8B-Instruct',
            'huggingface': 'meta-llama/Llama-3.1-8B-Instruct',
            'together': 'meta-llama/Llama-3.1-8B-Instruct',
            'replicate': 'meta/llama-3.1-8b-instruct'
        }
        return models.get(self.service, 'mistral-small-latest')
    
    def _call_mistral_api(self, prompt: str, temperature: float = 0.7,
                          max_tokens: int = 500) -> str:
        """Call Mistral AI API."""
        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a writing style assistant. Rewrite text in the user's personal writing style."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _call_groq_api(self, prompt: str, temperature: float = 0.7, 
                      max_tokens: int = 500) -> str:
        """Call Groq API."""
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a writing style assistant. Rewrite text in the user's personal writing style."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    
    def _call_hf_api(self, prompt: str, temperature: float = 0.7,
                    max_tokens: int = 500) -> str:
        """Call Hugging Face Inference API."""
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "inputs": prompt,
            "parameters": {
                "temperature": temperature,
                "max_new_tokens": max_tokens,
                "return_full_text": False
            }
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', '')
        return str(result)
    
    def _call_together_api(self, prompt: str, temperature: float = 0.7,
                          max_tokens: int = 500) -> str:
        """Call Together AI API."""
        url = "https://api.together.xyz/inference"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": ["<|end_of_text|>", "<|eot_id|>"]
        }
        
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['output']['choices'][0]['text']
    
    def _call_replicate_api(self, prompt: str, temperature: float = 0.7,
                           max_tokens: int = 500) -> str:
        """Call Replicate API."""
        import replicate
        
        output = replicate.run(
            self.model_name,
            input={
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
        
        return ''.join(output)
    
    def generate(self, prompt: str, temperature: float = 0.7,
                max_tokens: int = 500) -> str:
        """
        Generate text using the selected API service.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        try:
            if self.service == 'mistral':
                return self._call_mistral_api(prompt, temperature, max_tokens)
            elif self.service == 'groq':
                return self._call_groq_api(prompt, temperature, max_tokens)
            elif self.service in ['hf', 'huggingface']:
                return self._call_hf_api(prompt, temperature, max_tokens)
            elif self.service == 'together':
                return self._call_together_api(prompt, temperature, max_tokens)
            elif self.service == 'replicate':
                return self._call_replicate_api(prompt, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown service: {self.service}")
        except requests.exceptions.RequestException as e:
            return f"API Error: {str(e)}"
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
        prompt = f"Rewrite the following text in your writing style:\n\n{text}\n\nRewritten text:"
        return self.generate(prompt, temperature=temperature, max_tokens=500)
    
    def write_about(self, topic: str, max_length: int = 300,
                   temperature: float = 0.7) -> str:
        """
        Write about a topic in user's writing style.
        
        Args:
            topic: Topic to write about
            max_length: Maximum length (approximate)
            temperature: Sampling temperature
            
        Returns:
            Generated text about the topic
        """
        prompt = f"Write about the following topic in your writing style:\n\nTopic: {topic}\n\nText:"
        return self.generate(prompt, temperature=temperature, max_tokens=max_length)


def generate_with_api(text: Optional[str] = None,
                      topic: Optional[str] = None,
                      service: str = "groq",
                      api_key: Optional[str] = None,
                      temperature: float = 0.7,
                      max_tokens: int = 500) -> str:
    """
    Convenience function for API-based generation.
    
    Args:
        text: Text to rewrite (optional)
        topic: Topic to write about (optional)
        service: API service to use ('groq', 'hf', 'together', 'replicate')
        api_key: API key (optional, will use environment variable if not provided)
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        
    Returns:
        Generated text
    """
    generator = APIStyleGenerator(service=service, api_key=api_key)
    
    if text:
        return generator.rewrite(text, temperature=temperature)
    elif topic:
        return generator.write_about(topic, max_length=max_tokens, temperature=temperature)
    else:
        raise ValueError("Either 'text' or 'topic' must be provided")

