"""Model fine-tuning module using Unsloth and LoRA."""

import os
from pathlib import Path
from typing import Optional

# Check if Unsloth is available
UNSLOTH_AVAILABLE = False
FastLanguageModel = None
try:
    from unsloth import FastLanguageModel
    import torch
    UNSLOTH_AVAILABLE = True
except (ImportError, NotImplementedError, Exception):
    UNSLOTH_AVAILABLE = False


class FineTuner:
    """Fine-tune Llama 3.1 8B model using Unsloth with LoRA/QLoRA."""
    
    def __init__(self, model_name: str = "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit"):
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: Name of the model to fine-tune
                Recommended models (NO APPROVAL NEEDED):
                - "unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit" (Best - default, no approval!)
                - "unsloth/Phi-3-mini-4k-instruct" (Fastest training, no approval)
                - "unsloth/TinyLlama-1.1B-Chat-v1.0" (Testing only, no approval)
                
                Models requiring approval:
                - "unsloth/llama-3.1-8b-bnb-4bit" (Requires Hugging Face approval)
                - "unsloth/llama-3.1-70b-bnb-4bit" (Requires approval + Colab Pro)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.checkpoint_dir = Path("checkpoints/lora_adapter")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def load_model(self):
        """Load model and tokenizer using Unsloth."""
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError(
                "Unsloth is not available. It requires GPU and Linux/WSL.\n"
                "For CPU-only or Windows, you'll need to use a different fine-tuning approach.\n"
                "Consider using Ollama's built-in fine-tuning or cloud GPU services."
            )
        
        print("Loading model with Unsloth...")
        print(f"Model: {self.model_name}")
        
        # Load model with 4-bit quantization
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        
        print("[OK] Model and tokenizer loaded")
    
    def setup_lora(self, r: int = 16, target_modules: Optional[list] = None):
        """
        Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning.
        
        Args:
            r: LoRA rank
            target_modules: Modules to apply LoRA to
        """
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is required for LoRA setup")
        
        if self.model is None:
            raise RuntimeError("Model must be loaded first. Call load_model()")
        
        print("Setting up LoRA...")
        
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]
        
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=r,
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print("[OK] LoRA setup complete")
    
    def train(self, train_dataset, val_dataset, 
              num_epochs: int = 3,
              batch_size: int = 2,
              learning_rate: float = 2e-4,
              warmup_steps: int = 5,
              logging_steps: int = 1,
              save_steps: int = 500):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            save_steps: Model saving frequency
        """
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is required for training")
        
        if self.model is None:
            raise RuntimeError("Model must be loaded first. Call load_model()")
        
        print("Starting training...")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        
        from transformers import TrainingArguments, Trainer
        
        # Prepare tokenizer
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(self.checkpoint_dir),
            save_steps=save_steps,
            save_total_limit=3,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            tokenizer=self.tokenizer,
        )
        
        # Train
        trainer.train()
        
        print("[OK] Training complete")
    
    def save_model(self):
        """Save the fine-tuned model."""
        if self.model is None:
            raise RuntimeError("Model must be loaded and trained first")
        
        print("Saving model...")
        
        # Save LoRA adapter
        self.model.save_pretrained(str(self.checkpoint_dir))
        self.tokenizer.save_pretrained(str(self.checkpoint_dir))
        
        print(f"[OK] Model saved to {self.checkpoint_dir}")
    
    def save_for_ollama(self):
        """Save model in format suitable for Ollama deployment."""
        if self.model is None:
            raise RuntimeError("Model must be loaded and trained first")
        
        print("Preparing model for Ollama...")
        
        # Merge LoRA weights
        if UNSLOTH_AVAILABLE:
            merged_model = FastLanguageModel.merge_and_unload(self.model)
            
            merged_dir = Path("checkpoints/merged_model")
            merged_dir.mkdir(parents=True, exist_ok=True)
            
            merged_model.save_pretrained(str(merged_dir))
            self.tokenizer.save_pretrained(str(merged_dir))
            
            print(f"[OK] Merged model saved to {merged_dir}")
            print("Note: You'll need to convert to GGUF format for Ollama deployment")
        else:
            print("[WARNING] Cannot merge model without Unsloth")

