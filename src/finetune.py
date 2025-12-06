"""Model fine-tuning module using Unsloth and LoRA."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

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
              save_steps: int = 500,
              max_length: int = 256,
              gradient_accumulation_steps: Optional[int] = None):
        """
        Train the model.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            num_epochs: Number of training epochs
            batch_size: Training batch size (use 1 for low memory)
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            save_steps: Model saving frequency
            max_length: Maximum sequence length (default 256 for memory efficiency)
            gradient_accumulation_steps: Gradient accumulation steps (auto-calculated if None)
        """
        if not UNSLOTH_AVAILABLE:
            raise RuntimeError("Unsloth is required for training")
        
        if self.model is None:
            raise RuntimeError("Model must be loaded first. Call load_model()")
        
        print("Starting training...")
        print(f"Epochs: {num_epochs}, Batch size: {batch_size}, Learning rate: {learning_rate}")
        print(f"Max sequence length: {max_length} (memory optimized)")
        
        from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
        
        # Auto-calculate gradient accumulation to maintain effective batch size of 8
        if gradient_accumulation_steps is None:
            gradient_accumulation_steps = max(1, 8 // batch_size)
        print(f"Gradient accumulation steps: {gradient_accumulation_steps} (effective batch size: {batch_size * gradient_accumulation_steps})")
        
        # Prepare tokenizer
        self.tokenizer.padding_side = "right"
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Format dataset - keep as text, tokenize on-the-fly to save memory
        print("Formatting dataset for training...")
        def format_prompts(examples):
            """Format prompts for instruction-following."""
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            
            texts = []
            for instruction, input_text, output in zip(instructions, inputs, outputs):
                # Create prompt in instruction format
                if input_text:
                    text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                else:
                    text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                texts.append(text)
            
            return {"text": texts}
        
        # Apply formatting (lightweight, just string formatting)
        train_dataset = train_dataset.map(format_prompts, batched=True, batch_size=100, remove_columns=train_dataset.column_names)
        val_dataset = val_dataset.map(format_prompts, batched=True, batch_size=100, remove_columns=val_dataset.column_names)
        
        # Clear memory after formatting
        import gc
        gc.collect()
        
        # Tokenize on-the-fly during training to save memory
        # Use smaller batch size for tokenization
        print(f"Tokenizing dataset (memory-efficient, max_length={max_length})...")
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,  # Reduced to save memory
                padding=False,
            )
        
        # Tokenize with smaller batches to avoid OOM
        train_dataset = train_dataset.map(
            tokenize_function, 
            batched=True, 
            batch_size=2,  # Very small batch size for tokenization to save memory
            remove_columns=["text"],
            desc="Tokenizing train dataset"
        )
        
        # Clear memory after train tokenization
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        val_dataset = val_dataset.map(
            tokenize_function, 
            batched=True, 
            batch_size=2,
            remove_columns=["text"],
            desc="Tokenizing val dataset"
        )
        
        # Clear memory after val tokenization
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Add labels (same as input_ids for causal LM)
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        print("Adding labels...")
        train_dataset = train_dataset.map(add_labels, batched=True, batch_size=50)
        val_dataset = val_dataset.map(add_labels, batched=True, batch_size=50)
        
        # Clear memory before training
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
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
            remove_unused_columns=False,
            dataloader_pin_memory=False,  # Save memory
            dataloader_num_workers=0,  # Disable multiprocessing to save memory
        )
        
        # Data collator for dynamic padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            args=training_args,
            data_collator=data_collator,
            processing_class=self.tokenizer,  # Use processing_class instead of tokenizer
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

