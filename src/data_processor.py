"""Data processing module for preparing training datasets."""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datasets import Dataset


class DataProcessor:
    """Process text data and create training datasets."""
    
    def __init__(self, data_file: str = "data/my-writing.txt", chunk_size: int = 512):
        """
        Initialize the data processor.
        
        Args:
            data_file: Path to the input text file
            chunk_size: Size of text chunks for training
        """
        self.data_file = Path(data_file)
        self.chunk_size = chunk_size
        self.output_dir = Path("data/datasets")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_text(self) -> str:
        """Load text from file."""
        print("Loading text file...")
        with open(self.data_file, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text by cleaning and normalizing."""
        print("Preprocessing text...")
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'"]', '', text)
        return text.strip()
    
    def calculate_statistics(self, text: str) -> Dict[str, int]:
        """Calculate text statistics."""
        print("Calculating statistics...")
        words = text.split()
        characters = len(text)
        lines = text.count('\n') + 1
        
        stats = {
            'total_words': len(words),
            'total_characters': characters,
            'total_lines': lines
        }
        
        print(f"Loaded {stats['total_words']:,} words, {stats['total_characters']:,} characters, {stats['total_lines']} lines")
        return stats
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks for training."""
        print("Chunking text...")
        
        # Try to use tokenizer if available
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("unsloth/llama-3.1-8b-bnb-4bit")
            tokens = tokenizer.encode(text)
            chunks = []
            for i in range(0, len(tokens), self.chunk_size):
                chunk_tokens = tokens[i:i + self.chunk_size]
                chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                chunks.append(chunk_text)
            print(f"Created {len(chunks)} chunks using tokenizer")
            return chunks
        except Exception as e:
            print(f"Warning: Could not load tokenizer unsloth/llama-3.1-8b-bnb-4bit")
            print(f"Error: {e}")
            print("Using simple word-based chunking instead.")
        
        # Fallback to word-based chunking
        words = text.split()
        chunks = []
        current_chunk = []
        current_size = 0
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_size = word_size
            else:
                current_chunk.append(word)
                current_size += word_size
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        print(f"Created {len(chunks)} chunks")
        return chunks
    
    def format_for_training(self, chunks: List[str]) -> List[Dict[str, str]]:
        """Format chunks into training examples."""
        print("Formatting for training...")
        
        examples = []
        for chunk in chunks:
            # Create instruction-following format
            example = {
                "instruction": "Rewrite the following text in your writing style:",
                "input": chunk,
                "output": chunk  # In fine-tuning, input and output are the same
            }
            examples.append(example)
        
        print(f"Created {len(examples)} training examples")
        return examples
    
    def create_train_val_split(self, examples: List[Dict[str, str]], 
                              val_ratio: float = 0.1) -> Tuple[List[Dict], List[Dict]]:
        """Split examples into train and validation sets."""
        print("Creating train/validation split...")
        
        split_idx = int(len(examples) * (1 - val_ratio))
        train_examples = examples[:split_idx]
        val_examples = examples[split_idx:]
        
        print(f"Train examples: {len(train_examples)}, Val examples: {len(val_examples)}")
        return train_examples, val_examples
    
    def save_datasets(self, train_examples: List[Dict], val_examples: List[Dict]):
        """Save datasets to JSONL files."""
        print("Saving datasets...")
        
        train_file = self.output_dir / "train_dataset.jsonl"
        val_file = self.output_dir / "val_dataset.jsonl"
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for example in val_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Saved train dataset to {train_file}")
        print(f"Saved val dataset to {val_file}")
    
    def create_huggingface_datasets(self, train_examples: List[Dict], 
                                   val_examples: List[Dict]) -> Tuple[Dataset, Dataset]:
        """Create HuggingFace Dataset objects."""
        print("Creating HuggingFace datasets...")
        
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        return train_dataset, val_dataset
    
    def process(self, save_datasets: bool = True) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
        """
        Process data and create training datasets.
        
        Args:
            save_datasets: Whether to save datasets to JSONL files
            
        Returns:
            Tuple of (train_dataset, val_dataset, statistics)
        """
        # Load and preprocess
        text = self.load_text()
        text = self.preprocess_text(text)
        stats = self.calculate_statistics(text)
        
        # Chunk and format
        chunks = self.chunk_text(text)
        examples = self.format_for_training(chunks)
        
        # Split
        train_examples, val_examples = self.create_train_val_split(examples)
        
        # Save if requested
        if save_datasets:
            self.save_datasets(train_examples, val_examples)
        
        # Create HuggingFace datasets
        train_dataset, val_dataset = self.create_huggingface_datasets(train_examples, val_examples)
        
        # Update stats
        stats['train_examples'] = len(train_examples)
        stats['val_examples'] = len(val_examples)
        
        return train_dataset, val_dataset, stats

