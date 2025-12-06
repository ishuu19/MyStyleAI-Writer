# Google Colab Setup Guide for Fine-tuning

## Free Model API Services for Fine-tuning

### Option 1: Hugging Face Inference API (Recommended - Free Tier)
**Best for:** Easy setup, good free tier, direct model access

**Steps:**
1. Go to https://huggingface.co/ and create a free account
2. Go to https://huggingface.co/settings/tokens
3. Create a new token with "Read" permissions
4. Copy the token (starts with `hf_...`)
5. Use it in the code: `HF_TOKEN = "your_token_here"`

**Free Tier Limits:**
- 1,000 requests/day
- Good for testing and small fine-tuning tasks

---

### Option 2: Groq API (Recommended - Very Fast & Free)
**Best for:** Fast inference, generous free tier

**Steps:**
1. Go to https://console.groq.com/
2. Sign up for free account
3. Go to API Keys section
4. Create a new API key
5. Copy the key
6. Use it in the code: `GROQ_API_KEY = "your_key_here"`

**Free Tier Limits:**
- 14,400 requests/day
- Very fast inference
- Supports Llama 3.1 8B

---

### Option 3: Together AI (Free Tier)
**Best for:** Fine-tuning support, good free credits

**Steps:**
1. Go to https://together.ai/
2. Sign up for free account
3. Go to API Keys section
4. Create a new API key
5. Copy the key
6. Use it in the code: `TOGETHER_API_KEY = "your_key_here"`

**Free Tier Limits:**
- $25 free credits
- Supports fine-tuning
- Good for production use

---

### Option 4: Replicate API (Free Tier)
**Best for:** Easy model deployment

**Steps:**
1. Go to https://replicate.com/
2. Sign up for free account
3. Go to Account Settings > API Tokens
4. Create a new token
5. Copy the token
6. Use it in the code: `REPLICATE_API_TOKEN = "your_token_here"`

**Free Tier Limits:**
- Limited free credits
- Pay-as-you-go after free tier

---

## Google Colab Setup

### Step 1: Open Google Colab
1. Go to https://colab.research.google.com/
2. Create a new notebook

### Step 2: Enable GPU
1. Go to Runtime > Change runtime type
2. Select "GPU" as Hardware accelerator
3. Save

### Step 3: Install Dependencies
Run this in the first cell:
```python
!pip install transformers datasets peft accelerate bitsandbytes unsloth -q
!pip install huggingface_hub -q
```

### Step 4: Set Up API Keys
Create a cell with your API keys:
```python
import os

# Choose one or more:
os.environ["HF_TOKEN"] = "your_huggingface_token_here"
os.environ["GROQ_API_KEY"] = "your_groq_key_here"
os.environ["TOGETHER_API_KEY"] = "your_together_key_here"
```

### Step 5: Upload Your Data
Upload `data/my-writing.txt` to Colab or use:
```python
from google.colab import files
uploaded = files.upload()
```

---

## Recommended Approach for Colab

**For Fine-tuning:**
- Use Colab's free GPU with Unsloth (works perfectly in Colab!)
- No API needed for training

**For Inference/Generation:**
- Use Groq API (fastest, most free requests)
- Or Hugging Face Inference API (easy setup)

---

## Quick Start Commands

### Install in Colab:
```bash
!git clone https://github.com/your-repo/Mystyle-AI.git
%cd Mystyle-AI
!pip install -r requirements.txt
```

### Run Fine-tuning:
```python
from src.finetune import FineTuner
from src.data_processor import DataProcessor

# Process data
processor = DataProcessor()
train_dataset, val_dataset, stats = processor.process()

# Fine-tune (works in Colab with GPU!)
fine_tuner = FineTuner()
fine_tuner.load_model()
fine_tuner.setup_lora()
fine_tuner.train(train_dataset, val_dataset)
fine_tuner.save_model()
```

