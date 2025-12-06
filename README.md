# MyStyle AI Writer

Fine-tune Llama 3.1 8B (or Mistral 7B) to learn your writing style and generate text that matches your unique voice.

## 🚀 Features

- **Fine-tune models** on your writing samples using LoRA/QLoRA
- **Style transfer** - Rewrite any text in your writing style
- **Text generation** - Generate content about any topic in your style
- **Google Colab ready** - Complete notebook for GPU-based fine-tuning
- **API integration** - Use Mistral, Groq, or Hugging Face APIs for generation
- **No approval needed** - Uses Mistral 7B by default (completely open!)

## 📋 Quick Start

### Option 1: Google Colab (Recommended)

1. Open `colab_finetune.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Enable GPU: Runtime > Change runtime type > GPU
3. Replace `YOUR_USERNAME` with your GitHub username in the first cell
4. Run cells in order
5. Upload your writing sample when prompted
6. Fine-tune and generate!

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/ishuu19/MyStyleAI-Writer.git
cd MyStyleAI-Writer

# Install dependencies
pip install -r requirements.txt

# Run the main script
python main.py
```

## 📁 Project Structure

```
MyStyleAI-Writer/
├── main.py                 # Main execution script
├── colab_finetune.ipynb    # Google Colab notebook
├── src/
│   ├── setup.py           # Environment verification
│   ├── data_processor.py  # Data processing and dataset creation
│   ├── finetune.py        # Model fine-tuning with LoRA
│   ├── deploy.py          # Model deployment to Ollama
│   ├── style_transfer.py  # Style transfer functionality
│   ├── generate.py        # Text generation (Ollama)
│   └── api_generate.py    # Text generation (API-based)
├── data/
│   └── my-writing.txt     # Your writing sample
└── checkpoints/           # Saved models
```

## 🎯 Usage

### Fine-tuning

```python
from src.finetune import FineTuner
from src.data_processor import DataProcessor

# Process your data
processor = DataProcessor()
train_dataset, val_dataset, stats = processor.process()

# Fine-tune model (uses Mistral 7B by default - no approval needed!)
fine_tuner = FineTuner()
fine_tuner.load_model()
fine_tuner.setup_lora()
fine_tuner.train(train_dataset, val_dataset)
fine_tuner.save_model()
```

### Text Generation

```python
from src.api_generate import APIStyleGenerator

# Initialize generator (Mistral API is pre-configured!)
generator = APIStyleGenerator(service="mistral")

# Rewrite text in your style
rewritten = generator.rewrite("Your text here")

# Write about a topic
generated = generator.write_about("The importance of education")
```

## 🔧 Configuration

### Models

- **Default**: Mistral 7B (`unsloth/Mistral-7B-Instruct-v0.3-bnb-4bit`)
  - No approval needed
  - Excellent for style transfer
  - Fast training (20-40 minutes)

- **Alternative**: Phi-3 Mini (`unsloth/Phi-3-mini-4k-instruct`)
  - Fastest training (10-20 minutes)
  - Good for quick experiments

### API Services

- **Mistral AI** (Pre-configured) - Official Mistral API
- **Groq** - 14,400 free requests/day, very fast
- **Hugging Face** - 1,000 free requests/day

## 📚 Documentation

- `API_SETUP_GUIDE.md` - Complete guide for setting up free API services
- `colab_setup.md` - Google Colab setup instructions

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- Transformers
- Unsloth (for fine-tuning)
- GPU recommended (works in free Colab!)

## 📝 License

This project is open source and available for personal and commercial use.

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Made with ❤️ for personalized AI writing**
