# Free API Setup Guide for Model Fine-tuning & Generation

## 🎯 Quick Comparison

| Service | Free Tier | Speed | Best For | Setup Difficulty |
|---------|-----------|-------|----------|------------------|
| **Groq** | 14,400 req/day | ⚡⚡⚡ Very Fast | Inference/Generation | ⭐ Easy |
| **Hugging Face** | 1,000 req/day | ⚡⚡ Fast | Model Access | ⭐ Easy |
| **Together AI** | $25 credits | ⚡⚡ Fast | Fine-tuning + Inference | ⭐⭐ Medium |
| **Replicate** | Limited | ⚡ Medium | Model Deployment | ⭐⭐ Medium |

---

## 🚀 Option 1: Groq API (Recommended for Generation)

### Why Groq?
- ✅ **Fastest inference** (up to 500 tokens/sec)
- ✅ **Generous free tier** (14,400 requests/day)
- ✅ **Supports Llama 3.1 8B**
- ✅ **Easy setup**

### Step-by-Step Setup:

1. **Sign Up**
   - Go to https://console.groq.com/
   - Click "Sign Up" (use Google/GitHub for quick signup)

2. **Get API Key**
   - After login, go to "API Keys" in the left sidebar
   - Click "Create API Key"
   - Give it a name (e.g., "Colab Fine-tuning")
   - Copy the key (starts with `gsk_...`)

3. **Use in Code**
   ```python
   import os
   os.environ["GROQ_API_KEY"] = "gsk_your_key_here"
   
   from src.api_generate import APIStyleGenerator
   generator = APIStyleGenerator(service="groq")
   result = generator.rewrite("Your text here")
   ```

**Free Limits:**
- 14,400 requests per day
- No credit card required
- Perfect for testing and small projects

---

## 🔥 Option 2: Hugging Face Inference API

### Why Hugging Face?
- ✅ **Direct access to models**
- ✅ **Good for experimentation**
- ✅ **Free tier available**

### Step-by-Step Setup:

1. **Create Account**
   - Go to https://huggingface.co/
   - Click "Sign Up" (free)

2. **Get Access Token**
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Name it (e.g., "Colab API")
   - Select "Read" permissions
   - Click "Generate token"
   - **Copy the token** (starts with `hf_...`)

3. **Request Model Access** (for Llama models)
   - Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - Click "Agree and access repository"
   - Fill out the form (takes 1-2 minutes)
   - Wait for approval (usually instant)

4. **Use in Code**
   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_your_token_here"
   
   from src.api_generate import APIStyleGenerator
   generator = APIStyleGenerator(service="hf")
   result = generator.rewrite("Your text here")
   ```

**Free Limits:**
- 1,000 requests per day
- Rate limited but sufficient for testing

---

## 💎 Option 3: Together AI (Best for Fine-tuning)

### Why Together AI?
- ✅ **Supports fine-tuning**
- ✅ **$25 free credits**
- ✅ **Good for production**

### Step-by-Step Setup:

1. **Sign Up**
   - Go to https://together.ai/
   - Click "Sign Up" (free)

2. **Get API Key**
   - After login, go to "API Keys"
   - Click "Create API Key"
   - Copy the key

3. **Use in Code**
   ```python
   import os
   os.environ["TOGETHER_API_KEY"] = "your_key_here"
   
   from src.api_generate import APIStyleGenerator
   generator = APIStyleGenerator(service="together")
   result = generator.rewrite("Your text here")
   ```

**Free Limits:**
- $25 free credits
- ~500,000 tokens (enough for testing)
- Pay-as-you-go after free tier

---

## 🎨 Option 4: Replicate API

### Why Replicate?
- ✅ **Easy model deployment**
- ✅ **Good for demos**

### Step-by-Step Setup:

1. **Sign Up**
   - Go to https://replicate.com/
   - Sign up with GitHub/Google

2. **Get API Token**
   - Go to Account Settings > API Tokens
   - Click "Create token"
   - Copy the token (starts with `r8_...`)

3. **Use in Code**
   ```python
   import os
   os.environ["REPLICATE_API_TOKEN"] = "r8_your_token_here"
   
   from src.api_generate import APIStyleGenerator
   generator = APIStyleGenerator(service="replicate")
   result = generator.rewrite("Your text here")
   ```

**Free Limits:**
- Limited free credits
- Good for testing

---

## 📝 Complete Example for Colab

```python
# Cell 1: Install dependencies
!pip install requests -q

# Cell 2: Set API key
import os
os.environ["GROQ_API_KEY"] = "gsk_your_key_here"  # Get from https://console.groq.com/

# Cell 3: Use the generator
from src.api_generate import APIStyleGenerator

# Initialize
generator = APIStyleGenerator(service="groq")

# Rewrite text
text = "Artificial intelligence is transforming every industry."
rewritten = generator.rewrite(text)
print("Original:", text)
print("\nRewritten:", rewritten)

# Write about topic
topic = "The future of AI"
generated = generator.write_about(topic, max_length=300)
print(f"\nTopic: {topic}")
print(f"Generated:\n{generated}")
```

---

## 🎯 Recommended Workflow for Colab

### For Fine-tuning:
1. **Use Colab's free GPU** (Runtime > Change runtime type > GPU)
2. **Use Unsloth** (works perfectly in Colab!)
3. **No API needed** for training

### For Generation/Inference:
1. **Use Groq API** (fastest, most free requests)
2. **Or Hugging Face API** (easy setup)
3. **Set API key** in Colab environment

---

## 🔒 Security Tips

1. **Never commit API keys to GitHub**
   - Use environment variables
   - Use Colab's "Secrets" feature
   - Use `.env` files (add to `.gitignore`)

2. **Rotate keys regularly**
   - Delete old keys
   - Create new ones if compromised

3. **Use read-only tokens when possible**
   - Hugging Face: Use "Read" token for inference
   - Only use "Write" tokens when needed

---

## 🆘 Troubleshooting

### "API key not found"
- Check environment variable name matches
- Restart Colab runtime after setting env var
- Use `os.getenv("YOUR_KEY")` to verify

### "Rate limit exceeded"
- Wait a few minutes
- Switch to a different service
- Upgrade to paid tier if needed

### "Model not found"
- For Hugging Face: Request access to the model
- Check model name is correct
- Verify API key has proper permissions

---

## 📚 Additional Resources

- **Groq Documentation**: https://console.groq.com/docs
- **Hugging Face API Docs**: https://huggingface.co/docs/api-inference
- **Together AI Docs**: https://docs.together.ai/
- **Replicate Docs**: https://replicate.com/docs

---

## ✅ Quick Checklist

- [ ] Choose a service (Groq recommended)
- [ ] Create account
- [ ] Get API key
- [ ] Set environment variable
- [ ] Test with simple request
- [ ] Start fine-tuning in Colab!

