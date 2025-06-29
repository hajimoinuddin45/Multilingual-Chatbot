# Step 2: Load Pretrained Models and Translator
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator
import torch

# Load DialoGPT
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

# Load Google Translate API
translator = Translator()