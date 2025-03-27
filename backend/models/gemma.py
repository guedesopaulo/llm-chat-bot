# backend/models/gemma.py
from transformers import Gemma3ForConditionalGeneration, AutoProcessor
import torch

MODEL_NAME = "google/gemma-3-4b-it"

def load_gemma_chat():
    model = Gemma3ForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    processor = AutoProcessor.from_pretrained(MODEL_NAME)

    return processor, model
