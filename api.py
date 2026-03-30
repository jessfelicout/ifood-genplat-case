from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI(
    title="Generative Fine-Tuned API",
    description="API for fine-tuned",
    version="1.0"
)

# load trained data
model_path = "./output"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

@app.post("/predict")
def predict(prompt: str, max_length: int = 50):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=max_length)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return {"prompt": prompt, "generated_text": text}