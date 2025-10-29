# app/utils/model_utils.py
from typing import Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model_tokenizer(model_name: str) -> Tuple[AutoModelForSeq2SeqLM, AutoTokenizer, str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = get_device()
    if device == "cuda":
        model = model.to(device)
    model.eval()
    return model, tokenizer, device


def generate_summary(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    text: str,
    device: str,
    max_length: int,
    min_length: int,
    num_beams: int = 4,
    length_penalty: float = 2.0,
    early_stopping: bool = True,
) -> str:
    inputs = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    with torch.no_grad():
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=early_stopping,
        )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


