import os
from unittest.mock import patch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.dynamic_module_utils import get_imports
import torch

MODEL = "deepseek-ai/DeepSeek-Coder-V2-Lite-Base"


def fixed_get_imports(filename: str | os.PathLike) -> list[str]:
    """
    Helper function to remove flash_attn dependency on non-CUDA devices.
    Used as a monkey patch for Apple Silicon compatibility.
    """
    imports = get_imports(filename)
    if not torch.cuda.is_available() and "flash_attn" in imports:
        imports.remove("flash_attn")
    return imports


def load_model():
    """
    Load and configure the DeepSeek model and tokenizer.
    On Apple Silicon, the DeepSeek Coder model requires a monkey patch to remove the `flash_attn` dependency.
    https://huggingface.co/qnguyen3/nanoLLaVA-1.5/discussions/4
    """
    device = "cuda" if torch.cuda.is_available() else "mps"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with patch("transformers.dynamic_module_utils.get_imports", fixed_get_imports):
        tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        ).to(device)

    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = load_model()
    prompt = "#write a quick sort algorithm"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=128)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(result)
