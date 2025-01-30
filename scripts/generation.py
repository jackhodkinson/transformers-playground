# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "mistralai/Mistral-7B-v0.1"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
print("loaded tokenizer")
model = AutoModelForCausalLM.from_pretrained(MODEL)
print("loaded model")
model_inputs = tokenizer(["A list of colors: red, blue"], return_tensors="pt").to("mps")
print("calculated model inputs")
generated_ids = model.generate(**model_inputs)
print("generated ids")
result = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("the decoded result is:")
print(result)
