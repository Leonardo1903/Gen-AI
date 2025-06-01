import os
import torch
from dotenv import load_dotenv
from transformers import AutoTokenizer,AutoModelForCausalLM, pipeline

load_dotenv()

# os.environ["HF_TOKEN"] = os.getenv("HUGGINGFACE_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

model_name ="google/gemma-3-1b-it"

# Load the tokenizer 
tokenizer=AutoTokenizer.from_pretrained(model_name,
                                        token=huggingface_api_key,
)
print(tokenizer("Hello, how are you?"))
print(tokenizer.get_vocab_size())

input_tokens = tokenizer("Hello, how are you?")["input_ids"]
print(input_tokens)

# Load the model
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype=torch.bfloat16)

gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

gen_pipeline("Hello, how are you?",max_new_tokens=50)



# Make pipeline from scratch
input_prompt = [
    "The capital of India is"
]


tokenized = tokenizer(input_prompt, return_tensors="pt")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16
)

gen_result = model.generate(tokenized["input_ids"], max_new_tokens=25)

print(gen_result)

output = tokenizer.batch_decode(gen_result)
print(output)
