# Install Hugging Face libraries
# !pip install -q transformers datasets evaluate
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# Import main classes for code generation
from transformers import AutoModelForCausalLM, AutoTokenizer

# I. Load the Code LLM

from transformers import AutoModelForCausalLM, AutoTokenizer

# Hugging Face model repo (Qwen2.5 Coder, 1.5B parameters, instruction-tuned version)
model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

# Load the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",   # automatically chooses best precision (saves memory)
    device_map="auto"     # automatically uses GPU if available
)

# Load the tokenizer (responsible for converting text ↔ tokens)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# II. Prepare the Prompt

# Define the task we want the model to perform
prompt = "write a quick sort algorithm in Python."

# Chat-style messages for instruction-tuned LLMs
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Apply the chat template to convert messages into model-readable input
# - `tokenize=False`: we want text, not token IDs yet
# - `add_generation_prompt=True`: adds the model's generation cue
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Show the prepared prompt
print("=== Prepared Input for the Model ===")
print(text)

# III. Generate Code from the Prompt

# Tokenize the prepared prompt and move to the same device as the model
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate code
# - max_new_tokens: limits the number of tokens generated
# - other parameters (temperature, top_p) can control creativity
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# Remove the input tokens from the output to get only the newly generated tokens
generated_ids_only = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode token IDs back to readable text
response = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)[0]

# Display the generated code
print("=== Generated Code ===")
print(response)

# I. Prepare Code Translation Prompt

# Define source and target languages
source_programming_language = "Java"
target_programming_language = "Python"

# Example code snippet in the source language
code = """
public static boolean isEven(int number) {
    return number % 2 == 0;
}
"""

# Prepare the translation instruction
translation_prompt = f"Translate this {source_programming_language} code into {target_programming_language}:\n{code}"

messages = [
    {"role": "system", "content": "You are Qwen, an expert programmer."},
    {"role": "user", "content": translation_prompt}
]

# Prepare model input
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# II. Run the Translation

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate translation
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

# Remove prompt portion
generated_ids_only = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode and print
translated_code = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)[0]
print("=== Translated Code ===")
print(translated_code)

# I. Code Explanation Prompt

code_to_explain = """
def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
"""

explanation_prompt = f"Explain what the following Python code does:\n{code_to_explain}"

messages = [
    {"role": "system", "content": "You are Qwen, an expert code explainer."},
    {"role": "user", "content": explanation_prompt}
]

# Prepare model input
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# II. Generate Explanation

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)

generated_ids_only = [
    output_ids[len(input_ids):]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode and print explanation
explanation = tokenizer.batch_decode(generated_ids_only, skip_special_tokens=True)[0]
print("=== Code Explanation ===")
print(explanation)

