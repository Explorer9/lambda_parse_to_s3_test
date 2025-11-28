import pandas as pd
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------
INPUT_FILE = 'dataset.csv'          # Replace with your actual filename
OUTPUT_FILE = 'gemma_corrections.csv'
BATCH_SIZE = 10                     # Keep small for local models (10-20)

# Replace this with your specific local model path or HuggingFace ID
# If you meant Gemma 2B Instruction Tuned, use "google/gemma-2b-it" or "google/gemma-2-2b-it"
MODEL_ID = "google/gemma-2-2b-it" 

# device_map="auto" moves model to GPU if available. 
# torch_dtype=torch.float16 reduces memory usage by half.
print(f"Loading model: {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    device_map="auto", 
    torch_dtype=torch.float16
)
print("Model loaded successfully.")

# --------------------------------------------------------------------------------
# 2. THE PROMPT LOGIC (Designed for Small Models)
# --------------------------------------------------------------------------------
def construct_prompt(batch_df):
    """
    Creates the prompt string with Few-Shot examples to guide the small model.
    """
    # Convert batch to JSON string for the prompt
    batch_json = batch_df[['id', 'combined_text', 'current_intent', 'dialogueact']].to_json(orient='records')
    
    prompt = f"""
    Role: You are a Data Quality Assistant. 
    Task: Review banking logs and fix errors in 'current_intent' (Action) and 'dialogueact'.

    RULES:
    1. Intent is the ACTION (Apply, Pay, Check), not the Product.
    2. "Loans/Mortgages" are Account Management (Apply) or Account Details (View). NEVER 'Investments'.
    3. "Can you..." questions asking for action are 'request', NOT 'question'.

    EXAMPLES:
    Input: {{ "combined_text": "[TARGET_USER] apply for loan", "current_intent": "Investments" }}
    Output: {{ "id": "...", "new_intent": "Account Management", "reasoning": "Loan application is Management" }}

    Input: {{ "combined_text": "[TARGET_USER] whats my balance", "dialogueact": "request" }}
    Output: {{ "id": "...", "new_act": "question", "reasoning": "Asking for info is a question" }}

    CURRENT BATCH to Process:
    {batch_json}

    INSTRUCTIONS:
    - Analyze the batch above.
    - Return a JSON list of objects for ONLY the rows with errors.
    - Format: [ {{ "id": "...", "original_intent": "...", "new_intent": "...", "error_flag": true, "reasoning": "..." }} ]
    - If no errors, return: []
    - Output JSON ONLY. No markdown, no explanations outside the JSON.
    """
    return prompt

# --------------------------------------------------------------------------------
# 3. GENERATION & PARSING
# --------------------------------------------------------------------------------
def get_corrections(batch_df):
    user_prompt = construct_prompt(batch_df)
    
    # Format for Gemma (using the chat template is crucial for performance)
    chat = [
        { "role": "user", "content": user_prompt },
    ]
    prompt_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

    # Generate
    outputs = model.generate(
        prompt_ids, 
        max_new_tokens=1024, # Enough space for the JSON output
        do_sample=False,     # Deterministic (better for data cleaning)
        temperature=0.0      # Zero temp for strict logic
    )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0][len(prompt_ids[0]):], skip_special_tokens=True)
    
    return parse_json_output(generated_text)

def parse_json_output(text):
    """
    Robust parsing to find JSON list inside the model's chatter.
    """
    try:
        # 1. Try strict JSON load
        return json.loads(text)
    except:
        # 2. Try regex to find the list [...]
        try:
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
            
    # If parsing fails or model returns text, return empty (safest for automation)
    # print(f"Warning: Could not parse JSON. Raw output: {text[:100]}...")
    return []

# --------------------------------------------------------------------------------
# 4. MAIN LOOP
# --------------------------------------------------------------------------------
def main():
    # Load Data
    try:
        df = pd.read_csv(INPUT_FILE)
        # Ensure ID column exists (or create index)
        if 'id' not in df.columns:
            df['id'] = df.index
    except FileNotFoundError:
        print(f"Error: {INPUT_FILE} not found.")
        return

    all_corrections = []
    total_batches = (len(df) // BATCH_SIZE) + 1

    print(f"Starting processing of {len(df)} records in {total_batches} batches.")

    for i in range(0, len(df), BATCH_SIZE):
        batch = df.iloc[i : i + BATCH_SIZE]
        
        print(f"Processing batch {i // BATCH_SIZE + 1}/{total_batches}...", end="\r")
        
        try:
            corrections = get_corrections(batch)
            if corrections:
                all_corrections.extend(corrections)
        except Exception as e:
            print(f"\nError in batch {i}: {e}")
            # Continue to next batch even if one fails
            continue

    print("\nProcessing complete.")

    # Save Results
    if all_corrections:
        result_df = pd.DataFrame(all_corrections)
        result_df.to_csv(OUTPUT_FILE, index=False)
        print(f"Found {len(all_corrections)} errors. Saved to {OUTPUT_FILE}")
    else:
        print("No errors found or model output format failed to parse.")

if __name__ == "__main__":
    main()
