# --------------------------------------------------------------------------------
# 1. CONFIGURATION (Updated)
# --------------------------------------------------------------------------------
# Paste the FULL text of the Refactored Instructions we generated earlier here.
# (I have included the Critical Changes summary here, but you can paste the full doc if you prefer)
# --------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------

NEW_GUIDELINES = """
*** CRITICAL RULES (PRIORITY) ***
1. LOANS & MORTGAGES:
   - Apply/Modify -> 'Account Management'
   - View Balance/Rate -> 'Account Details'
   - Pay -> 'Money Movement'
   - NEVER classify Loans as 'Investments'.
   
2. INTENT = ACTION:
   - Ignore the product noun. Focus on the Verb (Apply, Check, Pay).

3. DIALOGUE ACTS:
   - "Can you [do action]?" is a REQUEST, not a question.

*** DETAILED ANNOTATION DEFINITIONS ***
(Paste your full generated document content here if you want the model to have definitions for things like 'Card Maintenance' or 'Security')
...
...
"""

# --------------------------------------------------------------------------------
# 2. THE PROMPT LOGIC
# --------------------------------------------------------------------------------
def construct_prompt(batch_df):
    """
    Injects the Guidelines + Few-Shot Examples + Data Batch
    """
    # Convert batch to JSON string
    batch_json = batch_df[['id', 'combined_text', 'current_intent', 'dialogueact']].to_json(orient='records')
    
    prompt = f"""
    Role: You are a Data Quality Assistant for a Banking Bot.
    Task: Validate and Correct 'current_intent' and 'dialogueact' based on the Guidelines below.

    {NEW_GUIDELINES}

    FEW-SHOT EXAMPLES (How to fix errors):
    Input: {{ "combined_text": "...[TARGET_USER] I want to get a mortgage", "current_intent": "Investments" }}
    Output: {{ "id": "...", "new_intent": "Account Management", "reasoning": "Mortgage Application is Account Management (Action: Apply), not Investments." }}

    Input: {{ "combined_text": "...[TARGET_USER] what is the rate on my car loan", "current_intent": "Investments" }}
    Output: {{ "id": "...", "new_intent": "Account Details", "reasoning": "Viewing loan rate is Account Details (Action: View)." }}

    Input: {{ "combined_text": "...[TARGET_USER] Can you tell me my balance?", "dialogueact": "question" }}
    Output: {{ "id": "...", "new_act": "request", "reasoning": "User is asking for an action (Tell me), which is a Request." }}

    CURRENT BATCH TO PROCESS:
    {batch_json}

    INSTRUCTIONS:
    1. Read the Batch.
    2. Compare 'current_intent' and 'dialogueact' against the GUIDELINES.
    3. Return a JSON list of objects for ONLY the rows that are WRONG.
    4. JSON Format: [ {{ "id": "...", "original_intent": "...", "new_intent": "...", "error_flag": true, "reasoning": "..." }} ]
    5. If all rows are correct, return: []
    """
    return prompt




import re

# --------------------------------------------------------------------------------
# HELPER: ROBUST JSON PARSER
# --------------------------------------------------------------------------------
def clean_and_parse_json(text):
    """
    Attempts to extract and parse JSON from LLM output, handling markdown and common errors.
    """
    # 1. Strip Markdown Code Blocks (```json ... ```)
    if "```" in text:
        # Find the content inside the first code block
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
    
    # 2. Extract strictly from first '{' to last '}'
    # This removes any "Here is the JSON:" prefix text
    start_idx = text.find('{')
    end_idx = text.rfind('}') + 1
    if start_idx == -1 or end_idx == 0:
        return None, "No JSON brackets found"
    
    json_candidate = text[start_idx:end_idx]
    
    # 3. Try Parsing
    try:
        return json.loads(json_candidate), None
    except json.JSONDecodeError as e:
        # 4. Common Fix: The model sometimes forgets to escape quotes inside strings
        # This is a risky heuristic, but helps for simple cases if needed.
        # For now, we return the error to let the user see it.
        return None, str(e)

# --------------------------------------------------------------------------------
# MAIN PROCESSING LOOP (UPDATED)
# --------------------------------------------------------------------------------
def process_file():
    print("Starting annotation pipeline...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile):
            if not line.strip(): continue

            try:
                data = json.loads(line)
                records_to_process = data if isinstance(data, list) else [data]
                
                for record in records_to_process:
                    user_prompt = construct_prompt(record)
                    
                    # Prepare Inputs
                    chat = [{"role": "user", "content": user_prompt}]
                    inputs = tokenizer.apply_chat_template(
                        chat, 
                        tokenize=True, 
                        add_generation_prompt=True, 
                        return_tensors="pt",
                        return_dict=True
                    ).to(model.device)
                    
                    # Generate
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=500, 
                        do_sample=False,     
                        temperature=0.0,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                    
                    # Decode
                    generated_ids = outputs[0][len(inputs['input_ids'][0]):]
                    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    # --- NEW PARSING LOGIC ---
                    annotated_record, error_msg = clean_and_parse_json(generated_text)
                    
                    if annotated_record:
                        # Success! Write to file
                        outfile.write(json.dumps(annotated_record) + '\n')
                        print(f"Annotated Conv {record.get('conversation_id')}, Turn {record.get('turn_id')}")
                    else:
                        # FAILURE: Print the RAW output so you can see what broke
                        print(f"\n[ERROR] Line {line_num} JSON Parse Failed: {error_msg}")
                        print(f"--- MODEL OUTPUT START ---\n{generated_text}\n--- MODEL OUTPUT END ---\n")
                        
            except Exception as e:
                print(f"General Error on line {line_num}: {e}")

    print(f"Annotation complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_file()
   
