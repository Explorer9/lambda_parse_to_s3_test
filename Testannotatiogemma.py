# --------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------
INPUT_FILE = 'conversations_chunked_1000.jsonl' 
OUTPUT_FILE = 'annotated_conversations.jsonl'
MODEL_ID = "google/gemma-3-4b-it"

ANNOTATION_INSTRUCTIONS = """
*** ANNOTATION GUIDELINES ***
1. INTENT = ACTION (VERB). Ignore the product noun.
2. "LOAN" RULES (CRITICAL):
   - Apply/Modify Loan -> 'Account Management'
   - View Loan Balance/Status -> 'Account Details'
   - Pay Loan -> 'Money Movement'
   - Loans are NEVER 'Investments'.

3. DIALOGUE ACTS:
   - request (Asking for action/help)
   - problem_statement (Reporting issue)
   - question (Asking for info)
   - acknowledgment ("Okay", "Thanks")

4. BOUNDARY (is_boundary):
   - True if the user changes the topic/intent compared to the PREVIOUS turn.
   - True for the very first message.

5. SEQUENTIAL LOGIC (previous_intent):
   - To find 'previous_intent', you must look at the *immediately preceding* [USER] message in the CONVERSATION LOG.
   - Classify that previous message's intent based on the rules above.
   - That classification becomes the 'previous_intent' for the current step.
   - If there is no previous user message (start of conversation), 'previous_intent' is "no_intent".
"""
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------------------------------------
INPUT_FILE = 'conversations_chunked_1000.jsonl' 
OUTPUT_FILE = 'annotated_conversations.jsonl'
MODEL_ID = "google/gemma-3-4b-it"

# YOUR NEW ANNOTATION RULES (The Brain)
ANNOTATION_INSTRUCTIONS = """
*** ANNOTATION GUIDELINES ***
1. INTENT = ACTION (VERB). Ignore the product noun.
2. "LOAN" RULES (CRITICAL):
   - Apply/Modify Loan -> 'Account Management'
   - View Loan Balance/Status -> 'Account Details'
   - Pay Loan -> 'Money Movement'
   - Loans are NEVER 'Investments'.
3. DIALOGUE ACTS:
   - request (Asking for action: "Can you help?", "I need X")
   - problem_statement (Reporting issue: "It's not working")
   - question (Asking for info: "What is X?")
   - acknowledgment ("Okay", "Thanks")
   - provide_information ("My zip is 90210")
4. BOUNDARY (is_boundary):
   - True if the user changes the topic/intent.
   - True for the very first message.
   - False if continuing the same topic.
"""

# --------------------------------------------------------------------------------
# 2. MODEL LOADING (Native FP16 for High-VRAM GPUs)
# --------------------------------------------------------------------------------
print(f"Loading {MODEL_ID} in native Float16 on 32GB GPU...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",           # Automatically finds your GPU
    torch_dtype=torch.float16,   # Standard Half-Precision (Uses ~8-9GB VRAM)
    trust_remote_code=True
)

print(f"Model loaded. VRAM Check: You have plenty of headroom.")

# --------------------------------------------------------------------------------
# 3. PROMPT CONSTRUCTOR
# --------------------------------------------------------------------------------
def construct_prompt(record):
    """
    Builds the prompt with explicit Sequential Logic.
    """
    conversation_history = ""
    last_user_text = "N/A (Start of Conversation)"
    
    # 1. Build Context & Capture Last User Message
    if 'context' in record and record['context']:
        for turn in record['context']:
            role = "[USER]" if turn['speaker'] == "Consumer" else "[AGENT]"
            conversation_history += f"{role} {turn['text']}\n"
            
            # Track the text of the last USER message for the prompt
            if turn['speaker'] == "Consumer":
                last_user_text = turn['text']
    
    current_text = record['current_turn']['text']
    conversation_history += f"[TARGET_USER] {current_text}"

    # 2. System Instruction
    prompt = f"""
    Role: Banking Data Annotator.
    Task: Annotate the [TARGET_USER] message.

    {ANNOTATION_INSTRUCTIONS}

    CONVERSATION LOG:
    {conversation_history}

    STEP-BY-STEP REASONING:
    1. Look at the "Last Previous User Message": "{last_user_text}"
    2. Classify its intent -> This is your 'previous_intent'.
    3. Look at the "[TARGET_USER]" message: "{current_text}"
    4. Classify its intent -> This is your 'current_intent'.
    5. Compare them: Did the intent change? -> Set 'is_boundary'.

    REQUIRED JSON FORMAT:
    {{
      "conversation_id": {record.get('conversation_id', 0)},
      "turn_id": {record.get('turn_id', 0)},
      "previous_intent": "...",
      "current_intent": "...",
      "dialogue_act": "...",
      "is_boundary": true/false
    }}
    """
    return prompt

# --------------------------------------------------------------------------------
# 4. MAIN PROCESSING LOOP
# --------------------------------------------------------------------------------
def process_file():
    print("Starting annotation pipeline...")
    
    # Using 'utf-8' encoding to be safe
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile):
            if not line.strip(): continue

            try:
                # Parse the line (Handling the array [ ... ] structure from your image)
                data = json.loads(line)
                
                # If the line is a list (e.g. [{"conv_id":...}]), process items inside
                records_to_process = data if isinstance(data, list) else [data]
                
                for record in records_to_process:
                    # Construct Prompt
                    user_prompt = construct_prompt(record)
                    
                    # Tokenize
                    chat = [{"role": "user", "content": user_prompt}]
                    prompt_ids = tokenizer.apply_chat_template(chat, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
                    
                    # Generate
                    outputs = model.generate(
                        prompt_ids,
                        max_new_tokens=500, 
                        do_sample=False,     # Deterministic (Greedy Search)
                        temperature=0.0      # Zero creativity = Maximum Logic
                    )
                    
                    # Decode
                    generated_text = tokenizer.decode(outputs[0][len(prompt_ids[0]):], skip_special_tokens=True)
                    
                    # Extract JSON
                    try:
                        start_idx = generated_text.find('{')
                        end_idx = generated_text.rfind('}') + 1
                        if start_idx != -1 and end_idx != -1:
                            json_str = generated_text[start_idx:end_idx]
                            annotated_record = json.loads(json_str)
                            
                            # Save to file
                            outfile.write(json.dumps(annotated_record) + '\n')
                            print(f"Annotated Conv {record.get('conversation_id')}, Turn {record.get('turn_id')}")
                        else:
                            print(f"Warning: No JSON found in output for line {line_num}")
                            # Optional: print(generated_text) to debug
                    except Exception as e:
                        print(f"JSON Parsing error on line {line_num}: {e}")
                        
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON on line {line_num}")

    print(f"Annotation complete. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    process_file()
