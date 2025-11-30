# --------------------------------------------------------------------------------
# 1. CONFIGURATION (Updated)
# --------------------------------------------------------------------------------
# Paste the FULL text of the Refactored Instructions we generated earlier here.
# (I have included the Critical Changes summary here, but you can paste the full doc if you prefer)
NEW_GUIDELINES = """
OFFICIAL ANNOTATION GUIDELINES (HYBRID MODEL):

1. INTENT = ACTION (VERB)
   - Do not classify based on the product (Noun).
   - "Apply for [Product]" -> Account Management
   - "Check balance of [Product]" -> Account Details
   - "Pay [Product]" -> Money Movement

2. LOAN & MORTGAGE RULES (CRITICAL CHANGE):
   - Loans are NEVER 'Investments'.
   - Applying/Modifying a Loan -> Account Management
   - Viewing Loan Balance/Rate -> Account Details
   - Paying a Loan -> Money Movement

3. DIALOGUE ACT PRIORITY:
   - request > problem_statement > question
   - Rule: "Can you help me transfer?" is a REQUEST, not a question.
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
