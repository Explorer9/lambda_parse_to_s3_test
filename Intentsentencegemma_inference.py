```python
"""
Inference code for SentenceTransformer model with 3 classifier heads.
(Intent + Boundary + Dialogue Act)
"""

import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import json

# ============================================================================
# 1. MODEL CLASS (Same as training)
# ============================================================================

class SentenceTransformerIntentClassifier(nn.Module):
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_intents: int = 20,
        num_dialogue_acts: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.encoder = SentenceTransformer(model_name)
        encoder_dim = self.encoder.get_sentence_embedding_dimension()
        
        self.prev_intent_embedding = nn.Embedding(num_intents, 64)
        combined_dim = encoder_dim + 64
        
        self.shared_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        self.intent_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_intents)
        )
        
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 2)
        )
        
        self.dialogue_act_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_dialogue_acts)
        )
    
    def forward(self, texts, prev_intent):
        text_embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        prev_intent_emb = self.prev_intent_embedding(prev_intent)
        combined = torch.cat([text_embeddings, prev_intent_emb], dim=-1)
        shared_repr = self.shared_layer(combined)
        
        intent_logits = self.intent_classifier(shared_repr)
        boundary_logits = self.boundary_detector(shared_repr)
        dialogue_act_logits = self.dialogue_act_classifier(shared_repr)
        
        return {
            'intent_logits': intent_logits,
            'boundary_logits': boundary_logits,
            'dialogue_act_logits': dialogue_act_logits
        }


# ============================================================================
# 2. LOAD MODEL AND MAPPINGS
# ============================================================================

def load_model(checkpoint_path='best_model.pt', device='cuda'):
    """Load trained model from checkpoint."""
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get label mappings from checkpoint
    intent2idx = checkpoint['intent2idx']
    dialogue_act2idx = checkpoint['dialogue_act2idx']
    unique_intents = checkpoint['unique_intents']
    unique_dialogue_acts = checkpoint['unique_dialogue_acts']
    
    # Create reverse mappings
    idx2intent = {idx: label for label, idx in intent2idx.items()}
    idx2dialogue_act = {idx: label for label, idx in dialogue_act2idx.items()}
    
    # Initialize model
    model = SentenceTransformerIntentClassifier(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_intents=len(unique_intents),
        num_dialogue_acts=len(unique_dialogue_acts),
        hidden_dim=512
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    print(f"Validation F1: {checkpoint['val_f1']:.4f}")
    print(f"Intents: {len(unique_intents)}")
    print(f"Dialogue Acts: {len(unique_dialogue_acts)}")
    
    return model, intent2idx, idx2intent, dialogue_act2idx, idx2dialogue_act


# ============================================================================
# 3. SINGLE PREDICTION
# ============================================================================

def predict_single(
    model,
    text,
    previous_intent,
    intent2idx,
    idx2intent,
    idx2dialogue_act,
    device='cuda'
):
    """
    Predict intent, boundary, and dialogue act for a single text.
    
    Args:
        model: Trained model
        text: Input text (e.g., "[USER] Hi [AGENT] Hello [TARGET_USER] I need help")
        previous_intent: Previous intent string (e.g., "no_intent")
        intent2idx: Intent to index mapping
        idx2intent: Index to intent mapping
        idx2dialogue_act: Index to dialogue act mapping
        device: Device to run on
    
    Returns:
        Dictionary with predictions
    """
    model.eval()
    
    # Convert previous intent to index
    prev_intent_idx = intent2idx[previous_intent]
    prev_intent_tensor = torch.tensor([prev_intent_idx]).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model([text], prev_intent_tensor)
    
    # Get predictions and probabilities
    intent_probs = torch.softmax(outputs['intent_logits'], dim=-1)
    boundary_probs = torch.softmax(outputs['boundary_logits'], dim=-1)
    dialogue_act_probs = torch.softmax(outputs['dialogue_act_logits'], dim=-1)
    
    intent_idx = intent_probs.argmax(dim=-1).item()
    intent_conf = intent_probs.max().item()
    
    boundary_idx = boundary_probs.argmax(dim=-1).item()
    boundary_conf = boundary_probs[0, 1].item()  # Prob of boundary=True
    
    dialogue_act_idx = dialogue_act_probs.argmax(dim=-1).item()
    dialogue_act_conf = dialogue_act_probs.max().item()
    
    return {
        'intent': idx2intent[intent_idx],
        'intent_confidence': intent_conf,
        'is_boundary': boundary_idx == 1,
        'boundary_confidence': boundary_conf,
        'dialogue_act': idx2dialogue_act[dialogue_act_idx],
        'dialogue_act_confidence': dialogue_act_conf,
        'intent_probabilities': {
            idx2intent[i]: prob.item() 
            for i, prob in enumerate(intent_probs[0])
        }
    }


# ============================================================================
# 4. BATCH PREDICTION
# ============================================================================

def predict_batch(
    model,
    texts,
    previous_intents,
    intent2idx,
    idx2intent,
    idx2dialogue_act,
    device='cuda',
    batch_size=32
):
    """
    Predict for multiple texts in batches.
    
    Args:
        texts: List of text strings
        previous_intents: List of previous intent strings (same length as texts)
        batch_size: Batch size for inference
    
    Returns:
        List of prediction dictionaries
    """
    model.eval()
    results = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_prev_intents = previous_intents[i:i+batch_size]
        
        # Convert to indices
        prev_intent_indices = [intent2idx[pi] for pi in batch_prev_intents]
        prev_intent_tensor = torch.tensor(prev_intent_indices).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(batch_texts, prev_intent_tensor)
        
        # Process outputs
        intent_probs = torch.softmax(outputs['intent_logits'], dim=-1)
        boundary_probs = torch.softmax(outputs['boundary_logits'], dim=-1)
        dialogue_act_probs = torch.softmax(outputs['dialogue_act_logits'], dim=-1)
        
        for j in range(len(batch_texts)):
            intent_idx = intent_probs[j].argmax().item()
            boundary_idx = boundary_probs[j].argmax().item()
            dialogue_act_idx = dialogue_act_probs[j].argmax().item()
            
            results.append({
                'text': batch_texts[j],
                'intent': idx2intent[intent_idx],
                'intent_confidence': intent_probs[j].max().item(),
                'is_boundary': boundary_idx == 1,
                'boundary_confidence': boundary_probs[j, 1].item(),
                'dialogue_act': idx2dialogue_act[dialogue_act_idx],
                'dialogue_act_confidence': dialogue_act_probs[j].max().item()
            })
    
    return results


# ============================================================================
# 5. CONVERSATION-LEVEL INFERENCE
# ============================================================================

class ConversationInference:
    """
    Stateful inference for processing entire conversations turn-by-turn.
    Tracks previous intent automatically.
    """
    
    def __init__(self, model, intent2idx, idx2intent, idx2dialogue_act, device='cuda'):
        self.model = model
        self.intent2idx = intent2idx
        self.idx2intent = idx2intent
        self.idx2dialogue_act = idx2dialogue_act
        self.device = device
        
        self.reset()
    
    def reset(self):
        """Reset conversation state."""
        self.current_intent = 'no_intent'
        self.conversation_history = []
    
    def predict_turn(self, text):
        """
        Predict for next turn in conversation.
        Automatically uses previous intent from state.
        
        Args:
            text: Input text for current turn
        
        Returns:
            Prediction dictionary
        """
        result = predict_single(
            self.model,
            text,
            self.current_intent,
            self.intent2idx,
            self.idx2intent,
            self.idx2dialogue_act,
            self.device
        )
        
        # Update state
        if result['intent'] not in ['no_intent', 'continuation']:
            self.current_intent = result['intent']
        
        # Store in history
        self.conversation_history.append({
            'text': text,
            'result': result
        })
        
        return result
    
    def get_conversation_summary(self):
        """Get summary of all predictions in conversation."""
        return self.conversation_history


# ============================================================================
# 6. EXAMPLE USAGE
# ============================================================================

def main():
    """Example usage of inference functions."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    model, intent2idx, idx2intent, dialogue_act2idx, idx2dialogue_act = load_model(
        'best_model.pt', 
        device=device
    )
    
    print("\n" + "=" * 80)
    print("SINGLE PREDICTION EXAMPLE")
    print("=" * 80)
    
    # Single prediction
    text = "[USER] I can't log in [AGENT] What error do you see? [TARGET_USER] Invalid password"
    previous_intent = "no_intent"
    
    result = predict_single(
        model, text, previous_intent,
        intent2idx, idx2intent, idx2dialogue_act, device
    )
    
    print(f"\nInput: {text}")
    print(f"Previous Intent: {previous_intent}")
    print(f"\nPredictions:")
    print(f"  Intent: {result['intent']} (confidence: {result['intent_confidence']:.3f})")
    print(f"  Boundary: {result['is_boundary']} (confidence: {result['boundary_confidence']:.3f})")
    print(f"  Dialogue Act: {result['dialogue_act']} (confidence: {result['dialogue_act_confidence']:.3f})")
    
    print("\n  Top 3 Intent Probabilities:")
    top_intents = sorted(result['intent_probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]
    for intent, prob in top_intents:
        print(f"    {intent:30s}: {prob:.3f}")
    
    print("\n" + "=" * 80)
    print("CONVERSATION INFERENCE EXAMPLE")
    print("=" * 80)
    
    # Conversation-level inference
    conv_inference = ConversationInference(model, intent2idx, idx2intent, idx2dialogue_act, device)
    
    conversation_turns = [
        "[USER] Hi [AGENT] Hello, how can I help?",
        "[TARGET_USER] I need to dispute a transaction",
        "[AGENT] Which transaction? [TARGET_USER] The $50 charge from yesterday",
        "[AGENT] I'll help with that [TARGET_USER] Thanks",
        "[AGENT] Anything else? [TARGET_USER] Yes, what are your ATM fees?"
    ]
    
    print("\nProcessing conversation turn-by-turn:\n")
    for i, turn in enumerate(conversation_turns, 1):
        result = conv_inference.predict_turn(turn)
        print(f"Turn {i}: {turn[:60]}...")
        print(f"  → Intent: {result['intent']:20s} | Boundary: {result['is_boundary']} | Act: {result['dialogue_act']}")
    
    print("\n" + "=" * 80)
    print("BATCH PREDICTION EXAMPLE")
    print("=" * 80)
    
    # Batch prediction
    texts = [
        "[USER] I forgot my password [TARGET_USER] Can you reset it?",
        "[USER] What are your hours? [TARGET_USER] Are you open on weekends?",
        "[USER] I want to close my account [TARGET_USER] How do I do that?"
    ]
    previous_intents = ["no_intent", "no_intent", "no_intent"]
    
    results = predict_batch(
        model, texts, previous_intents,
        intent2idx, idx2intent, idx2dialogue_act,
        device, batch_size=32
    )
    
    print(f"\nProcessed {len(results)} examples:\n")
    for i, result in enumerate(results, 1):
        print(f"{i}. Intent: {result['intent']:20s} | Confidence: {result['intent_confidence']:.3f}")


# ============================================================================
# 7. SAVE/LOAD PREDICTIONS
# ============================================================================

def save_predictions(predictions, output_file='predictions.json'):
    """Save predictions to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"Saved {len(predictions)} predictions to {output_file}")


def predict_from_csv(model, csv_path, intent2idx, idx2intent, idx2dialogue_act, device='cuda'):
    """Predict for all examples in a CSV file."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    texts = df['text'].tolist()
    previous_intents = df['previous_intent'].tolist()
    
    results = predict_batch(
        model, texts, previous_intents,
        intent2idx, idx2intent, idx2dialogue_act,
        device
    )
    
    # Add predictions to dataframe
    df['predicted_intent'] = [r['intent'] for r in results]
    df['intent_confidence'] = [r['intent_confidence'] for r in results]
    df['predicted_boundary'] = [r['is_boundary'] for r in results]
    df['predicted_dialogue_act'] = [r['dialogue_act'] for r in results]
    
    return df


# ============================================================================
# RUN INFERENCE
# ============================================================================

if __name__ == "__main__":
    main()
```

**Usage examples:**

```python
# 1. Single prediction
from inference import load_model, predict_single

model, intent2idx, idx2intent, _, idx2dialogue_act = load_model('best_model.pt')
result = predict_single(model, "[TARGET_USER] I need help", "no_intent", 
                        intent2idx, idx2intent, idx2dialogue_act)
print(result['intent'])

# 2. Conversation tracking
from inference import ConversationInference

conv = ConversationInference(model, intent2idx, idx2intent, idx2dialogue_act)
result1 = conv.predict_turn("[TARGET_USER] Can't log in")
result2 = conv.predict_turn("[TARGET_USER] Says invalid password")
# Automatically uses previous intent!

# 3. Batch prediction from CSV
df = predict_from_csv(model, 'test.csv', intent2idx, idx2intent, idx2dialogue_act)
df.to_csv('predictions.csv', index=False)
```

Save this as `inference.py` and run it! 🚀
