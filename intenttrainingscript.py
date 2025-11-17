"""
Complete training script with Focal Loss for imbalanced intent detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import json

# ============================================================================
# 1. FOCAL LOSS IMPLEMENTATION
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Focuses learning on hard/rare examples.
    """
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha: Class weights (tensor of shape [num_classes])
            gamma: Focusing parameter (2.0 is typical)
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
        """
        # Standard cross entropy
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        
        # Get probabilities
        probs = torch.softmax(inputs, dim=1)
        
        # Get probability of correct class
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal term: (1 - pt)^gamma
        # Easy examples (pt close to 1) get low weight
        # Hard examples (pt close to 0) get high weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ============================================================================
# 2. COMPUTE CLASS WEIGHTS FROM YOUR DATA
# ============================================================================

def compute_class_weights(data, intent2idx, dialogue_act2idx):
    """
    Compute class weights for intents and dialogue acts.
    """
    # Get intent labels
    intent_labels = [intent2idx[item['labels']['intent']] for item in data]
    
    # Compute intent weights
    intent_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(intent_labels),
        y=intent_labels
    )
    
    # Get dialogue act labels
    dialogue_act_labels = [dialogue_act2idx[item['labels']['dialogue_act']] for item in data]
    
    # Compute dialogue act weights
    dialogue_act_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(dialogue_act_labels),
        y=dialogue_act_labels
    )
    
    # Print distribution
    print("Intent Distribution and Weights:")
    from collections import Counter
    intent_counts = Counter(intent_labels)
    for idx in sorted(intent_counts.keys()):
        intent_name = [k for k, v in intent2idx.items() if v == idx][0]
        count = intent_counts[idx]
        weight = intent_weights[idx]
        print(f"  {intent_name:25s}: {count:4d} examples, weight: {weight:.3f}")
    
    return torch.FloatTensor(intent_weights), torch.FloatTensor(dialogue_act_weights)


# ============================================================================
# 3. COMPLETE TRAINING SCRIPT
# ============================================================================

# Load data
with open('annotated_data.json', 'r') as f:
    data = json.load(f)

# Define labels
INTENT_LABELS = [
    'no_intent', 'continuation', 'login_issue', 'fee_inquiry', 
    'balance_inquiry', 'card_request', 'transaction_dispute', 
    'transfer_funds', 'statement_request', 'account_info_update',
    'fraud_report', 'account_closure', 'technical_support', 
    'branch_location', 'bereavement'
]

DIALOGUE_ACT_LABELS = [
    'question', 'problem_statement', 'provide_information', 
    'acknowledgment', 'request'
]

intent2idx = {label: idx for idx, label in enumerate(INTENT_LABELS)}
dialogue_act2idx = {label: idx for idx, label in enumerate(DIALOGUE_ACT_LABELS)}

# Setup tokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenizer.add_special_tokens({'additional_special_tokens': ['[AGENT]', '[USER]', '[TARGET_USER]']})

# Create dataset
from your_dataset import IntentDataset  # Your dataset class

# Split data (80% train, 20% val)
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_dataset = IntentDataset(train_data, tokenizer, intent2idx, dialogue_act2idx)
val_dataset = IntentDataset(val_data, tokenizer, intent2idx, dialogue_act2idx)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Compute class weights
intent_weights, dialogue_act_weights = compute_class_weights(train_data, intent2idx, dialogue_act2idx)

# Initialize model
from your_model import ConversationIntentTracker
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ConversationIntentTracker(
    model_name="distilbert-base-uncased",
    num_intents=len(INTENT_LABELS),
    num_dialogue_acts=len(DIALOGUE_ACT_LABELS)
)
model.encoder.resize_token_embeddings(len(tokenizer))
model = model.to(device)

# ============================================================================
# KEY SECTION: INITIALIZE LOSS FUNCTIONS WITH FOCAL LOSS
# ============================================================================

# Move weights to device
intent_weights = intent_weights.to(device)
dialogue_act_weights = dialogue_act_weights.to(device)

# Create loss functions
intent_criterion = FocalLoss(alpha=intent_weights, gamma=2.0)  # Focal Loss for intents
boundary_criterion = nn.CrossEntropyLoss()  # Regular CE (usually balanced)
dialogue_act_criterion = FocalLoss(alpha=dialogue_act_weights, gamma=2.0)  # Focal Loss for dialogue acts

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

# Learning rate scheduler
from transformers import get_linear_schedule_with_warmup
num_training_steps = len(train_loader) * 5  # 5 epochs
num_warmup_steps = num_training_steps // 10

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

# ============================================================================
# TRAINING LOOP
# ============================================================================

print("\n" + "=" * 80)
print("STARTING TRAINING")
print("=" * 80)

best_val_f1 = 0.0
num_epochs = 5

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        prev_intent = batch['prev_intent'].to(device)
        
        intent_label = batch['intent_label'].to(device)
        boundary_label = batch['boundary_label'].to(device)
        dialogue_act_label = batch['dialogue_act_label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, prev_intent)
        
        # ============================================================================
        # KEY SECTION: COMPUTE LOSSES WITH FOCAL LOSS
        # ============================================================================
        intent_loss = intent_criterion(outputs['intent_logits'], intent_label)
        boundary_loss = boundary_criterion(outputs['boundary_logits'], boundary_label)
        dialogue_act_loss = dialogue_act_criterion(outputs['dialogue_act_logits'], dialogue_act_label)
        
        # Combined loss (weighted)
        total_loss = intent_loss + 0.5 * boundary_loss + 0.3 * dialogue_act_loss
        
        # Backward pass
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        train_loss += total_loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                  f"Loss: {total_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    all_intent_preds = []
    all_intent_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            prev_intent = batch['prev_intent'].to(device)
            
            intent_label = batch['intent_label'].to(device)
            boundary_label = batch['boundary_label'].to(device)
            dialogue_act_label = batch['dialogue_act_label'].to(device)
            
            outputs = model(input_ids, attention_mask, prev_intent)
            
            intent_loss = intent_criterion(outputs['intent_logits'], intent_label)
            boundary_loss = boundary_criterion(outputs['boundary_logits'], boundary_label)
            dialogue_act_loss = dialogue_act_criterion(outputs['dialogue_act_logits'], dialogue_act_label)
            
            total_loss = intent_loss + 0.5 * boundary_loss + 0.3 * dialogue_act_loss
            val_loss += total_loss.item()
            
            # Collect predictions
            intent_preds = outputs['intent_logits'].argmax(dim=-1)
            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_label.cpu().numpy())
    
    avg_val_loss = val_loss / len(val_loader)
    
    # Compute metrics
    from sklearn.metrics import f1_score
    val_f1 = f1_score(all_intent_labels, all_intent_preds, average='macro')
    
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Macro F1: {val_f1:.4f}")
    
    # Detailed classification report
    if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
        print("\nDetailed Classification Report:")
        report = classification_report(
            all_intent_labels, 
            all_intent_preds, 
            target_names=INTENT_LABELS,
            zero_division=0
        )
        print(report)
    
    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_f1': val_f1,
        }, 'best_intent_model.pt')
        print(f"  ✅ Saved best model (F1: {val_f1:.4f})")
    
    print("-" * 80)

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print(f"Best Validation F1: {best_val_f1:.4f}")
print("=" * 80)

# Save final model
torch.save(model.state_dict(), 'final_intent_model.pt')
tokenizer.save_pretrained('./intent_tokenizer')

print("\nModel saved to: final_intent_model.pt")
print("Tokenizer saved to: ./intent_tokenizer")
