"""
Intent Classifier using SentenceTransformers (cleaner API).
Uses any SentenceTransformer model with classification heads.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import numpy as np
import json

# ============================================================================
# 1. FOCAL LOSS
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        probs = torch.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


# ============================================================================
# 2. DATASET CLASS
# ============================================================================

class SentenceTransformerIntentDataset(Dataset):
    """
    Dataset for SentenceTransformers.
    Returns text strings (no tokenization needed - handled by model).
    """
    
    def __init__(self, data, intent2idx, dialogue_act2idx):
        self.data = data
        self.intent2idx = intent2idx
        self.dialogue_act2idx = dialogue_act2idx
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Return raw text (SentenceTransformers handles tokenization)
        text = item['text']
        
        # Convert labels to indices
        prev_intent_idx = self.intent2idx[item['previous_intent']]
        intent_label_idx = self.intent2idx[item['labels']['intent']]
        boundary_label = 1 if item['labels']['is_boundary'] else 0
        dialogue_act_idx = self.dialogue_act2idx[item['labels']['dialogue_act']]
        
        return {
            'text': text,  # Raw string
            'prev_intent': torch.tensor(prev_intent_idx, dtype=torch.long),
            'intent_label': torch.tensor(intent_label_idx, dtype=torch.long),
            'boundary_label': torch.tensor(boundary_label, dtype=torch.long),
            'dialogue_act_label': torch.tensor(dialogue_act_idx, dtype=torch.long)
        }


# ============================================================================
# 3. CUSTOM COLLATE FUNCTION
# ============================================================================

def collate_fn(batch):
    """
    Custom collate function to handle text strings.
    Batches text separately from tensors.
    """
    texts = [item['text'] for item in batch]
    
    return {
        'texts': texts,  # List of strings
        'prev_intent': torch.stack([item['prev_intent'] for item in batch]),
        'intent_label': torch.stack([item['intent_label'] for item in batch]),
        'boundary_label': torch.stack([item['boundary_label'] for item in batch]),
        'dialogue_act_label': torch.stack([item['dialogue_act_label'] for item in batch])
    }


# ============================================================================
# 4. MODEL CLASS WITH SENTENCETRANSFORMERS
# ============================================================================

class SentenceTransformerIntentClassifier(nn.Module):
    """
    Intent classifier using SentenceTransformers.
    
    Supports any SentenceTransformer model:
    - sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast)
    - sentence-transformers/all-mpnet-base-v2 (768-dim, better)
    - sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim, multilingual)
    - Alibaba-NLP/gte-large-en-v1.5 (1024-dim, SOTA)
    """
    
    def __init__(
        self, 
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        num_intents: int = 20,
        num_dialogue_acts: int = 5,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load SentenceTransformer model
        self.encoder = SentenceTransformer(model_name)
        
        # Get embedding dimension
        encoder_dim = self.encoder.get_sentence_embedding_dimension()
        print(f"Encoder embedding dimension: {encoder_dim}")
        
        # Previous intent embedding
        self.prev_intent_embedding = nn.Embedding(num_intents, 64)
        
        # Combined dimension
        combined_dim = encoder_dim + 64
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
        # Task-specific heads
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
        
        print(f"Model initialized:")
        print(f"  Encoder: {model_name}")
        print(f"  Embedding dim: {encoder_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Intent classes: {num_intents}")
    
    def forward(self, texts, prev_intent):
        """
        Forward pass.
        
        Args:
            texts: List of strings (batch of text)
            prev_intent: Tensor [batch_size]
        
        Returns:
            Dict with logits for each task
        """
        # Encode texts (returns embeddings)
        text_embeddings = self.encoder.encode(
            texts,
            convert_to_tensor=True,
            show_progress_bar=False
        )  # [batch_size, encoder_dim]
        
        # Get previous intent embeddings
        prev_intent_emb = self.prev_intent_embedding(prev_intent)
        
        # Combine
        combined = torch.cat([text_embeddings, prev_intent_emb], dim=-1)
        
        # Shared representation
        shared_repr = self.shared_layer(combined)
        
        # Task-specific predictions
        intent_logits = self.intent_classifier(shared_repr)
        boundary_logits = self.boundary_detector(shared_repr)
        dialogue_act_logits = self.dialogue_act_classifier(shared_repr)
        
        return {
            'intent_logits': intent_logits,
            'boundary_logits': boundary_logits,
            'dialogue_act_logits': dialogue_act_logits
        }


# ============================================================================
# 5. TRAINING SCRIPT
# ============================================================================

def train_sentencetransformer_classifier():
    """Complete training pipeline."""
    
    # Configuration
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
    
    # Load data
    print("Loading data...")
    with open('annotated_data.json', 'r') as f:
        data = json.load(f)
    
    print(f"Total examples: {len(data)}")
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Initialize model
    print("\nInitializing SentenceTransformer model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = SentenceTransformerIntentClassifier(
        model_name="sentence-transformers/all-MiniLM-L6-v2",  # Fast, 384-dim
        # model_name="sentence-transformers/all-mpnet-base-v2",  # Better, 768-dim
        # model_name="Alibaba-NLP/gte-large-en-v1.5",  # SOTA, 1024-dim
        num_intents=len(INTENT_LABELS),
        num_dialogue_acts=len(DIALOGUE_ACT_LABELS),
        hidden_dim=512
    ).to(device)
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = SentenceTransformerIntentDataset(train_data, intent2idx, dialogue_act2idx)
    val_dataset = SentenceTransformerIntentDataset(val_data, intent2idx, dialogue_act2idx)
    
    # DataLoaders with custom collate function
    train_loader = DataLoader(
        train_dataset, 
        batch_size=16, 
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Compute class weights
    print("\nComputing class weights...")
    intent_labels = [intent2idx[item['labels']['intent']] for item in train_data]
    intent_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(intent_labels),
        y=intent_labels
    )
    intent_weights = torch.FloatTensor(intent_weights).to(device)
    
    dialogue_act_labels = [dialogue_act2idx[item['labels']['dialogue_act']] for item in train_data]
    dialogue_act_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(dialogue_act_labels),
        y=dialogue_act_labels
    )
    dialogue_act_weights = torch.FloatTensor(dialogue_act_weights).to(device)
    
    # Loss functions
    intent_criterion = FocalLoss(alpha=intent_weights, gamma=2.0)
    boundary_criterion = nn.CrossEntropyLoss()
    dialogue_act_criterion = FocalLoss(alpha=dialogue_act_weights, gamma=2.0)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # Learning rate scheduler
    from transformers import get_linear_schedule_with_warmup
    num_training_steps = len(train_loader) * 5
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Training loop
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
            # Get texts and move labels to device
            texts = batch['texts']  # List of strings
            prev_intent = batch['prev_intent'].to(device)
            intent_label = batch['intent_label'].to(device)
            boundary_label = batch['boundary_label'].to(device)
            dialogue_act_label = batch['dialogue_act_label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass (texts stay as strings)
            outputs = model(texts, prev_intent)
            
            # Calculate losses
            intent_loss = intent_criterion(outputs['intent_logits'], intent_label)
            boundary_loss = boundary_criterion(outputs['boundary_logits'], boundary_label)
            dialogue_act_loss = dialogue_act_criterion(outputs['dialogue_act_logits'], dialogue_act_label)
            
            total_loss = intent_loss + 0.5 * boundary_loss + 0.3 * dialogue_act_loss
            
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += total_loss.item()
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, "
                      f"Loss: {total_loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_intent_preds = []
        all_intent_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                texts = batch['texts']
                prev_intent = batch['prev_intent'].to(device)
                intent_label = batch['intent_label'].to(device)
                boundary_label = batch['boundary_label'].to(device)
                dialogue_act_label = batch['dialogue_act_label'].to(device)
                
                outputs = model(texts, prev_intent)
                
                intent_loss = intent_criterion(outputs['intent_logits'], intent_label)
                boundary_loss = boundary_criterion(outputs['boundary_logits'], boundary_label)
                dialogue_act_loss = dialogue_act_criterion(outputs['dialogue_act_logits'], dialogue_act_label)
                
                total_loss = intent_loss + 0.5 * boundary_loss + 0.3 * dialogue_act_loss
                val_loss += total_loss.item()
                
                intent_preds = outputs['intent_logits'].argmax(dim=-1)
                all_intent_preds.extend(intent_preds.cpu().numpy())
                all_intent_labels.extend(intent_label.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader)
        val_f1 = f1_score(all_intent_labels, all_intent_preds, average='macro')
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Val Macro F1: {val_f1:.4f}")
        
        if (epoch + 1) % 2 == 0:
            print("\nClassification Report:")
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
                'val_f1': val_f1,
            }, 'best_st_intent_model.pt')
            print(f"  ✅ Saved best model (F1: {val_f1:.4f})")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print(f"Best Validation F1: {best_val_f1:.4f}")
    print("=" * 80)


# ============================================================================
# 6. INFERENCE
# ============================================================================

def predict_with_sentencetransformer():
    """Load model and make predictions."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SentenceTransformerIntentClassifier(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_intents=15,
        num_dialogue_acts=5
    ).to(device)
    
    checkpoint = torch.load('best_st_intent_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Example prediction
    text = "[USER] I can't log in [AGENT] What error? [TARGET_USER] Invalid password"
    prev_intent_idx = 0  # no_intent
    
    prev_intent = torch.tensor([prev_intent_idx]).to(device)
    
    with torch.no_grad():
        outputs = model([text], prev_intent)  # texts as list
    
    intent_pred = outputs['intent_logits'].argmax(dim=-1).item()
    boundary_pred = outputs['boundary_logits'].argmax(dim=-1).item()
    
    print(f"Intent: {intent_pred}")
    print(f"Boundary: {boundary_pred}")


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    train_sentencetransformer_classifier()
