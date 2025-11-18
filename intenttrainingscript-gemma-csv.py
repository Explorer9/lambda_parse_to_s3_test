"""
Training script for CSV data with train/val/test split.
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
import pandas as pd

# ============================================================================
# 1. FOCAL LOSS (Same as before)
# ============================================================================

class FocalLoss(nn.Module):
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
# 2. DATASET CLASS FOR CSV
# ============================================================================

class CSVIntentDataset(Dataset):
    """
    Dataset for CSV with columns: text, dialogueact, isboundary, current_intent, previous_intent
    """
    
    def __init__(self, df, intent2idx, dialogue_act2idx):
        self.df = df.reset_index(drop=True)
        self.intent2idx = intent2idx
        self.dialogue_act2idx = dialogue_act2idx
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Get values from CSV
        text = str(row['text'])
        dialogue_act = str(row['dialogueact'])
        is_boundary = row['isboundary']  # Should be True/False or 1/0
        current_intent = str(row['current_intent'])
        previous_intent = str(row['previous_intent'])
        
        # Convert to indices
        prev_intent_idx = self.intent2idx[previous_intent]
        intent_label_idx = self.intent2idx[current_intent]
        
        # Handle boundary (convert to 0/1 if needed)
        if isinstance(is_boundary, bool):
            boundary_label = 1 if is_boundary else 0
        elif isinstance(is_boundary, str):
            boundary_label = 1 if is_boundary.lower() in ['true', '1', 'yes'] else 0
        else:
            boundary_label = int(is_boundary)
        
        dialogue_act_idx = self.dialogue_act2idx[dialogue_act]
        
        return {
            'text': text,
            'prev_intent': torch.tensor(prev_intent_idx, dtype=torch.long),
            'intent_label': torch.tensor(intent_label_idx, dtype=torch.long),
            'boundary_label': torch.tensor(boundary_label, dtype=torch.long),
            'dialogue_act_label': torch.tensor(dialogue_act_idx, dtype=torch.long)
        }


# ============================================================================
# 3. CUSTOM COLLATE FUNCTION
# ============================================================================

def collate_fn(batch):
    texts = [item['text'] for item in batch]
    
    return {
        'texts': texts,
        'prev_intent': torch.stack([item['prev_intent'] for item in batch]),
        'intent_label': torch.stack([item['intent_label'] for item in batch]),
        'boundary_label': torch.stack([item['boundary_label'] for item in batch]),
        'dialogue_act_label': torch.stack([item['dialogue_act_label'] for item in batch])
    }


# ============================================================================
# 4. MODEL CLASS (Same as before)
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
        
        print(f"Model initialized with embedding dim: {encoder_dim}")
    
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
# 5. TRAINING SCRIPT FOR CSV
# ============================================================================

def train_from_csv(csv_path='training_data.csv'):
    """
    Train model from CSV file.
    
    CSV format:
    text,dialogueact,isboundary,current_intent,previous_intent
    "[USER] Hi [AGENT] Hello",question,True,login_issue,no_intent
    ...
    """
    
    print("=" * 80)
    print("LOADING DATA FROM CSV")
    print("=" * 80)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} examples from {csv_path}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Get unique labels
    unique_intents = sorted(df['current_intent'].unique().tolist())
    unique_dialogue_acts = sorted(df['dialogueact'].unique().tolist())
    
    print(f"\n\nUnique intents ({len(unique_intents)}):")
    for intent in unique_intents:
        count = (df['current_intent'] == intent).sum()
        print(f"  {intent:30s}: {count:4d} examples")
    
    print(f"\nUnique dialogue acts ({len(unique_dialogue_acts)}):")
    for act in unique_dialogue_acts:
        count = (df['dialogueact'] == act).sum()
        print(f"  {act:30s}: {count:4d} examples")
    
    # Create label mappings
    intent2idx = {label: idx for idx, label in enumerate(unique_intents)}
    dialogue_act2idx = {label: idx for idx, label in enumerate(unique_dialogue_acts)}
    
    # Split data: 70% train, 15% val, 15% test
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df['current_intent'])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df['current_intent'])
    
    print(f"\n\nData split:")
    print(f"  Train: {len(train_df)} examples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} examples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} examples ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print("\nSaved splits to: train.csv, val.csv, test.csv")
    
    # Initialize model
    print("\n" + "=" * 80)
    print("INITIALIZING MODEL")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = SentenceTransformerIntentClassifier(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        num_intents=len(unique_intents),
        num_dialogue_acts=len(unique_dialogue_acts),
        hidden_dim=512
    ).to(device)
    
    # Create datasets
    train_dataset = CSVIntentDataset(train_df, intent2idx, dialogue_act2idx)
    val_dataset = CSVIntentDataset(val_df, intent2idx, dialogue_act2idx)
    test_dataset = CSVIntentDataset(test_df, intent2idx, dialogue_act2idx)
    
    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # Compute class weights
    print("\nComputing class weights...")
    intent_labels = [intent2idx[intent] for intent in train_df['current_intent']]
    intent_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(intent_labels),
        y=intent_labels
    )
    intent_weights = torch.FloatTensor(intent_weights).to(device)
    
    dialogue_act_labels = [dialogue_act2idx[act] for act in train_df['dialogueact']]
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
    
    # Scheduler
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
            texts = batch['texts']
            prev_intent = batch['prev_intent'].to(device)
            intent_label = batch['intent_label'].to(device)
            boundary_label = batch['boundary_label'].to(device)
            dialogue_act_label = batch['dialogue_act_label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(texts, prev_intent)
            
            intent_loss = intent_criterion(outputs['intent_logits'], intent_label)
            boundary_loss = boundary_criterion(outputs['boundary_logits'], boundary_label)
            dialogue_act_loss = dialogue_act_criterion(outputs['dialogue_act_logits'], dialogue_act_label)
            
            total_loss = intent_loss + 0.5 * boundary_loss + 0.3 * dialogue_act_loss
            
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
            print("\nValidation Classification Report:")
            report = classification_report(
                all_intent_labels, 
                all_intent_preds, 
                target_names=unique_intents,
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
                'intent2idx': intent2idx,
                'dialogue_act2idx': dialogue_act2idx,
                'unique_intents': unique_intents,
                'unique_dialogue_acts': unique_dialogue_acts
            }, 'best_model.pt')
            print(f"  ✅ Saved best model (F1: {val_f1:.4f})")
        
        print("-" * 80)
    
    # Final test evaluation
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)
    
    model.eval()
    all_intent_preds = []
    all_intent_labels = []
    all_boundary_preds = []
    all_boundary_labels = []
    all_dialogue_act_preds = []
    all_dialogue_act_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['texts']
            prev_intent = batch['prev_intent'].to(device)
            intent_label = batch['intent_label'].to(device)
            boundary_label = batch['boundary_label'].to(device)
            dialogue_act_label = batch['dialogue_act_label'].to(device)
            
            outputs = model(texts, prev_intent)
            
            intent_preds = outputs['intent_logits'].argmax(dim=-1)
            boundary_preds = outputs['boundary_logits'].argmax(dim=-1)
            dialogue_act_preds = outputs['dialogue_act_logits'].argmax(dim=-1)
            
            all_intent_preds.extend(intent_preds.cpu().numpy())
            all_intent_labels.extend(intent_label.cpu().numpy())
            all_boundary_preds.extend(boundary_preds.cpu().numpy())
            all_boundary_labels.extend(boundary_label.cpu().numpy())
            all_dialogue_act_preds.extend(dialogue_act_preds.cpu().numpy())
            all_dialogue_act_labels.extend(dialogue_act_label.cpu().numpy())
    
    # Intent results
    test_intent_f1 = f1_score(all_intent_labels, all_intent_preds, average='macro')
    print(f"\nTest Intent Macro F1: {test_intent_f1:.4f}")
    print("\nIntent Classification Report:")
    print(classification_report(
        all_intent_labels, 
        all_intent_preds, 
        target_names=unique_intents,
        zero_division=0
    ))
    
    # Boundary results
    test_boundary_f1 = f1_score(all_boundary_labels, all_boundary_preds, average='macro')
    print(f"\nTest Boundary Macro F1: {test_boundary_f1:.4f}")
    
    # Dialogue act results
    test_dialogue_f1 = f1_score(all_dialogue_act_labels, all_dialogue_act_preds, average='macro')
    print(f"\nTest Dialogue Act Macro F1: {test_dialogue_f1:.4f}")
    print("\nDialogue Act Classification Report:")
    print(classification_report(
        all_dialogue_act_labels, 
        all_dialogue_act_preds, 
        target_names=unique_dialogue_acts,
        zero_division=0
    ))
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"Best Val F1: {best_val_f1:.4f}")
    print(f"Test Intent F1: {test_intent_f1:.4f}")
    print(f"Test Boundary F1: {test_boundary_f1:.4f}")
    print(f"Test Dialogue Act F1: {test_dialogue_f1:.4f}")
    print("\nModel saved to: best_model.pt")
    print("Data splits saved to: train.csv, val.csv, test.csv")


# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    train_from_csv('training_data.csv')
