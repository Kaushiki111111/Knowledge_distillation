import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, LlamaTokenizer, LlamaForCausalLM
import numpy as np
from tqdm import tqdm
import argparse
import os
from typing import List, Dict, Tuple, Optional, Union

# Configuration
class Config:
    def __init__(self):
        self.teacher_model_name = "gpt2"  # or "meta-llama/Llama-2-7b-hf" for Llama2
        self.vocab_size = 50257  # GPT2 vocab size (will be updated based on teacher model)
        self.max_length = 128
        self.hidden_size = 768  # Size of embeddings
        self.num_heads = 12  # Number of attention heads for the student
        self.batch_size = 32
        self.epochs = 5
        self.learning_rate = 3e-4
        self.temperature = 2.0  # Softens probability distributions for knowledge distillation
        self.alpha = 0.5  # Weight for KL divergence loss vs task-specific loss
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.save_path = "student_model.pt"
        self.data_path = "data.txt"  # Path to your text data for training

# Dataset class
class TextDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize the data
        with open(data_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Split text into chunks of max_length for training
        self.examples = []
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length * 2]  # Overlap to ensure we have enough context
            if len(chunk) > max_length:
                self.examples.append(chunk)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        encodings = self.tokenizer(chunk, truncation=True, max_length=self.max_length, 
                                   padding="max_length", return_tensors="pt")
        
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        
        # For causal language modeling, labels are the same as inputs
        labels = input_ids.clone()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

# Single-layer Transformer Decoder (Student Model)
class SingleLayerTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        
        # Single transformer decoder layer
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, input_ids, attention_mask=None):
        # Get sequence length
        seq_length = input_ids.size(1)
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal mask for decoder
        sz = seq_length
        causal_mask = (torch.triu(torch.ones(sz, sz, device=input_ids.device)) == 1).transpose(0, 1)
        causal_mask = causal_mask.float().masked_fill(causal_mask == 0, float('-inf')).masked_fill(causal_mask == 1, float(0.0))
        
        # If attention_mask is provided, combine with causal mask
        if attention_mask is not None:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            causal_mask = causal_mask.unsqueeze(0) + extended_attention_mask
        
        # Pass through decoder layer
        # The memory parameter is None because we're not doing cross-attention in a decoder-only model
        tgt_mask = causal_mask.squeeze(0) if attention_mask is not None else causal_mask
        hidden_states = self.decoder_layer(hidden_states, None, tgt_mask=tgt_mask)
        
        # Get logits
        logits = self.output_layer(hidden_states)
        
        return logits

# Knowledge Distillation Loss
def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    # KL divergence loss
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl_div = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
    
    # Task-specific loss (standard cross-entropy)
    ce_loss = F.cross_entropy(student_logits.view(-1, student_logits.size(-1)), labels.view(-1), ignore_index=-100)
    
    # Combine losses
    total_loss = (alpha * kl_div) + ((1.0 - alpha) * ce_loss)
    
    return total_loss, kl_div, ce_loss

def load_teacher_model(config):
    """Load the teacher model (GPT2 or Llama2)"""
    if "gpt2" in config.teacher_model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
        model = GPT2LMHeadModel.from_pretrained(config.teacher_model_name)
        config.vocab_size = model.config.vocab_size
    elif "llama" in config.teacher_model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(config.teacher_model_name)
        model = LlamaForCausalLM.from_pretrained(config.teacher_model_name)
        config.vocab_size = model.config.vocab_size
    else:
        raise ValueError(f"Unsupported model: {config.teacher_model_name}")
    
    model = model.to(config.device)
    model.eval()  # Set to evaluation mode
    return model, tokenizer

def train_with_knowledge_distillation(config):
    """Main training function for knowledge distillation"""
    # Load the teacher model
    teacher_model, tokenizer = load_teacher_model(config)
    print(f"Teacher model loaded: {config.teacher_model_name}")
    
    # Initialize the student model
    student_model = SingleLayerTransformerDecoder(config).to(config.device)
    print("Student model initialized")
    
    # Prepare dataset
    train_dataset = TextDataset(tokenizer, config.data_path, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Dataset loaded with {len(train_dataset)} examples")
    
    # Optimizer
    optimizer = optim.AdamW(student_model.parameters(), lr=config.learning_rate)
    
    # Training loop
    print("Starting training...")
    for epoch in range(config.epochs):
        student_model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass through student model
            student_logits = student_model(input_ids, attention_mask)
            
            # Get teacher's predictions
            with torch.no_grad():
                if "gpt2" in config.teacher_model_name:
                    teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                else:  # Llama model
                    teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask)
                teacher_logits = teacher_outputs.logits
            
            # Calculate distillation loss
            loss, kl_loss, ce_loss = distillation_loss(
                student_logits, 
                teacher_logits, 
                labels, 
                config.temperature, 
                config.alpha
            )
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Update progress bar
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
        
        # Save the model after each epoch
        torch.save(student_model.state_dict(), config.save_path)
        print(f"Epoch {epoch+1} completed. Model saved to {config.save_path}")
    
    print("Training completed!")
    return student_model

# Generate text with the student model
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, config=None):
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.config.device)
    attention_mask = torch.ones_like(input_ids)
    
    # Generate text auto-regressively
    generated = input_ids.clone()
    
    for _ in range(max_length):
        # Get predictions
        with torch.no_grad():
            outputs = model(generated, attention_mask=attention_mask)
            next_token_logits = outputs[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode the generated text
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return generated_text

# Evaluate the model on a test set
def evaluate_model(model, tokenizer, test_data_path, config):
    model.eval()
    
    # Prepare test dataset
    test_dataset = TextDataset(tokenizer, test_data_path, config.max_length)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)
            labels = batch["labels"].to(config.device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            
            # Calculate loss
            loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1), ignore_index=-100)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader)
    perplexity = np.exp(avg_loss)
    
    print(f"Evaluation results: Loss = {avg_loss:.4f}, Perplexity = {perplexity:.4f}")
    return avg_loss, perplexity

# Main function to run everything
def main():
    # Setup configuration
    config = Config()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Knowledge Distillation from LLM to Single-Layer Transformer")
    parser.add_argument("--teacher", default=config.teacher_model_name, help="Teacher model name")
    parser.add_argument("--data", default=config.data_path, help="Path to training data")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="Batch size")
    parser.add_argument("--epochs", type=int, default=config.epochs, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=config.learning_rate, help="Learning rate")
    parser.add_argument("--temp", type=float, default=config.temperature, help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=config.alpha, help="Weight for KL divergence loss")
    parser.add_argument("--max_len", type=int, default=config.max_length, help="Maximum sequence length")
    parser.add_argument("--save_path", default=config.save_path, help="Path to save the student model")
    args = parser.parse_args()
    
    # Update config from args
    config.teacher_model_name = args.teacher
    config.data_path = args.data
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.temperature = args.temp
    config.alpha = args.alpha
    config.max_length = args.max_len
    config.save_path = args.save_path
    
    print(f"Using device: {config.device}")
    
    # Training
    student_model = train_with_knowledge_distillation(config)
    
    # Load the teacher model for testing
    teacher_model, tokenizer = load_teacher_model(config)
    
    # Example of text generation with student model
    prompt = "Once upon a time"
    student_text = generate_text(student_model, tokenizer, prompt, max_length=50, config=config)
    print(f"Student generated text: {student_text}")
    
    # Generate with teacher for comparison
    teacher_model.to(config.device)
    inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
    teacher_output = teacher_model.generate(
        inputs["input_ids"], 
        max_length=len(inputs["input_ids"][0]) + 50,
        temperature=1.0,
        do_sample=True
    )
    teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True)
    print(f"Teacher generated text: {teacher_text}")

if __name__ == "__main__":
    main()