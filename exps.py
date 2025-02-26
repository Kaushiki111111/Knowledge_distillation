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
import time
from typing import List, Dict, Tuple, Optional, Union

# --------------------------------
# Configuration
# --------------------------------
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
        self.test_data_path = "test_data.txt"  # Path to your test data

# --------------------------------
# Dataset class
# --------------------------------
class TextDataset(Dataset):
    def __init__(self, tokenizer, data_path, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load and tokenize the data
        with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
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

# --------------------------------
# Single-layer Transformer Decoder (Student Model)
# --------------------------------
'''
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
        
        key_padding_mask = None
        if attention_mask is not None:
        # Invert attention_mask to match key_padding_mask format (1 = masked position)
            key_padding_mask = (1 - attention_mask).bool()
    
        hidden_states = self.decoder_layer(hidden_states, None,  # No memory for decoder-only model
        tgt_mask=causal_mask,
        tgt_key_padding_mask=key_padding_mask
        )
      
        
        
        # Get logits
        logits = self.output_layer(hidden_states)
        
        return logits

# --------------------------------
# Knowledge Distillation Loss
# --------------------------------
class SingleLayerTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(0.1)
        )
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
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
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal attention mask (lower triangular)
        attn_mask = torch.ones(seq_length, seq_length, device=input_ids.device)
        attn_mask = torch.tril(attn_mask).view(1, seq_length, seq_length)
        
        # Incorporate padding mask if attention_mask is provided
        if attention_mask is not None:
            # Expand attention_mask to match attn_mask
            extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, seq_length, -1)
            # Combine causal mask with padding mask (multiply to keep 0s in either mask)
            attn_mask = attn_mask * extended_attention_mask
        
        # Self-attention
        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        # Residual connection and layer norm
        hidden_states = hidden_states + attn_output
        hidden_states = self.layer_norm2(hidden_states)
        
        # Feed-forward network
        ff_output = self.ff_network(hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + ff_output
        
        # Get logits
        logits = self.output_layer(hidden_states)
        
        return logits
'''    

class SingleLayerTransformerDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embeddings
        self.position_embeddings = nn.Embedding(config.max_length, config.hidden_size)
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            batch_first=True
        )
        
        # Feed-forward network
        self.ff_network = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(0.1)
        )
        
        # Layer norms
        self.layer_norm1 = nn.LayerNorm(config.hidden_size)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size)
        
        # Output layer
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)
        
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
    
    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_length = input_ids.size()
        
        # Create position ids
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        
        # Combine embeddings
        hidden_states = token_embeds + position_embeds
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Create causal mask
        causal_mask = self.generate_square_subsequent_mask(seq_length).to(input_ids.device)
        
        # Handle padding mask
        key_padding_mask = None
        if attention_mask is not None:
            # Invert attention_mask because in PyTorch, 1 = position to be masked
            key_padding_mask = (1 - attention_mask).bool()
        
        # Self-attention
        attn_output, _ = self.self_attn(
            query=hidden_states,
            key=hidden_states,
            value=hidden_states,
            attn_mask=causal_mask,  # Use the causal mask
            key_padding_mask=key_padding_mask,  # Use padding mask if provided
            need_weights=False
        )
        
        # Residual connection and layer norm
        hidden_states = hidden_states + attn_output
        hidden_states = self.layer_norm2(hidden_states)
        
        # Feed-forward network
        ff_output = self.ff_network(hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + ff_output
        
        # Get logits
        logits = self.output_layer(hidden_states)
        
        return logits

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

# --------------------------------
# Model Loading Functions
# --------------------------------
def load_teacher_model(config):
    """Load the teacher model (GPT2 or Llama2)"""
    if "gpt2" in config.teacher_model_name:
        tokenizer = GPT2Tokenizer.from_pretrained(config.teacher_model_name)
        # Fix the padding token issue
        tokenizer.pad_token = tokenizer.eos_token
        model = GPT2LMHeadModel.from_pretrained(config.teacher_model_name)
        # Set pad token id in the model config as well
        model.config.pad_token_id = model.config.eos_token_id
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

# --------------------------------
# Training Functions
# --------------------------------
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

# --------------------------------
# Text Generation Functions
# --------------------------------
def generate_text(model, tokenizer, prompt, max_length=50, temperature=1.0, config=None):
    model.eval()
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(config.device)
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

# --------------------------------
# Evaluation Functions
# --------------------------------
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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compare_models(student_model, teacher_model, tokenizer, prompts, config):
    """Compare text generation between student and teacher models"""
    print("\nComparing text generation between student and teacher models:")
    
    for i, prompt in enumerate(prompts):
        print(f"\nPrompt {i+1}: {prompt}")
        
        # Time and generate with student model
        start_time = time.time()
        student_text = generate_text(student_model, tokenizer, prompt, max_length=50, config=config)
        student_time = time.time() - start_time
        
        # Time and generate with teacher model
        start_time = time.time()
        inputs = tokenizer(prompt, return_tensors="pt").to(config.device)
        teacher_output = teacher_model.generate(
            inputs["input_ids"], 
            max_length=len(inputs["input_ids"][0]) + 50,
            temperature=1.0,
            do_sample=True
        )
        teacher_time = time.time() - start_time
        teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True)
        
        # Print results
        print(f"Student ({student_time:.2f}s): {student_text}")
        print(f"Teacher ({teacher_time:.2f}s): {teacher_text}")
        print(f"Generation speedup: {teacher_time/student_time:.2f}x")
    
    # Count parameters
    student_params = count_parameters(student_model)
    teacher_params = count_parameters(teacher_model)
    print(f"\nModel size comparison:")
    print(f"Student parameters: {student_params:,}")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x")

# --------------------------------
# Main Functions
# --------------------------------
def main():
    parser = argparse.ArgumentParser(description="Knowledge Distillation from LLM to Single-Layer Transformer")
    
    # Add mode argument
    parser.add_argument("--mode", choices=["train", "evaluate", "generate"], default="train",
                       help="Mode: train, evaluate, or generate text")
    
    # Add data arguments
    parser.add_argument("--data", default="synthetic_data.txt",
                       help="Path to training data file")
    parser.add_argument("--test_data", default=None,
                       help="Path to test data file (if not specified, uses training data)")
    
    # Add model arguments
    parser.add_argument("--teacher", default="gpt2",
                       help="Teacher model name (gpt2, gpt2-medium, etc.)")
    parser.add_argument("--model_path", default="student_model.pt",
                       help="Path to saved student model (for evaluation or generation)")
    parser.add_argument("--save_path", default="student_model.pt",
                       help="Path to save the student model (for training)")
    
    # Add training arguments
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training and evaluation")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--max_len", type=int, default=128,
                       help="Maximum sequence length")
    parser.add_argument("--temp", type=float, default=2.0,
                       help="Temperature for distillation")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Weight for KL divergence loss")
    
    # Add generation arguments
    parser.add_argument("--prompt", default="Once upon a time",
                       help="Prompt for text generation")
    parser.add_argument("--gen_max_len", type=int, default=100,
                       help="Maximum length for generated text")
    
    args = parser.parse_args()
    
    # Set test data to training data if not specified
    if args.test_data is None:
        args.test_data = args.data
    
    # Initialize config
    config = Config()
    config.teacher_model_name = args.teacher
    config.data_path = args.data
    config.test_data_path = args.test_data
    config.save_path = args.save_path
    config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.temperature = args.temp
    config.alpha = args.alpha
    config.max_length = args.max_len
    
    # Run in the specified mode
    if args.mode == "train":
        print("Starting knowledge distillation training...")
        student_model = train_with_knowledge_distillation(config)
        
        # Optionally evaluate after training
        teacher_model, tokenizer = load_teacher_model(config)
        print("\nEvaluating trained model:")
        evaluate_model(student_model, tokenizer, config.test_data_path, config)
        
    elif args.mode == "evaluate":
        print(f"Evaluating model from {args.model_path}...")
        
        # Load teacher and tokenizer
        teacher_model, tokenizer = load_teacher_model(config)
        
        # Load student model
        student_model = SingleLayerTransformerDecoder(config)
        student_model.load_state_dict(torch.load(args.model_path))
        student_model.to(config.device)
        
        # Evaluate
        student_loss, student_ppl = evaluate_model(student_model, tokenizer, config.test_data_path, config)
        teacher_loss, teacher_ppl = evaluate_model(teacher_model, tokenizer, config.test_data_path, config)
        
        print(f"\nPerplexity comparison:")
        print(f"Student model perplexity: {student_ppl:.2f}")
        print(f"Teacher model perplexity: {teacher_ppl:.2f}")
        print(f"Relative performance: {(student_ppl/teacher_ppl):.2f}x worse than teacher")
        
        # Compare text generation
        prompts = [
            "Once upon a time",
            "The future of artificial intelligence",
            "In recent studies, scientists have discovered",
            "The best way to learn is",
            "When I consider the meaning of life"
        ]
        
        compare_models(student_model, teacher_model, tokenizer, prompts, config)
        
    elif args.mode == "generate":
        print(f"Generating text using model from {args.model_path}...")
        
        # Load teacher and tokenizer
        teacher_model, tokenizer = load_teacher_model(config)
        
        # Load student model
        student_model = SingleLayerTransformerDecoder(config)
        student_model.load_state_dict(torch.load(args.model_path))
        student_model.to(config.device)
        
        # Generate text
        print(f"Prompt: {args.prompt}")
        student_text = generate_text(student_model, tokenizer, args.prompt, max_length=args.gen_max_len, config=config)
        print(f"Student generated: {student_text}")
        
        # Generate with teacher for comparison
        inputs = tokenizer(args.prompt, return_tensors="pt").to(config.device)
        teacher_output = teacher_model.generate(
            inputs["input_ids"], 
            max_length=len(inputs["input_ids"][0]) + args.gen_max_len,
            temperature=1.0,
            do_sample=True
        )
        teacher_text = tokenizer.decode(teacher_output[0], skip_special_tokens=True)
        print(f"Teacher generated: {teacher_text}")

if __name__ == "__main__":
    main()