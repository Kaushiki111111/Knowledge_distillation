import os
import argparse
import random
import requests
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import GPT2Tokenizer
import nltk
from nltk.tokenize import sent_tokenize
import zipfile
import io

def download_gutenberg_books(output_dir, num_books=10, min_size_kb=50):
    """
    Download books from Project Gutenberg for training data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the list of books from Project Gutenberg
    print("Fetching book list from Project Gutenberg...")
    try:
        # Get a list of book IDs
        book_ids = list(range(1, 1000))  # Use first 1000 books as candidates
        random.shuffle(book_ids)
        
        books_downloaded = 0
        for book_id in tqdm(book_ids):
            if books_downloaded >= num_books:
                break
                
            try:
                url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
                response = requests.get(url, timeout=10)
                
                # If we get a 404, try alternative URL
                if response.status_code == 404:
                    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
                    response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # Check if the book is large enough
                    if len(content.encode('utf-8')) >= min_size_kb * 1024:
                        output_file = os.path.join(output_dir, f"book_{book_id}.txt")
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        books_downloaded += 1
                        print(f"Downloaded book {book_id} ({len(content.encode('utf-8')) / 1024:.2f} KB)")
            except Exception as e:
                print(f"Error downloading book {book_id}: {e}")
                continue
        
        print(f"Successfully downloaded {books_downloaded} books to {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")

def download_wikipedia_articles(output_dir, num_articles=100, min_size_tokens=200):
    """
    Download Wikipedia articles for training data
    """
    try:
        nltk.download('punkt')
    except:
        print("NLTK punkt download failed, but may already be installed.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Wikipedia dataset...")
    #dataset = load_dataset("wikipedia", "20220301.en", split="train[:5000]")
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:5000]", trust_remote_code=True)
    
    # Filter and save articles
    articles_downloaded = 0
    
    for i, article in enumerate(tqdm(dataset)):
        if articles_downloaded >= num_articles:
            break
        
        text = article['text']
        title = article['title']
        
        # Clean filename
        safe_title = "".join([c if c.isalnum() else "_" for c in title])
        
        # Skip short articles
        if len(text.split()) < min_size_tokens:
            continue
        
        # Save the article
        output_file = os.path.join(output_dir, f"{safe_title[:50]}.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        articles_downloaded += 1
    
    print(f"Successfully downloaded {articles_downloaded} Wikipedia articles to {output_dir}")

def create_wikitext_dataset(output_dir, split="train", subset="wikitext-103-v1"):
    """
    Download and prepare WikiText dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading {subset} {split} split...")
    dataset = load_dataset("wikitext", subset, split=split)
    
    output_file = os.path.join(output_dir, f"wikitext_{split}.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            text = item['text']
            if text.strip():  # Skip empty lines
                f.write(text + "\n")
    
    print(f"Saved {subset} {split} data to {output_file}")
    return output_file

def download_c4_dataset(output_dir, num_examples=1000):
    """
    Download and prepare a subset of the C4 dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading C4 dataset (this might take some time)...")
    dataset = load_dataset("c4", "en", split=f"train[:{num_examples}]")
    
    output_file = os.path.join(output_dir, "c4_subset.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            text = item['text']
            if text.strip():  # Skip empty lines
                f.write(text + "\n\n")
    
    print(f"Saved C4 subset with {num_examples} examples to {output_file}")
    return output_file

def combine_files(input_dir, output_file, max_size_mb=None):
    """
    Combine all text files in a directory into a single file,
    optionally limiting the total size
    """
    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                if os.path.isfile(os.path.join(input_dir, f)) and f.endswith('.txt')]
    
    current_size = 0
    max_size_bytes = max_size_mb * 1024 * 1024 if max_size_mb else float('inf')
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for fname in tqdm(all_files, desc="Combining files"):
            with open(fname, 'r', encoding='utf-8', errors='ignore') as infile:
                content = infile.read()
                outfile.write(content)
                outfile.write('\n\n')
                
                current_size += len(content.encode('utf-8'))
                if current_size >= max_size_bytes:
                    print(f"Reached size limit of {max_size_mb} MB")
                    break
    
    print(f"Combined {len(all_files)} files into {output_file}")
    print(f"Total size: {current_size / (1024*1024):.2f} MB")

def analyze_text_file(file_path, tokenizer=None):
    """
    Analyze a text file and provide statistics
    """
    if tokenizer is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Basic statistics
    num_chars = len(text)
    num_words = len(text.split())
    
    # Calculate number of tokens
    tokens = tokenizer.encode(text[:1000000])  # Limit to first 1M chars to avoid OOM
    tokens_per_char = len(tokens) / min(num_chars, 1000000)
    estimated_tokens = int(tokens_per_char * num_chars)
    
    # Get number of lines
    num_lines = text.count('\n') + 1
    
    # Get file size
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    # Print statistics
    print(f"Analysis of {file_path}:")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Characters: {num_chars:,}")
    print(f"  Words: {num_words:,}")
    print(f"  Lines: {num_lines:,}")
    print(f"  Estimated tokens: {estimated_tokens:,}")
    print(f"  Tokens per word: {estimated_tokens / num_words:.2f}")
    
    # Sample text
    print("\nSample text:")
    sample_text = text[:1000] + "..." if len(text) > 1000 else text
    print(sample_text)
    
    return {
        "file_size_mb": file_size_mb,
        "characters": num_chars,
        "words": num_words,
        "lines": num_lines,
        "estimated_tokens": estimated_tokens
    }

def create_synthetic_data(output_file, num_examples=1000, min_length=50, max_length=200):
    """
    Create synthetic text data with various patterns for training
    """
    patterns = [
        "The {adj} {noun} {verb} the {adj} {noun}.",
        "In the {time} of {event}, the {person} {verb} {adverb}.",
        "When {person} {verb} the {noun}, {person2} {verb2} {adverb}.",
        "{person} said, '{exclamation} I {verb} {adverb} about the {adj} {noun}!'",
        "The {adj} {noun} is {adj2} than the {adj3} {noun2}.",
        "If you {verb} the {noun}, then you will {verb2} the {noun2}.",
        "{person} {verb} that the {noun} {verb2} {adverb} during the {event}.",
        "Despite the {adj} {noun}, {person} decided to {verb} the {noun2}.",
        "The {noun} {verb} {adverb}, causing the {noun2} to {verb2}.",
        "Before the {event}, {person} {verb} the {noun} and {verb2} the {noun2}."
    ]
    
    adjectives = ['happy', 'sad', 'bright', 'dark', 'large', 'small', 'quiet', 'loud', 'ancient', 'modern', 
                 'beautiful', 'ugly', 'peaceful', 'chaotic', 'mysterious', 'obvious', 'delicious', 'bitter']
    
    nouns = ['tree', 'river', 'mountain', 'city', 'book', 'child', 'dog', 'cat', 'house', 'car', 
            'computer', 'phone', 'sun', 'moon', 'rain', 'snow', 'forest', 'ocean', 'desert', 'garden']
    
    verbs = ['runs', 'jumps', 'walks', 'flies', 'swims', 'reads', 'writes', 'speaks', 'listens', 'eats',
            'drinks', 'sleeps', 'wakes', 'laughs', 'cries', 'sings', 'dances', 'thinks', 'watches', 'builds']
    
    adverbs = ['quickly', 'slowly', 'loudly', 'quietly', 'happily', 'sadly', 'carefully', 'carelessly',
              'patiently', 'impatiently', 'beautifully', 'poorly', 'perfectly', 'awkwardly']
    
    people = ['John', 'Mary', 'David', 'Sarah', 'Michael', 'Emma', 'James', 'Linda', 
             'Robert', 'Patricia', 'Daniel', 'Jennifer', 'Matthew', 'Elizabeth']
    
    times = ['morning', 'afternoon', 'evening', 'night', 'dawn', 'dusk', 'day', 'week', 
            'month', 'year', 'century', 'moment', 'era', 'age', 'season']
    
    events = ['celebration', 'disaster', 'war', 'peace', 'revolution', 'discovery', 'creation',
             'destruction', 'victory', 'defeat', 'coronation', 'wedding', 'funeral', 'birth', 'graduation']
    
    exclamations = ['Wow', 'Oh my', 'Goodness', 'Dear me', 'Alas', 'Hooray', 'Finally', 
                   'Indeed', 'Surprisingly', 'Unfortunately', 'Fortunately', 'Oddly enough']
    
    # Generate the text data
    with open(output_file, 'w', encoding='utf-8') as f:
        for _ in tqdm(range(num_examples), desc="Generating synthetic data"):
            # Determine paragraph length for this example
            paragraph_length = random.randint(min_length, max_length)
            
            paragraph = []
            current_length = 0
            
            while current_length < paragraph_length:
                pattern = random.choice(patterns)
                
                # Fill in the pattern
                sentence = pattern.format(
                    adj=random.choice(adjectives),
                    noun=random.choice(nouns),
                    verb=random.choice(verbs),
                    adverb=random.choice(adverbs),
                    person=random.choice(people),
                    person2=random.choice(people),
                    time=random.choice(times),
                    event=random.choice(events),
                    exclamation=random.choice(exclamations),
                    adj2=random.choice(adjectives),
                    adj3=random.choice(adjectives),
                    noun2=random.choice(nouns),
                    verb2=random.choice(verbs)
                )
                
                paragraph.append(sentence)
                current_length += len(sentence.split())
            
            f.write(' '.join(paragraph) + '\n\n')
    
    print(f"Created synthetic dataset with {num_examples} examples in {output_file}")

def download_openwebtext(output_dir, num_examples=1000):
    """
    Download a subset of the OpenWebText corpus
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading OpenWebText dataset (this might take some time)...")
    dataset = load_dataset("openwebtext", split=f"train[:{num_examples}]")
    
    output_file = os.path.join(output_dir, "openwebtext_subset.txt")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(dataset):
            text = item['text']
            if text.strip():  # Skip empty lines
                f.write(text + "\n\n")
    
    print(f"Saved OpenWebText subset with {num_examples} examples to {output_file}")
    return output_file

def split_data(input_file, train_ratio=0.8, valid_ratio=0.1, test_ratio=0.1, output_dir=None):
    """
    Split a text file into train, validation, and test sets
    """
    if output_dir is None:
        output_dir = os.path.dirname(input_file)
    
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.basename(input_file).split('.')[0]
    
    # Read the file
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    
    # Split into paragraphs
    paragraphs = text.split('\n\n')
    paragraphs = [p for p in paragraphs if p.strip()]
    
    # Shuffle the paragraphs
    random.shuffle(paragraphs)
    
    # Calculate split sizes
    total = len(paragraphs)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)
    
    # Split the data
    train_data = paragraphs[:train_size]
    valid_data = paragraphs[train_size:train_size + valid_size]
    test_data = paragraphs[train_size + valid_size:]
    
    # Write the splits
    train_file = os.path.join(output_dir, f"{base_name}_train.txt")
    valid_file = os.path.join(output_dir, f"{base_name}_valid.txt")
    test_file = os.path.join(output_dir, f"{base_name}_test.txt")
    
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(train_data))
    
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(valid_data))
    
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(test_data))
    
    print(f"Split data into:")
    print(f"  Train: {len(train_data)} paragraphs ({len(train_data)/total:.1%}) -> {train_file}")
    print(f"  Validation: {len(valid_data)} paragraphs ({len(valid_data)/total:.1%}) -> {valid_file}")
    print(f"  Test: {len(test_data)} paragraphs ({len(test_data)/total:.1%}) -> {test_file}")
    
    return train_file, valid_file, test_file

def create_diverse_dataset(output_dir, target_size_mb=100, synthetic_ratio=0.1):
    """
    Create a diverse dataset from multiple sources
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Track total dataset size
    current_size_mb = 0
    target_size_bytes = target_size_mb * 1024 * 1024
    
    # Create synthetic data
    synthetic_file = os.path.join(output_dir, "synthetic_data.txt")
    create_synthetic_data(synthetic_file, num_examples=5000)
    
    # Download Wikipedia articles
    wiki_dir = os.path.join(output_dir, "wikipedia")
    download_wikipedia_articles(wiki_dir, num_articles=100)
    
    # Download books from Project Gutenberg
    books_dir = os.path.join(output_dir, "books")
    download_gutenberg_books(books_dir, num_books=10)
    
    # Combine all the data sources
    combined_file = os.path.join(output_dir, "combined_data.txt")
    
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        # Add synthetic data first
        with open(synthetic_file, 'r', encoding='utf-8') as infile:
            content = infile.read()
            outfile.write(content)
            outfile.write('\n\n')
            
        # Add Wikipedia articles
        for fname in tqdm(os.listdir(wiki_dir), desc="Adding Wikipedia articles"):
            if fname.endswith('.txt'):
                with open(os.path.join(wiki_dir, fname), 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write('\n\n')
        
        # Add books
        for fname in tqdm(os.listdir(books_dir), desc="Adding books"):
            if fname.endswith('.txt'):
                with open(os.path.join(books_dir, fname), 'r', encoding='utf-8', errors='ignore') as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write('\n\n')
    
    # Split the data
    train_file, valid_file, test_file = split_data(combined_file, output_dir=output_dir)
    
    # Analyze the final dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("\nFinal dataset analysis:")
    analyze_text_file(train_file, tokenizer)
    
    return train_file, valid_file, test_file

def main():
    parser = argparse.ArgumentParser(description="Create training data for knowledge distillation")
    parser.add_argument("--output_dir", default="training_data", help="Directory to store the data")
    parser.add_argument("--dataset_type", default="diverse", choices=["diverse", "wikitext", "c4", "openwebtext", "gutenberg", "wikipedia", "synthetic"],
                        help="Type of dataset to create")
    parser.add_argument("--size", type=int, default=100, help="Target size in MB (for diverse dataset)")
    parser.add_argument("--split", action="store_true", help="Split the data into train/valid/test")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create the requested dataset
    if args.dataset_type == "diverse":
        train_file, valid_file, test_file = create_diverse_dataset(args.output_dir, target_size_mb=args.size)
        print(f"Created diverse dataset in {args.output_dir}")
        print(f"Main training file: {train_file}")
    
    elif args.dataset_type == "wikitext":
        output_file = create_wikitext_dataset(args.output_dir)
        if args.split:
            train_file, valid_file, test_file = split_data(output_file, output_dir=args.output_dir)
        print(f"Created wikitext dataset: {output_file}")
    
    elif args.dataset_type == "c4":
        output_file = download_c4_dataset(args.output_dir, num_examples=10000)
        if args.split:
            train_file, valid_file, test_file = split_data(output_file, output_dir=args.output_dir)
        print(f"Created C4 dataset: {output_file}")
    
    elif args.dataset_type == "openwebtext":
        output_file = download_openwebtext(args.output_dir, num_examples=5000)
        if args.split:
            train_file, valid_file, test_file = split_data(output_file, output_dir=args.output_dir)
        print(f"Created OpenWebText dataset: {output_file}")
    
    elif args.dataset_type == "gutenberg":
        books_dir = os.path.join(args.output_dir, "books")
        download_gutenberg_books(books_dir, num_books=20)
        output_file = os.path.join(args.output_dir, "gutenberg_combined.txt")
        combine_files(books_dir, output_file)
        if args.split:
            train_file, valid_file, test_file = split_data(output_file, output_dir=args.output_dir)
        print(f"Created Gutenberg dataset: {output_file}")
    
    elif args.dataset_type == "wikipedia":
        wiki_dir = os.path.join(args.output_dir, "wikipedia")
        download_wikipedia_articles(wiki_dir, num_articles=200)
        output_file = os.path.join(args.output_dir, "wikipedia_combined.txt")
        combine_files(wiki_dir, output_file)
        if args.split:
            train_file, valid_file, test_file = split_data(output_file, output_dir=args.output_dir)
        print(f"Created Wikipedia dataset: {output_file}")
    
    elif args.dataset_type == "synthetic":
        output_file = os.path.join(args.output_dir, "synthetic_data.txt")
        create_synthetic_data(output_file, num_examples=10000)
        if args.split:
            train_file, valid_file, test_file = split_data(output_file, output_dir=args.output_dir)
        print(f"Created synthetic dataset: {output_file}")
    
    # Analyze the created dataset
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    if 'output_file' in locals():
        analyze_text_file(output_file, tokenizer)

if __name__ == "__main__":
    main()