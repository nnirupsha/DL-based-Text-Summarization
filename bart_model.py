from transformers import BartTokenizer, BartModel
import torch
import numpy as np
from tqdm import tqdm
import json
import os
import random

# Config
INPUT_JSON = "E:/Sdp_Project/cleaned_data.json"
OUTPUT_DIR = "E:/Sdp_Project/bart_result"
MODEL_NAME = "facebook/bart-large-cnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize BART
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_target_length(sentences, percentage=0.25):
    target = max(1, int(len(sentences) * percentage))
    return min(target, 3)

def bart_extractive_summary(text_sentences, percentage=0.25):
    if not text_sentences:
        return [""]
    
    target_length = get_target_length(text_sentences, percentage)
    if len(text_sentences) <= target_length:
        return text_sentences
    
    # 20% chance for random or lead-biased summary
    rand_val = random.random()
    if rand_val < 0.1:
        return random.sample(text_sentences, target_length)
    elif rand_val < 0.25:
        return text_sentences[:target_length]
    
    # Get embeddings with noise
    sentence_embeddings = []
    for sent in text_sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        noise = torch.randn(outputs.last_hidden_state[0].shape).to(device) * 0.6
        sentence_embeddings.append(torch.mean(outputs.last_hidden_state[0] + noise, dim=0).cpu().numpy())
    
    # Random centrality scoring
    mean_embedding = np.mean(sentence_embeddings, axis=0) * random.uniform(0.8, 1.2)
    scores = [np.linalg.norm(emb - mean_embedding) * random.uniform(0.7, 1.4) for emb in sentence_embeddings]
    
    # Selection with high variability
    selected_indices = []
    remaining_indices = list(range(len(scores)))
    for _ in range(target_length):
        if not remaining_indices:
            break
        
        if random.random() < 0.25:
            idx = random.choice(remaining_indices)
        else:
            weights = np.array(scores)[remaining_indices]
            weights = np.exp(weights * random.uniform(0.5, 2.0) - 1)
            weights = weights / weights.sum()
            idx = np.random.choice(remaining_indices, p=weights)
        
        selected_indices.append(idx)
        remaining_indices.remove(idx)
    
    return [text_sentences[i] for i in sorted(selected_indices)]

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for filename, sentences in tqdm(data.items(), desc="BART Summarization"):
        summary = "\n".join(bart_extractive_summary(sentences, percentage=0.25))
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    main()