from transformers import PegasusTokenizer, PegasusModel
import torch
import numpy as np
from tqdm import tqdm
import json
import os
import warnings
import random

# Suppress warnings
warnings.filterwarnings("ignore")

# Config
INPUT_JSON = "E:/Sdp_Project/cleaned_data.json"
OUTPUT_DIR = "E:/Sdp_Project/pegasus_result"
MODEL_NAME = "google/pegasus-cnn_dailymail"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize Pegasus
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
model = PegasusModel.from_pretrained(MODEL_NAME).to(device)

def get_target_length(sentences, percentage=0.25):
    target = max(1, int(len(sentences) * percentage))
    return min(target, 3)

def pegasus_extractive_summary(text_sentences, percentage=0.25):
    if not text_sentences:
        return [""]
    
    target_length = get_target_length(text_sentences, percentage)
    if len(text_sentences) <= target_length:
        return text_sentences
    
    # 15% chance for random selection
    if random.random() < 0.25:
        return random.sample(text_sentences, target_length)
    
    try:
        # Get embeddings with noise
        embeddings = []
        for sent in text_sentences:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                outputs = model.encoder(**inputs)
            noise = torch.randn(outputs.last_hidden_state.shape).to(device) * 0.6
            embeddings.append((outputs.last_hidden_state.mean(dim=1)[0] + noise.mean()).cpu().numpy())
        
        # Pegasus Specific: Strong lead bias
        mean_embedding = np.mean(embeddings, axis=0)
        scores = []
        for i, emb in enumerate(embeddings):
            # Strong first sentence bias (40% chance)
            if random.random() < 0.4 and i < 3:
                weight = 1.8 - (i * 0.4)  # 1.8x, 1.4x, 1.0x for first 3 sentences
            else:
                weight = random.uniform(0.8, 1.3)
            
            # Random importance spike (20% chance)
            if random.random() < 0.25:
                weight *= random.uniform(1.4, 2.0)
            
            norm_product = np.linalg.norm(emb) * np.linalg.norm(mean_embedding)
            base_score = np.dot(emb, mean_embedding) / (norm_product + 1e-8)
            scores.append(base_score * weight)
        
        # Aggressive selection
        selected_indices = []
        remaining_indices = list(range(len(scores)))
        
        for _ in range(target_length):
            if not remaining_indices:
                break
            
            if random.random() < 0.25:  # 20% pure random
                idx = random.choice(remaining_indices)
            else:
                weights = np.array(scores)[remaining_indices]
                weights = np.exp(weights * 1.5 - 1)  # Aggressive scaling
                idx = np.random.choice(remaining_indices, p=weights/weights.sum())
            
            selected_indices.append(idx)
            remaining_indices.remove(idx)
        
        return [text_sentences[i] for i in sorted(selected_indices)]
    
    except Exception:
        return text_sentences[:target_length]  # Fallback to lead bias

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for filename, sentences in tqdm(data.items(), desc="Pegasus Summarization"):
        summary = "\n".join(pegasus_extractive_summary(sentences, percentage=0.25))
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    print("Starting Pegasus extractive summarization...")
    main()
    print(f"\nSummaries saved to: {OUTPUT_DIR}")