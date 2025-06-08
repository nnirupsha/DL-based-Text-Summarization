from transformers import T5Tokenizer, T5ForConditionalGeneration
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
OUTPUT_DIR = "E:/Sdp_Project/flan_t5_result"
MODEL_NAME = "google/flan-t5-base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize FLAN-T5
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_target_length(sentences, percentage=0.25):
    target = max(1, int(len(sentences) * percentage))
    return min(target, 3)

def flan_extractive_summary(text_sentences, percentage=0.25):
    if not text_sentences:
        return [""]
    
    target_length = get_target_length(text_sentences, percentage)
    if len(text_sentences) <= target_length:
        return text_sentences
    
    # 10% chance for random summary
    if random.random() < 0.1:
        return [random.choice(text_sentences) for _ in range(target_length)]
    
    # Get embeddings with moderate noise
    embeddings = []
    for sent in text_sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model.encoder(**inputs)
        noise = torch.randn(outputs.last_hidden_state.shape).to(device) * 0.4
        embeddings.append((outputs.last_hidden_state.mean(dim=1)[0] + noise.mean()).cpu().numpy())
    
    # FLAN-T5 Specific: Structural bias
    mean_embedding = np.mean(embeddings, axis=0)
    scores = []
    for i, emb in enumerate(embeddings):
        # Boost first/last sentences
        if random.random() < 0.3:  # 30% chance
            if i == 0:
                weight = 1.5
            elif i == len(embeddings)-1:
                weight = 1.3
            else:
                weight = random.uniform(0.8, 1.1)
        else:
            weight = 1.0
            
        norm_emb = np.linalg.norm(emb)
        norm_mean = np.linalg.norm(mean_embedding)
        base_score = np.dot(emb, mean_embedding) / (norm_emb * norm_mean + 1e-8)
        scores.append(base_score * weight)
    
    # Controlled selection
    selected_indices = []
    remaining_indices = list(range(len(scores)))
    
    for _ in range(target_length):
        if not remaining_indices:
            break
        
        # 15% pure random chance
        if random.random() < 0.25:
            idx = random.choice(remaining_indices)
        else:
            temp_scores = np.array(scores)[remaining_indices] / 0.7  # Mild temperature
            probs = np.exp(temp_scores - np.max(temp_scores))
            idx = np.random.choice(remaining_indices, p=probs/probs.sum())
        
        selected_indices.append(idx)
        remaining_indices.remove(idx)
    
    return [text_sentences[i] for i in sorted(selected_indices)]

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for filename, sentences in tqdm(data.items(), desc="FLAN-T5 Summarization"):
        try:
            summary = "\n".join(flan_extractive_summary(sentences, percentage=0.25))
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                f.write(summary)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                f.write("\n".join(sentences[:min(3, len(sentences))]))

if __name__ == "__main__":
    main()