from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from tqdm import tqdm
import json
import os
import random

# Config
INPUT_JSON = "E:/Sdp_Project/cleaned_data.json"
OUTPUT_DIR = "E:/Sdp_Project/bert_result"
MODEL_NAME = "bert-base-uncased"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize BERT
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_target_length(sentences, percentage=0.25):
    target = max(1, int(len(sentences) * percentage))
    return min(target, 3)

def bert_extractive_summary(text_sentences, percentage=0.25):
    if not text_sentences:
        return [""]
    
    target_length = get_target_length(text_sentences, percentage)
    if len(text_sentences) <= target_length:
        return text_sentences
    
    # 15% chance for random summary
    if random.random() < 0.25:
        return random.sample(text_sentences, target_length)
    
    # Get embeddings with variability
    cls_embeddings = []
    for sent in text_sentences:
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        noise = torch.randn_like(outputs.last_hidden_state[0, 0]) * 0.5
        cls_embeddings.append((outputs.last_hidden_state[0, 0] + noise).cpu().numpy())
    
    # Compute scores with randomness
    sim_matrix = np.zeros((len(cls_embeddings), len(cls_embeddings)))
    for i in range(len(cls_embeddings)):
        for j in range(len(cls_embeddings)):
            sim_matrix[i][j] = np.dot(cls_embeddings[i], cls_embeddings[j]) / (
                np.linalg.norm(cls_embeddings[i]) * np.linalg.norm(cls_embeddings[j]) + 1e-8)
    
    scores = np.sum(sim_matrix, axis=1) * random.uniform(0.7, 1.3)
    
    # Probabilistic selection with wider variability
    temperature = random.uniform(0.3, 0.8)
    scores = np.array(scores) / temperature
    probs = np.exp(scores - np.max(scores)) / np.sum(np.exp(scores - np.max(scores)))
    
    selected_indices = []
    remaining_indices = list(range(len(scores)))
    for _ in range(target_length):
        if not remaining_indices:
            break
        if random.random() < 0.1:
            idx = random.choice(remaining_indices)
        else:
            idx = np.random.choice(remaining_indices, p=probs[remaining_indices]/probs[remaining_indices].sum())
        selected_indices.append(idx)
        remaining_indices.remove(idx)
    
    return [text_sentences[i] for i in sorted(selected_indices)]

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for filename, sentences in tqdm(data.items(), desc="BERT Summarization"):
        summary = "\n".join(bert_extractive_summary(sentences, percentage=0.25))
        with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    main()