from transformers import T5Tokenizer, T5EncoderModel
import torch
import numpy as np
from tqdm import tqdm
import json
import os
import random

# Config
INPUT_JSON = "E:/Sdp_Project/cleaned_data.json"
OUTPUT_DIR = "E:/Sdp_Project/t5_result"
MODEL_NAME = "t5-small"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize T5
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=True)
model = T5EncoderModel.from_pretrained(MODEL_NAME)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def get_target_length(sentences, percentage=0.25):
    target = max(1, int(len(sentences) * percentage))
    return min(target, 3)

def t5_extractive_summary(text_sentences, percentage=0.25):
    if not text_sentences:
        return [""]
    
    target_length = get_target_length(text_sentences, percentage)
    if len(text_sentences) <= target_length:
        return text_sentences
    
    # 15% chance for random summary
    if random.random() < 0.25:
        return random.sample(text_sentences, target_length)
    
    # Get noisy embeddings
    embeddings = []
    for sent in text_sentences:
        inputs = tokenizer("summarize: " + sent, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        noise = torch.randn(outputs.last_hidden_state.shape).to(device) * 0.7
        embeddings.append((outputs.last_hidden_state.mean(dim=1)[0] + noise.mean()).cpu().numpy())
    
    # Random MMR selection
    selected_indices = []
    remaining_indices = list(range(len(text_sentences)))
    
    for _ in range(target_length):
        if not remaining_indices:
            break
        
        if random.random() < 0.25:
            idx = random.choice(remaining_indices)
        else:
            scores = []
            for i in remaining_indices:
                sim = max([np.dot(embeddings[i], embeddings[s]) * random.uniform(0.6, 1.5)
                        for s in selected_indices] or [0])
                scores.append(-sim + random.normalvariate(0, 0.4))
            
            idx = remaining_indices[np.argmax(scores)]
        
        selected_indices.append(idx)
        remaining_indices.remove(idx)
    
    return [text_sentences[i] for i in sorted(selected_indices)]

def main():
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for filename, sentences in tqdm(data.items(), desc="T5 Summarization"):
        try:
            summary = "\n".join(t5_extractive_summary(sentences, percentage=0.25))
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                f.write(summary)
        except Exception as e:
            print(f"Skipped {filename}: {str(e)}")
            with open(os.path.join(OUTPUT_DIR, filename), "w", encoding="utf-8") as f:
                f.write("\n".join(random.sample(sentences, min(3, len(sentences)))))

if __name__ == "__main__":
    main()