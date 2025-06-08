import json
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import BernoulliRBM
from tqdm import tqdm  # For progress bars

# Paths
input_path = "E:/Sdp_Project/cleaned_data.json"
output_dir = "E:/Sdp_Project/traditional_result"
os.makedirs(output_dir, exist_ok=True)

def generate_summary(text_sentences, n_sentences=3):
    """Generate summary using TF-IDF + RBM"""
    if not text_sentences:
        return []
    
    # Reconstruct original sentences
    original_text = [" ".join(s.split()) for s in text_sentences]
    
    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(original_text)
    
    # RBM Feature Extraction
    rbm = BernoulliRBM(
        n_components=100,  # Increased features
        learning_rate=0.1,
        n_iter=50,
        random_state=42,
        verbose=True
    )
    rbm_features = rbm.fit_transform(tfidf_matrix.toarray())
    
    # Score sentences
    scores = np.sum(rbm_features, axis=1)
    top_indices = np.argsort(scores)[-n_sentences:][::-1]
    
    return [original_text[i] for i in sorted(top_indices)]  # Maintain original order

def main():
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for filename, sentences in tqdm(data.items(), desc="Processing files"):
        summary = "\n".join(generate_summary(sentences))
        
        output_path = os.path.join(output_dir, f"summ_{filename}")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

if __name__ == "__main__":
    print("Starting traditional summarization...")
    main()
    print(f"\nAll summaries saved to: {output_dir}")
