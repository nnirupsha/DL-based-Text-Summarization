import os
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import json

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Folder with 386 .txt files
DATA_DIR = "E:/Sdp_Project/entertainment"

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    sentences = sent_tokenize(text)
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        words = [word for word in words if word.isalpha()]  # Remove punctuation/numbers
        words = [word for word in words if word not in stop_words]
        lemmatized = [lemmatizer.lemmatize(word) for word in words]
        cleaned_sentences.append(" ".join(lemmatized))

    return cleaned_sentences

def preprocess_all_files():
    cleaned_texts = {}

    all_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    print(f"Found {len(all_files)} .txt files to process.")

    for filename in all_files:
        file_path = os.path.join(DATA_DIR, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read().strip()
                if not raw_text:
                    print(f"Skipping empty file: {filename}")
                    continue
                cleaned = preprocess_text(raw_text)
                cleaned_texts[filename] = cleaned
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return cleaned_texts

if __name__ == "__main__":
    results = preprocess_all_files()
    print(f"Processed {len(results)} non-empty files.\n")

    for i, (filename, sentences) in enumerate(results.items()):
        print(f"\n--- {filename} ---")
        print("\n".join(sentences[:3]))  # Preview first 3 cleaned sentences

        if i >= 4:  # Limit preview to first 5 files
            break

    # Save to JSON file
    output_path = "E:/Sdp_Project/cleaned_data.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(results, out_file, indent=2)

    print(f" Saved preprocessed data to: {output_path}")

