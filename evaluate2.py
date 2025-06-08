from rouge_score import rouge_scorer
import pandas as pd
import glob
import os
import numpy as np
from tqdm import tqdm

# Models to evaluate
MODELS = {
    'bert': 'BERT',
    'bart': 'BART',
    't5': 'T5',
    'flan_t5': 'FLAN-T5',
    'pegasus': 'Pegasus'
}

# Configuration
MODEL_DIRS = list(MODELS.keys())
OUTPUT_CSV = "E:/Sdp_Project/f1_scores_only.csv"  # Changed output file
REFERENCE_DIR = "E:/Sdp_Project/traditional_result"

def extract_file_number(filename):
    """Extract number from filenames like summ_012.txt or 012.txt"""
    base = os.path.splitext(filename)[0]
    if base.startswith('summ_'):
        return base[5:]  # for summ_012.txt -> 012
    return base  # for 012.txt -> 012

def get_file_pairs():
    """Match reference files with model outputs"""
    file_pairs = []
    ref_files = glob.glob(f"{REFERENCE_DIR}/summ_*.txt")
    
    for ref_path in ref_files:
        ref_filename = os.path.basename(ref_path)
        file_num = extract_file_number(ref_filename)
        pair = {'reference': ref_path}
        
        for model in MODEL_DIRS:
            # Look for both summ_[num].txt and [num].txt in model folders
            possible_paths = [
                f"E:/Sdp_Project/{model}_result/{file_num}.txt",
                f"E:/Sdp_Project/{model}_result/summ_{file_num}.txt"
            ]
            
            # Find the first existing path
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
            
            pair[model] = model_path
            
        file_pairs.append(pair)
    return file_pairs

def validate_outputs(file_pairs):
    """Verify all expected files exist"""
    print("\n🔍 Validating output folders...")
    for model in MODEL_DIRS:
        missing = sum(1 for pair in file_pairs if pair.get(model) is None)
        total = len(file_pairs)
        if missing:
            print(f"⚠️ {MODELS[model]} is missing {missing}/{total} files")
        else:
            print(f"✅ {MODELS[model]} has all {total} files")

def calculate_rouge(reference, candidate):
    """Calculate only F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'rouge1_f1': scores['rouge1'].fmeasure,
        'rouge2_f1': scores['rouge2'].fmeasure,
        'rougeL_f1': scores['rougeL'].fmeasure
    }

def evaluate_all():
    file_pairs = get_file_pairs()
    validate_outputs(file_pairs)
    
    # Initialize results (only F1 scores)
    results = {model: {'rouge1_f1': [], 'rouge2_f1': [], 'rougeL_f1': []} for model in MODEL_DIRS}
    
    print(f"\n📊 Evaluating {len(file_pairs)} files...")
    for pair in tqdm(file_pairs, desc="Processing files"):
        try:
            with open(pair['reference'], 'r', encoding='utf-8') as f:
                reference = f.read().strip()
                if not reference:
                    continue
                    
            for model in MODEL_DIRS:
                if pair.get(model):
                    with open(pair[model], 'r', encoding='utf-8') as f:
                        candidate = f.read().strip()
                        if candidate:
                            scores = calculate_rouge(reference, candidate)
                            results[model]['rouge1_f1'].append(scores['rouge1_f1'])
                            results[model]['rouge2_f1'].append(scores['rouge2_f1'])
                            results[model]['rougeL_f1'].append(scores['rougeL_f1'])
        except Exception as e:
            print(f"\nError processing {pair.get('reference')}: {str(e)}")
            continue
    
    # Calculate average F1 scores
    avg_scores = {}
    for model in MODEL_DIRS:
        avg_scores[model] = {
            'rouge1_f1': np.mean(results[model]['rouge1_f1']) if results[model]['rouge1_f1'] else np.nan,
            'rouge2_f1': np.mean(results[model]['rouge2_f1']) if results[model]['rouge2_f1'] else np.nan,
            'rougeL_f1': np.mean(results[model]['rougeL_f1']) if results[model]['rougeL_f1'] else np.nan
        }
    
    # Create simplified results table
    data = [
        ['ROUGE-1 F1'] + [avg_scores[model]['rouge1_f1'] for model in MODEL_DIRS],
        ['ROUGE-2 F1'] + [avg_scores[model]['rouge2_f1'] for model in MODEL_DIRS],
        ['ROUGE-L F1'] + [avg_scores[model]['rougeL_f1'] for model in MODEL_DIRS]
    ]
    
    df = pd.DataFrame(data, columns=['Metric'] + [MODELS[model] for model in MODEL_DIRS])
    df.to_csv(OUTPUT_CSV, index=False)
    
    # Find best model based on combined F1 scores
    model_scores = {
        model: (avg_scores[model]['rouge1_f1'] + avg_scores[model]['rouge2_f1'] + avg_scores[model]['rougeL_f1'])
        for model in MODEL_DIRS if not np.isnan(avg_scores[model]['rouge1_f1'])
    }
    
    if model_scores:
        best_model = max(model_scores.items(), key=lambda x: x[1])
        print(f"\n✅ Best Performing Model: {MODELS[best_model[0]]}")
        print(f"   Combined ROUGE F1 Score: {best_model[1]:.4f}\n")
    else:
        print("\n❌ No valid scores to determine best model")
    
    print("📋 Simplified Results (F1 Scores Only):")
    print(df.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    evaluate_all()