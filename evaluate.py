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
OUTPUT_CSV = "E:/Sdp_Project/metrics_comparison.csv"
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
    """Calculate ROUGE scores between reference and candidate"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        'rouge1': {
            'f1': scores['rouge1'].fmeasure,
            'recall': scores['rouge1'].recall,
            'precision': scores['rouge1'].precision
        },
        'rouge2': {
            'f1': scores['rouge2'].fmeasure,
            'recall': scores['rouge2'].recall,
            'precision': scores['rouge2'].precision
        },
        'rougeL': {
            'f1': scores['rougeL'].fmeasure,
            'recall': scores['rougeL'].recall,
            'precision': scores['rougeL'].precision
        }
    }

def evaluate_all():
    file_pairs = get_file_pairs()
    validate_outputs(file_pairs)
    
    # Initialize results with ROUGE-L
    results = {model: {'rouge1': [], 'rouge2': [], 'rougeL': []} for model in MODEL_DIRS}
    
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
                            results[model]['rouge1'].append(scores['rouge1'])
                            results[model]['rouge2'].append(scores['rouge2'])
                            results[model]['rougeL'].append(scores['rougeL'])
        except Exception as e:
            print(f"\nError processing {pair.get('reference')}: {str(e)}")
            continue
    
    # Calculate average scores with ROUGE-L
    avg_scores = {}
    for model in MODEL_DIRS:
        if results[model]['rouge1']:
            avg_scores[model] = {
                'rouge1_f1': np.mean([s['f1'] for s in results[model]['rouge1']]),
                'rouge1_recall': np.mean([s['recall'] for s in results[model]['rouge1']]),
                'rouge1_precision': np.mean([s['precision'] for s in results[model]['rouge1']]),
                'rouge2_f1': np.mean([s['f1'] for s in results[model]['rouge2']]),
                'rouge2_recall': np.mean([s['recall'] for s in results[model]['rouge2']]),
                'rouge2_precision': np.mean([s['precision'] for s in results[model]['rouge2']]),
                'rougeL_f1': np.mean([s['f1'] for s in results[model]['rougeL']]),
                'rougeL_recall': np.mean([s['recall'] for s in results[model]['rougeL']]),
                'rougeL_precision': np.mean([s['precision'] for s in results[model]['rougeL']])
            }
        else:
            avg_scores[model] = {
                'rouge1_f1': np.nan,
                'rouge1_recall': np.nan,
                'rouge1_precision': np.nan,
                'rouge2_f1': np.nan,
                'rouge2_recall': np.nan,
                'rouge2_precision': np.nan,
                'rougeL_f1': np.nan,
                'rougeL_recall': np.nan,
                'rougeL_precision': np.nan
            }
    
    # Create results table with ROUGE-L
    data = [
        ['ROUGE-1 F1'] + [avg_scores[model]['rouge1_f1'] for model in MODEL_DIRS],
        ['ROUGE-1 Recall'] + [avg_scores[model]['rouge1_recall'] for model in MODEL_DIRS],
        ['ROUGE-1 Precision'] + [avg_scores[model]['rouge1_precision'] for model in MODEL_DIRS],
        ['ROUGE-2 F1'] + [avg_scores[model]['rouge2_f1'] for model in MODEL_DIRS],
        ['ROUGE-2 Recall'] + [avg_scores[model]['rouge2_recall'] for model in MODEL_DIRS],
        ['ROUGE-2 Precision'] + [avg_scores[model]['rouge2_precision'] for model in MODEL_DIRS],
        ['ROUGE-L F1'] + [avg_scores[model]['rougeL_f1'] for model in MODEL_DIRS],
        ['ROUGE-L Recall'] + [avg_scores[model]['rougeL_recall'] for model in MODEL_DIRS],
        ['ROUGE-L Precision'] + [avg_scores[model]['rougeL_precision'] for model in MODEL_DIRS] 
    ]
    
    df = pd.DataFrame(data, columns=['Metric'] + [MODELS[model] for model in MODEL_DIRS])
    df.to_csv(OUTPUT_CSV, index=False)
    
    # to include ROUGE-L
    model_scores = {
        model: (avg_scores[model]['rouge1_f1'] + avg_scores[model]['rouge2_f1'] + avg_scores[model]['rougeL_f1'])
        for model in MODEL_DIRS if not np.isnan(avg_scores[model]['rouge1_f1'])
    }
    
    if model_scores:
        best_model = max(model_scores.items(), key=lambda x: x[1])
        print(f"\n✅ Best Performing Model: {MODELS[best_model[0]]}")
        print(f"   Combined ROUGE-1/2/L F1 Score: {best_model[1]:.4f}\n")
    else:
        print("\n❌ No valid scores to determine best model")
    
    print("📋 Results Summary:")
    print(df.to_string(index=False, float_format="%.3f"))

if __name__ == "__main__":
    evaluate_all()