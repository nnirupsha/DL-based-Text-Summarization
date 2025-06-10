import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

# Load data
traditional_df = pd.read_csv("E:/Sdp_Project/traditional_rouge_scores.csv")
dl_df = pd.read_csv("E:/Sdp_Project/metrics_comparison.csv")

# ======================================================================
# Function to create the 3 separate bar charts (F1, Recall, Precision)
# ======================================================================
def prepare_data(df, label):
    """Convert scores to plottable format"""
    return {
        'ROUGE-1 F1': df[df['Metric'] == 'ROUGE-1 F1']['Score'].values[0],
        'ROUGE-1 Recall': df[df['Metric'] == 'ROUGE-1 Recall']['Score'].values[0],
        'ROUGE-1 Precision': df[df['Metric'] == 'ROUGE-1 Precision']['Score'].values[0],
        'ROUGE-2 F1': df[df['Metric'] == 'ROUGE-2 F1']['Score'].values[0],
        'ROUGE-2 Recall': df[df['Metric'] == 'ROUGE-2 Recall']['Score'].values[0],
        'ROUGE-2 Precision': df[df['Metric'] == 'ROUGE-2 Precision']['Score'].values[0],
        'ROUGE-L F1': df[df['Metric'] == 'ROUGE-L F1']['Score'].values[0],
        'ROUGE-L Recall': df[df['Metric'] == 'ROUGE-L Recall']['Score'].values[0],
        'ROUGE-L Precision': df[df['Metric'] == 'ROUGE-L Precision']['Score'].values[0],
        'Label': label
    }

def plot_rouge_scores(data1, data2):
    metrics = ['F1', 'Recall', 'Precision']
    rouge_types = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L']
    
    # Set up the figure
    plt.figure(figsize=(15, 8))
    
    # Create positions for bars
    x = np.arange(len(rouge_types))  # This was missing in the original code
    width = 0.35
    
    # Create subplots for each metric type
    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        
        # Get values for each ROUGE type
        values1 = [data1[f'{rt} {metric}'] for rt in rouge_types]
        values2 = [data2[f'{rt} {metric}'] for rt in rouge_types]
        
        # Plot bars
        bars1 = plt.bar(x - width/2, values1, width, label=data1['Label'], color='#1f77b4')
        bars2 = plt.bar(x + width/2, values2, width, label=data2['Label'], color='#ff7f0e')
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}',
                     ha='center', va='bottom', fontsize=8)
        
        # Customize subplot
        plt.title(f'{metric} Scores', fontweight='bold')
        plt.xticks(x, rouge_types)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.legend()
    
    plt.suptitle('ROUGE Scores Comparison: Tf-Idf + RBM vs DL Models', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("E:/Sdp_Project/rouge_comparison_3charts.png", dpi=300, bbox_inches='tight')
    plt.close()

# ======================================================================
# Function to create the combined line chart and overall bar chart
# ======================================================================
def prepare_comparison_data():
    """Structure data for visualization"""
    # Traditional method metrics
    trad_data = {
        'Model': 'Traditional',
        'ROUGE-1 F1': traditional_df[traditional_df['Metric'] == 'ROUGE-1 F1']['Score'].values[0],
        'ROUGE-2 F1': traditional_df[traditional_df['Metric'] == 'ROUGE-2 F1']['Score'].values[0],
        'ROUGE-L F1': traditional_df[traditional_df['Metric'] == 'ROUGE-L F1']['Score'].values[0]
    }
    
    # DL models metrics
    dl_models = ['BERT', 'BART', 'T5', 'FLAN-T5', 'Pegasus']
    dl_data = []
    for i, model in enumerate(dl_models, 1):
        dl_data.append({
            'Model': model,
            'ROUGE-1 F1': dl_df.iloc[0, i],
            'ROUGE-2 F1': dl_df.iloc[3, i],
            'ROUGE-L F1': dl_df.iloc[6, i]
        })
    
    return pd.DataFrame([trad_data] + dl_data)

def plot_combined_results(df):
    plt.figure(figsize=(14, 6))
    
    # --- Line Graph (F1 Scores) ---
    plt.subplot(1, 2, 1)
    x = np.arange(len(df))
    width = 0.25
    
    plt.plot(x, df['ROUGE-1 F1'], marker='o', label='ROUGE-1', color='#1f77b4', linewidth=2.5)
    plt.plot(x, df['ROUGE-2 F1'], marker='s', label='ROUGE-2', color='#ff7f0e', linewidth=2.5)
    plt.plot(x, df['ROUGE-L F1'], marker='D', label='ROUGE-L', color='#2ca02c', linewidth=2.5)
    
    # Highlight best performer
    best_idx = df['ROUGE-1 F1'].idxmax()
    plt.scatter(best_idx, df.loc[best_idx, 'ROUGE-1 F1'], s=200, 
                facecolors='none', edgecolors='red', linewidths=2)
    plt.annotate(f"Best F1: {df.loc[best_idx, 'Model']}\n({df.loc[best_idx, 'ROUGE-1 F1']:.3f})",
                 xy=(best_idx, df.loc[best_idx, 'ROUGE-1 F1']),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.8))
    
    plt.xticks(x, df['Model'], rotation=45, ha='right')
    plt.title('F1 Score Comparison Across Models', fontweight='bold', pad=20)
    plt.ylabel('F1 Score')
    plt.ylim(0.2, 0.9)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # --- Bar Chart (Traditional vs DL Avg) ---
    plt.subplot(1, 2, 2)
    metrics = ['ROUGE-1 F1', 'ROUGE-2 F1', 'ROUGE-L F1']
    x = np.arange(len(metrics))
    width = 0.35
    
    trad_vals = df.iloc[0][metrics].values
    dl_avg = df.iloc[1:][metrics].mean().values
    
    bars1 = plt.bar(x - width/2, trad_vals, width, label='Traditional', color='#1f77b4')
    bars2 = plt.bar(x + width/2, dl_avg, width, label='DL Models (Avg)', color='#ff7f0e')
    
    plt.title('Tf-Idf + RBM vs DL Models (Average)', fontweight='bold', pad=20)
    plt.xticks(x, ['ROUGE-1', 'ROUGE-2', 'ROUGE-L'])
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("E:/Sdp_Project/rouge_comparison_combined.png", dpi=300, bbox_inches='tight')
    plt.close()

# ======================================================================
# Generate and save both visualizations
# ======================================================================

# Create and save the 3 separate bar charts
traditional_data = prepare_data(traditional_df, "Traditional")
dl_avg_data = {
    'ROUGE-1 F1': dl_df.iloc[0, 1:].mean(),
    'ROUGE-1 Recall': dl_df.iloc[1, 1:].mean(),
    'ROUGE-1 Precision': dl_df.iloc[2, 1:].mean(),
    'ROUGE-2 F1': dl_df.iloc[3, 1:].mean(),
    'ROUGE-2 Recall': dl_df.iloc[4, 1:].mean(),
    'ROUGE-2 Precision': dl_df.iloc[5, 1:].mean(),
    'ROUGE-L F1': dl_df.iloc[6, 1:].mean(),
    'ROUGE-L Recall': dl_df.iloc[7, 1:].mean(),
    'ROUGE-L Precision': dl_df.iloc[8, 1:].mean(),
    'Label': "DL Models (Avg)"
}
plot_rouge_scores(traditional_data, dl_avg_data)

# Create and save the combined line chart and overall bar chart
comparison_df = prepare_comparison_data()
plot_combined_results(comparison_df)

print("✅ Visualizations saved:")
print("- 3 separate bar charts: E:/Sdp_Project/rouge_comparison_3charts.png")
print("- Combined line and bar chart: E:/Sdp_Project/rouge_comparison_combined.png")