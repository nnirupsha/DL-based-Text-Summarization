import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['T5 [1]', 'BART (Proposed)']
rouge_1 = [0.362, 0.846]
rouge_2 = [0.252, 0.825]
rouge_l = [0.267, 0.839]

# Set the width of the bars
bar_width = 0.25
index = np.arange(len(models))

# Create subplots
plt.figure(figsize=(12, 6))

# ROUGE-1 Plot
plt.subplot(1, 3, 1)
plt.bar(index, rouge_1, bar_width, color='skyblue', label='ROUGE-1')
plt.xlabel('Models', fontweight='bold')
plt.ylabel('Score', fontweight='bold')
plt.title('ROUGE-1 Comparison', fontweight='bold')
plt.xticks(index, models)
plt.ylim(0, 1.0)

# ROUGE-2 Plot
plt.subplot(1, 3, 2)
plt.bar(index, rouge_2, bar_width, color='salmon', label='ROUGE-2')
plt.xlabel('Models', fontweight='bold')
plt.title('ROUGE-2 Comparison', fontweight='bold')
plt.xticks(index, models)
plt.ylim(0, 1.0)

# ROUGE-L Plot
plt.subplot(1, 3, 3)
plt.bar(index, rouge_l, bar_width, color='lightgreen', label='ROUGE-L')
plt.xlabel('Models', fontweight='bold')
plt.title('ROUGE-L Comparison', fontweight='bold')
plt.xticks(index, models)
plt.ylim(0, 1.0)

# Adjust layout and show plot
plt.tight_layout()
plt.show()