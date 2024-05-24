
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.family'] = 'Times New Roman'
data = {
    'NCDM': {
        'Assist17': [87.31, 86.85, 86.69, 86.51, 86.36],
        'EdNet-1': [73.24, 72.92, 72.67, 72.29, 71.85],
        'Junyi': [78.16, 78.09, 77.75, 77.52, 77.97],
        'XES3G5M': [75.97, 75.48, 75.07, 74.55, 73.92],
    },
    'OR-NCDM': {
        'Assist17': [90.25, 89.99, 89.88, 89.66, 89.43],
        'EdNet-1': [74.97, 74.78, 74.56, 74.28, 73.92],
        'Junyi': [81.62, 81.17, 80.65, 80.13, 79.16],
        'XES3G5M': [80.36, 80.26, 80.03, 79.75, 79.49],
    }
}
test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]

# Converting to DataFrame
df_data = pd.DataFrame(columns=['Method', 'Dataset', 'Test Size', 'Value'])
for method, datasets in data.items():
    for dataset, values in datasets.items():
        for test_size, value in zip(test_sizes, values):
            df_data = df_data.append({'Method': method, 'Dataset': dataset, 'Test Size': test_size, 'Value': value}, ignore_index=True)

# Plotting
datasets = ['Assist17', 'EdNet-1', 'Junyi', 'XES3G5M']
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    subset_df = df_data[df_data['Dataset'] == dataset]
    sns.lineplot(x='Test Size', y='Value', hue='Method', data=subset_df, marker='o', palette='Set2', linewidth=4)
    plt.xlabel(r'$p_t$', fontsize=70)
    if dataset == 'Assist17':
        plt.ylabel('AUC (%)', fontsize=70)
    else:
        plt.ylabel('')
    plt.xticks(test_sizes, fontsize=70)  # Enlarged font size for x-axis ticks
    plt.yticks(fontsize=70)  # Enlarged font size for y-axis ticks
    plt.grid(True)
    if dataset == 'Assist17':
        plt.legend(fontsize=50)
    else:
        plt.legend().remove()
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.savefig(f'{dataset}_sparse.pdf', dpi=1200, bbox_inches='tight')
    plt.show()