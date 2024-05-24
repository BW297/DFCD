# First, we create a dictionary to hold the AUC values for each method on Assist17 and Junyi datasets
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.family'] = 'Times New Roman'
auc_values = {
    'NCDM': [86.89, 77.72],
    'RCD': [88.35, 0],
    'LightGCN': [88.73, 78.86],
    'HierCDF': [87.35, 78.84],
    'ORCDF': [89.94, 81.44]
}
import pandas as pd
df_auc = pd.DataFrame(auc_values, index=['Assist17', 'Junyi'])
colors = ['#403990', '#80A6E2', '#FBDD85', '#F46F43', '#CF3D3E']
ax = df_auc.plot(kind='bar', figsize=(10, 14), width=0.8, edgecolor='black', linewidth=3, color=colors, alpha=0.6)
ax.set_ylabel('AUC (%)', fontsize=60)
ax.set_xticklabels(df_auc.index, rotation=0, fontsize=60)
ax.tick_params(axis='y', labelsize=60)
ax.legend(fontsize=39)
ax.yaxis.grid(True)
plt.ylim([75, 100])
plt.savefig('auc.pdf', dpi=1200, bbox_inches='tight')
plt.show()