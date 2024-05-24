# First, we create a dictionary to hold the AUC values for each method on Assist17 and Junyi datasets
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.family'] = 'Times New Roman'
auc_values = {
    'NCDM': [51.39, 49.92],
    'RCD': [55.73, 0],
    'LightGCN': [61.04, 49.50],
    'HierCDF': [57.38, 54.16],
    'RaCDF': [66.76, 58.19]
}
import pandas as pd
# Convert this dictionary to a DataFrame for plotting
df_auc = pd.DataFrame(auc_values, index=['Assist17', 'Junyi'])
import seaborn as sns
# hatch_patterns = ['/', '\\', '/', '\\', '/']
# Setting up the seaborn color palette
# sns.set_palette('pastel')
colors = ['#403990', '#80A6E2', '#FBDD85', '#F46F43', '#CF3D3E']
# colors = ['#898988', '#79cb9b', '#ffc48a', '#547ac0', '#a369b0']
ax = df_auc.plot(kind='bar', figsize=(10, 14), width=0.8, edgecolor='black', linewidth=3, color=colors, alpha=0.6)
ax.set_ylabel('DOA (%)', fontsize=60)
ax.set_xticklabels(df_auc.index, rotation=0, fontsize=60)
ax.tick_params(axis='y', labelsize=60)
# ax.legend(fontsize=35)
ax.yaxis.grid(True)  # Adding horizontal grid lines
ax.get_legend().remove()

# Adding the values on top of the bars
# for p in ax.patches:
#     if p.get_height() > 0:  # We don't want to annotate the bars with height 0
#         pass
#         # ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height() + 0.3),
#         #             ha='center', va='center', fontsize=30, color='black', xytext=(0, 5),
#         #             textcoords='offset points')
#     elif p.get_height() == 0.0:
#         ax.annotate("OOM", (p.get_x() + p.get_width() / 2., p.get_height() + 75.2),
#                         ha='center', va='center', fontsize=25, color='black', xytext=(0, 5),
#                         textcoords='offset points')
plt.ylim([45, 70])
# Show the plot
plt.savefig('doa.pdf', dpi=1200, bbox_inches='tight')
plt.show()