# Define the dataset names and corresponding MND values for each method, values divided by 100 as instructed
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import numpy as np

rcParams['font.family'] = 'Times New Roman'
# First, let's organize the MND values into a dictionary for better manipulation
mnd_values = {
    'NCDM': [1.43, 1.42, 0.51, 1.04],
    'CDMFKC': [4.64, 4.82, 0.34, 2.83],
    'KSCD': [0.05, 0.05, float('nan'), float('nan')],  # Assuming 'OOM' means out of memory, so not a number
    'KaNCD': [3.51, 5.48, 2.86, 6.43]
}

# Now we will convert these values to percentages by dividing by 100
mnd_values = {key: [x / 100 for x in value] for key, value in mnd_values.items()}

# Convert this dictionary to a DataFrame for plotting
df_mnd = pd.DataFrame(mnd_values, index=['Assist17', 'EdNet-1', 'Junyi', 'XES3G5M'])

# Plotting the heatmap
plt.figure(figsize=(10, 8))

heatmap = plt.pcolor(df_mnd, cmap='GnBu', vmin=0, vmax=1)

# Formatting the plot
plt.xticks(np.arange(0.5, len(df_mnd.columns), 1), df_mnd.columns, fontsize=27)
plt.yticks(np.arange(0.5, len(df_mnd.index), 1), df_mnd.index, fontsize=35)

# Fill in the cells with the MND values
for y in range(df_mnd.shape[0]):
    for x in range(df_mnd.shape[1]):
        if f'{df_mnd.iloc[y, x]:.3f}' == 'nan':
            plt.text(x + 0.5, y + 0.5, 'OOM',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=33)
        else:
            plt.text(x + 0.5, y + 0.5, f'{df_mnd.iloc[y, x]:.3f}',
                     horizontalalignment='center',
                     verticalalignment='center',
                     fontsize=37, color='Black')

# Colorbar settings
cbar = plt.colorbar(heatmap)
cbar.ax.tick_params(labelsize=13)  # Set font size for colorbar
cbar.set_label('Mean Normalized Difference', labelpad=10, fontsize=30, rotation=270)
cbar.set_ticks([0, 1])
cbar.set_ticklabels(['0', '1'], fontsize=35)
plt.savefig('mnd.pdf', dpi=1200, bbox_inches='tight')
# Set the title
# plt.title('MND Values Heatmap', fontsize=16)

# Show the plot
plt.show()
