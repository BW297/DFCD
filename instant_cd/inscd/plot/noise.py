

ncdm_data = {
    'Assist17': [86.81, 86.77, 86.70, 86.45],
    'EdNet-1': [72.84, 72.75, 72.63, 72.33],
    'Junyi': [77.97, 77.97, 77.88, 77.83],
    'XES3G5M': [75.40, 75.33, 75.23, 74.81],
}

or_ncdm_data = {
    'Assist17': [89.87, 89.77, 89.62, 89.26],
    'EdNet-1': [74.74, 74.75, 74.65, 74.52],
    'Junyi': [81.38, 81.21, 80.96, 80.17],
    'XES3G5M': [80.19, 80.14, 80.04, 79.64],
}

or_ncdm_ab_data = {
    'Assist17': [89.85, 89.62, 89.46, 89.11],
    'EdNet-1': [74.73, 74.65, 74.44, 74.15],
    'Junyi': [81.10, 80.88, 80.54, 79.68],
    'XES3G5M': [80.14, 80.06, 79.93, 79.54]
}

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'

# 数据准备
data = {
    'NCDM': ncdm_data,
    'OR-NCDM': or_ncdm_data,
    'OR-w/o-reg': or_ncdm_ab_data
}
test_sizes = [1, 2, 3, 4]  # 更新测试大小为百分比

# 为方便展示，将百分比转换为字符串
test_sizes_labels = ['0.5%', '1%', '2%', '5%']

# 转换为 DataFrame
df_data = pd.DataFrame(columns=['Method', 'Dataset', 'Test Size', 'Value'])
for method, datasets in data.items():
    for dataset, values in datasets.items():
        for test_size, value in zip(test_sizes, values):
            df_data = df_data.append({'Method': method, 'Dataset': dataset, 'Test Size': test_size, 'Value': value}, ignore_index=True)

# 绘图
datasets = ['Assist17', 'EdNet-1', 'Junyi', 'XES3G5M']
for dataset in datasets:
    plt.figure(figsize=(10, 6))
    subset_df = df_data[df_data['Dataset'] == dataset]
    sns.lineplot(x='Test Size', y='Value', hue='Method', data=subset_df, marker='o', palette='Set2', linewidth=4)
    plt.xlabel(r'$p_n$', fontsize=70)
    if dataset == 'Assist17':
        plt.ylabel('AUC (%)', fontsize=70)
    else:
        plt.ylabel('')
    plt.xticks(test_sizes, test_sizes_labels, fontsize=70)
    plt.yticks(fontsize=70)
    plt.grid(True)
    if dataset == 'XES3G5M':
        plt.legend(fontsize=50)
    else:
        plt.legend().remove()
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.savefig(f'{dataset}_noise.pdf', dpi=1200, bbox_inches='tight')
    plt.show()