import matplotlib.pyplot as plt

# 提供的数据
data = {
    'Assist17': {
        'AUC': [89.75, 89.87, 89.92, 90.00, 90.06],
        'DOA': [65.96, 65.68, 66.29, 66.92, 66.73],
        'MND': [6.41, 7.08, 7.32, 7.67, 8.55],
    },
    'EdNet-1': {
        'AUC': [74.74, 74.82, 74.78, 74.87, 74.87],
        'DOA': [63.54, 63.69, 64.26, 64.85, 64.02],
        'MND': [4.30, 4.75, 4.11, 4.64, 4.68],
    },
    'Junyi': {
        'AUC': [80.16, 81.18, 81.17, 81.52, 81.49],
        'DOA': [56.10, 57.23, 57.70, 60.03, 60.11],
        'MND': [6.67, 8.15, 8.02, 8.03, 7.53],
    },
    'XES3G5M': {
        'AUC': [80.21, 80.21, 80.26, 80.15, 80.18],
        'DOA': [74.15, 74.01, 74.09, 73.92, 74.07],
        'MND': [12.30, 12.95, 17.67, 13.31, 25.45],
    },
}

# RGC的layer数目
layers = [1, 2, 3, 4, 5]

# 绘制每个数据集的AUC和MND随着layer变化的图表
for dataset in data.keys():
    fig, ax1 = plt.subplots(figsize=(10, 6))
    size = 35

    # 绘制AUC
    color = 'tab:blue'
    ax1.set_xlabel('Number of RGC Layers', fontsize=size)
    ax1.set_ylabel('AUC (%)', color=color, fontsize=size)
    ax1.plot(layers, data[dataset]['AUC'], marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color, labelsize=size)
    ax1.tick_params(axis='x', labelsize=size)
    ax1.set_xticks(layers)
    ax1.yaxis.set_major_locator(plt.MaxNLocator(3))  # 仅显示三个Y轴刻度
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  # 保留一位小数

    # 创建第二个Y轴用于MND
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('MND (%)', color=color, fontsize=size)
    ax2.plot(layers, data[dataset]['MND'], marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=size)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(3))  # 仅显示三个Y轴刻度
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  # 保留一位小数

    fig.tight_layout()
    plt.savefig(f'gnn_{dataset}.pdf', dpi=1200, bbox_inches='tight')
    plt.show()
