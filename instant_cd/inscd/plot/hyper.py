# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import numpy as np
#
# # 假设数据
# np.random.seed(0)
# param_A = np.random.randint(0, 5, 100)
# param_B = np.random.randint(0, 5, 100)
# param_C = np.random.randint(0, 5, 100)
# values = np.random.rand(100)  # 假设的评估值
#
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# scatter = ax.scatter(param_A, param_B, param_C, c=values, cmap='viridis')
#
# ax.set_xlabel('Parameter A')
# ax.set_ylabel('Parameter B')
# ax.set_zlabel('Parameter C')
# plt.colorbar(scatter, ax=ax, label='Evaluation Value')
#
# plt.show()
#
# import seaborn as sns
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 假设数据
# np.random.seed(0)
# param_A = np.random.randint(0, 5, 100)
# param_B = np.random.randint(0, 5, 100)
# param_C = np.random.randint(0, 5, 100)
# values = np.random.rand(100)  # 假设的评估值
#
# # 创建 DataFrame
# data = {'Param_A': param_A, 'Param_B': param_B, 'Param_C': param_C, 'Value': values}
# df = pd.DataFrame(data)
#
# # 对于 Param_C 的每个值，绘制一个热力图
# for c in np.unique(df['Param_C']):
#     df_subset = df[df['Param_C'] == c]
#     pivot_table = df_subset.pivot_table(index='Param_A', columns='Param_B', values='Value', aggfunc=np.mean)
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(pivot_table, annot=True, cmap='viridis')
#     plt.title(f'Heatmap for Param_C = {c}')
#     plt.xlabel('Parameter B')
#     plt.ylabel('Parameter A')
#     plt.show()
#
# from pandas.plotting import parallel_coordinates
#
# # 标准化数据或将其转换为分类数据
# df['Param_A'] = df['Param_A'].astype(str)
# df['Param_B'] = df['Param_B'].astype(str)
# df['Param_C'] = df['Param_C'].astype(str)
#
# plt.figure(figsize=(12, 6))
# parallel_coordinates(df, 'Value', colormap='viridis')
# plt.title('Parallel Coordinates Plot')
# plt.xlabel('Parameters')
# plt.ylabel('Values')
# plt.show()
#
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from scipy.interpolate import griddata
# import numpy as np
#
# # 示例数据
# np.random.seed(0)
# param_A = np.random.randint(0, 5, 100)
# param_B = np.random.randint(0, 5, 100)
# param_C = np.random.randint(0, 5, 100)
# values = np.random.rand(100)  # 假设的评估值
#
# # 创建网格数据
# grid_x, grid_y = np.mgrid[0:4:100j, 0:4:100j]
# grid_z = griddata((param_A, param_B), values, (grid_x, grid_y), method='cubic')
#
# # 创建三维曲面图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
#
# # 曲面图
# surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='viridis', edgecolor='none')
#
# # 添加颜色条
# fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
#
# # 设置轴标签
# ax.set_xlabel('Parameter A')
# ax.set_ylabel('Parameter B')
# ax.set_zlabel('Values')
#
# # 显示图形
# plt.show()

flip_data = {
    'Assist17': {
        0.2: 90.14,
        0.15: 90.17,
        0.1: 90.12,
        0.05: 89.94,
    },
    'EdNet-1': {
        0.2: 74.87,
        0.15: 74.88,
        0.1: 74.85,
        0.05: 74.81,
    },
    'Junyi': {
        0.2: 81.35,
        0.15: 81.37,
        0.1: 81.32,
        0.05: 81.44,
    },
    "XES3G5M": {
        0.2: 80.35,
        0.15: 80.39,
        0.1: 80.32,
        0.05: 80.22,
    }
}

weight_data = {
    'Assist17': {
        1e-1: 90.14,
        1e-2: 90.17,
        1e-3: 89.98,
        1e-4: 89.95,
    },
    'EdNet-1': {
        1e-1: 74.35,
        1e-2: 74.81,
        1e-3: 74.81,
        1e-4: 74.80,
    },
    'Junyi': {
        1e-1: 80.61,
        1e-2: 81.20,
        1e-3: 81.19,
        1e-4: 81.37,
    },
    "XES3G5M": {
        1e-1: 76.96,
        1e-2: 79.51,
        1e-3: 80.39,
        1e-4: 89.35,
    }
}

temp_data = {
    'Assist17': {
        0.1: 90.14,
        0.5: 90.08,
        1.0: 90.02,
        3.0: 89.83,
        5.0: 89.70
    },
    'EdNet-1': {
        0.1: 74.79,
        0.5: 74.81,
        1.0: 74.81,
        3.0: 74.81,
        5.0: 74.81
    },
    'Junyi': {
        0.1: 80.74,
        0.5: 81.14,
        1.0: 81.18,
        3.0: 81.26,
        5.0: 81.37
    },
    "XES3G5M": {
        0.1: 80.34,
        0.5: 80.35,
        1.0: 80.35,
        3.0: 80.39,
        5.0: 80.35
    }
}
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams

fontsize=35
rcParams['font.family'] = 'Times New Roman'
for dataset, flips in flip_data.items():
    flip_ratios = list(flips.keys())
    auc_scores = list(flips.values())

    plt.figure(figsize=(10, 6))  # 为每个数据集创建一个新图形
    plt.plot(flip_ratios, auc_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel(r'$p_f$', fontsize=fontsize)
    plt.ylabel('AUC (%)', fontsize=fontsize)
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(5))  # 限制Y轴的刻度最多为五个值
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{dataset}_flip.pdf', dpi=1200, bbox_inches='tight')
    plt.show()


for dataset, flips in weight_data.items():
    flip_ratios = list(flips.keys())
    auc_scores = list(flips.values())

    plt.figure(figsize=(10, 6))  # 为每个数据集创建一个新图形
    plt.plot(flip_ratios, auc_scores, marker='o', linestyle='-', color='blue')
    plt.xlabel(r'$\lambda_{reg}$', fontsize=fontsize)
    plt.ylabel('AUC (%)', fontsize=fontsize)
    plt.grid(True)
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(5))  # 限制Y轴的刻度最多为五个值
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))
    plt.yticks(fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.savefig(f'{dataset}_reg.pdf', dpi=1200, bbox_inches='tight')
    plt.show()

for dataset, temps in temp_data.items():
    temp_ratios = list(temps.keys())  # Extracting temperature values
    auc_scores = list(temps.values())  # Extracting AUC scores

    plt.figure(figsize=(10, 6))  # Setting figure size
    plt.plot(temp_ratios, auc_scores, marker='o', linestyle='-', color='blue')  # Plotting the line graph
    plt.xlabel(r'$\tau$', fontsize=fontsize)  # Setting x-axis label with font size
    plt.ylabel('AUC (%)', fontsize=fontsize)  # Setting y-axis label with font size
    plt.grid(True)  # Enabling grid

    ax = plt.gca()  # Getting current axis
    ax.set_xticks(temp_ratios)  # Setting x-axis ticks to temperature values
    ax.yaxis.set_major_locator(MaxNLocator(5))  # Limiting y-axis to a maximum of 5 ticks
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}'))  # Formatting y-axis labels
    plt.xticks(fontsize=fontsize)  # Setting font size for x-axis ticks
    plt.yticks(fontsize=fontsize)
    plt.savefig(f'{dataset}_temp.pdf', dpi=1200, bbox_inches='tight')
    plt.show()