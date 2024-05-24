from sklearn.preprocessing import StandardScaler
import seaborn as sns
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib import rcParams
import numpy as np

rcParams['font.family'] = 'Times New Roman'


def plot_mas(Mastery_Level, Name=None):
    import torch
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 生成一个1700x102的随机矩阵作为示例数据
    np.random.seed(0)  # 保证示例数据的一致性
    matrix = torch.sigmoid(torch.tensor(Mastery_Level))

    # 使用Seaborn的heatmap函数绘制热力图
    plt.figure(figsize=(20, 10))  # 设置图形的尺寸
    heatmap = sns.heatmap(matrix, cmap='GnBu', vmin=0.3, vmax=0.7)
    plt.xticks([])  # 移除x轴刻度
    plt.yticks([])  # 移除y轴刻度
    plt.xlabel('')  # 移除x轴标签
    plt.ylabel('')  # 移除y轴标签
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=35)
    plt.savefig('viusal_Mas_{}.pdf'.format(Name), dpi=1200, bbox_inches='tight')
    plt.show()


def plot_tsne(datahub, Mastery_Level, name=None, legend=True, seed=2023):
    studentDict = {}
    studentScoreDict = {}
    for i in range(datahub.student_num):
        studentDict[i] = 0
        studentScoreDict[i] = 0
    for k in range(datahub['total'].shape[0]):
        stu = datahub['total'][k, 0]
        studentDict[stu] += 1
        score = datahub['total'][k, 2]
        studentScoreDict[stu] += score
    sorted_dict_stu = {k: v for k, v in sorted(studentDict.items(), key=lambda item: item[1], reverse=True) if v >= 50}
    choose_stu = list(sorted_dict_stu.keys())
    correct_rate_dict = {}
    for stu in choose_stu:
        correct_rate_dict[stu] = studentScoreDict[stu] / sorted_dict_stu[stu]

    X = Mastery_Level[choose_stu]
    y = list(correct_rate_dict.values())

    np.random.seed(seed)
    X_scaled = StandardScaler().fit_transform(X)
    k = len(set(y))

    tsne = TSNE(n_components=2, perplexity=30, random_state=seed, init='random', learning_rate=200.0)
    X_tsne = tsne.fit_transform(X_scaled)
    y = np.array(list(y))
    num_unique_elements = len(set(y))
    fontsize = 45
    fig, axs = plt.subplots(3, 1, figsize=(12, 22), gridspec_kw={'width_ratios': [1], 'height_ratios': [1.5, 4.0, 1]})
    for j in range(3):
        ax = axs[j]
        if j == 0:
            colors = sns.color_palette("Blues", n_colors=num_unique_elements)
            sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=colors, legend=False, ax=ax)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

        if j == 1:
            normalized_sne = X_tsne / np.linalg.norm(X_tsne, axis=1, keepdims=True)
            sns.kdeplot(x=normalized_sne[:, 0], y=normalized_sne[:, 1], cmap='Blues', shade=True, cbar=True, bw=0.05,
                        cbar_kws={'extend': 'both',
                                  'orientation': 'horizontal',
                                  'label': 'Density'}, ax=ax
                        )
            cax = plt.gcf().axes[-1]
            cax.tick_params(labelsize=30)
            cax.set_xlabel('Density', fontsize=fontsize)
            ax.set_xticks([-1.0, 0.0, 1.0])
            ax.set_xticklabels(['-1.0', '0.0', '1.0'], fontsize=fontsize)
            ax.set_yticks([-1, -0.5, 0, 0.5, 1])
            ax.set_yticklabels(['-1.0', '-0.5', '0.0', '0.5', '1.0'], fontsize=fontsize)
            ax.set_xlabel('Features', fontsize=fontsize)
            ax.set_ylabel('Features', fontsize=fontsize)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

        if j == 2:
            normalized_sne = X_tsne / np.linalg.norm(X_tsne, axis=1, keepdims=True)
            angles = np.arctan2(normalized_sne[:, 1], normalized_sne[:, 0])
            angles.sort()
            kde = gaussian_kde(angles)
            ax.plot(angles, kde(angles), color='#63b2ee', linewidth=4)
            ax.fill_between(angles, kde(angles), 0, color='#63b2ee', alpha=0.5)
            ax.set_xlabel('Angles', fontsize=fontsize)
            ax.set_ylabel('Density', fontsize=fontsize)
            ax.set_ylim([0.0, 0.4])
            ax.set_xlim([np.min(angles), np.max(angles)])
            ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
            ax.set_xticklabels([-3, -2, -1, 0, 1, 2, 3], fontsize=fontsize)
            ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4])
            ax.set_yticklabels([0.0, 0.1, 0.2, 0.3, 0.4], fontsize=fontsize)
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['right'].set_linewidth(2)
            ax.spines['top'].set_linewidth(2)

    plt.subplots_adjust(hspace=0.2, wspace=0.4)
    plt.savefig('viusal_{}.pdf'.format(name), dpi=1200, bbox_inches='tight')
    plt.show()
    return X_tsne


def plot_tsne_pure(datahub, Mastery_Level, name=None, legend=True, seed=2023):
    # 数据处理部分
    studentDict = {}
    studentScoreDict = {}
    for i in range(datahub.student_num):
        studentDict[i] = 0
        studentScoreDict[i] = 0
    for k in range(datahub['total'].shape[0]):
        stu = datahub['total'][k, 0]
        studentDict[stu] += 1
        score = datahub['total'][k, 2]
        studentScoreDict[stu] += score
    sorted_dict_stu = {k: v for k, v in sorted(studentDict.items(), key=lambda item: item[1], reverse=True) if v >= 50}
    choose_stu = list(sorted_dict_stu.keys())
    correct_rate_dict = {}
    for stu in choose_stu:
        correct_rate_dict[stu] = studentScoreDict[stu] / sorted_dict_stu[stu]

    X = Mastery_Level[choose_stu]
    y = list(correct_rate_dict.values())

    # TSNE处理和绘图
    np.random.seed(seed)
    X_scaled = StandardScaler().fit_transform(X)

    tsne = TSNE(n_components=2, perplexity=30, random_state=seed)
    X_tsne = tsne.fit_transform(X_scaled)
    y = np.array(list(y))
    num_unique_elements = len(set(y))

    # 绘制第一个图
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = sns.color_palette("Blues", n_colors=num_unique_elements)
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y, palette=colors, legend=False, ax=ax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)
    ax.spines['top'].set_linewidth(0)

    plt.savefig('viusal_{}.pdf'.format(name), dpi=1200, bbox_inches='tight')
    plt.show()
    return X_tsne