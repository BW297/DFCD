import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_mnd(mastery_level):
    row_sums = np.sum(mastery_level ** 2, axis=1)
    sum_square_diff = row_sums[:, np.newaxis] + row_sums - 2 * np.dot(mastery_level, mastery_level.T)
    sum_square_diff = np.maximum(sum_square_diff, 0)
    rmse = np.sqrt(sum_square_diff / mastery_level.shape[1])
    return rmse


def plot_mnd_map(mastery_level):
    def softmax(x):
        """计算矩阵的softmax"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    mnd_map = get_mnd(mastery_level=mastery_level)
    plt.figure(figsize=(12, 10))
    plt.xticks([])  # 移除x轴刻度
    plt.yticks([])  # 移除y轴刻度
    heatmap = plt.imshow(softmax(mnd_map), cmap='GnBu', aspect='auto', vmin=0, vmax=1)
    cbar = plt.colorbar(heatmap)  # 添加颜色条
    cbar.ax.tick_params(labelsize=14)  # Set font size for colorbar
    cbar.set_label('Mean Normalized Difference', labelpad=10, fontsize=30, rotation=270)
    # cbar.set_ticks([0, 1])
    # cbar.set_ticklabels(['0', '1'], fontsize=25)
    plt.show()



