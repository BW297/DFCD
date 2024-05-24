import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

rcParams['font.family'] = 'Times New Roman'
# 运行时间数据
ncdm_speed = {
    'NCDM': 12,
    'ORCDF': 21,
    'RCD': 360,
    'LightGCN': 72,
}
ncdm_auc= {
    'NCDM': 86.89,
    'ORCDF': 89.94,
    'RCD': 88.35,
    'LightGCN': 88.73,
}

# 计算相对速度值，以NCDM为1.0
relative_speed = {method: speed / ncdm_speed['NCDM'] for method, speed in ncdm_speed.items()}
fontsize = 30
markers = ['o', '^', 's', 'D']  # 圆形、三角形、方形、菱形
# 创建散点图
plt.figure(figsize=(10, 6))
for i, method in enumerate(ncdm_speed.keys()):
    plt.scatter(relative_speed[method], ncdm_auc[method], label=method, marker=markers[i], s=100)

        # 在每个点旁边标注方法名，调整文本位置以避免出框
    if i == 0:
        plt.text(relative_speed[method], ncdm_auc[method] + 0.1, method, fontsize=20)
    elif i == 2:
        plt.text(relative_speed[method] - 2, ncdm_auc[method] + 0.1, method, fontsize=20)
    elif i == 3:
        plt.text(relative_speed[method], ncdm_auc[method] + 0.1, method, fontsize=20)
    else:
        plt.text(relative_speed[method], ncdm_auc[method], method, fontsize=20)

# 设置坐标轴标签
plt.xlabel('Relative Training Speed', fontsize=fontsize)
plt.ylabel('AUC (%)', fontsize=fontsize)
plt.xticks(fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.grid(True)
plt.savefig('time.pdf', dpi=1200, bbox_inches='tight')
plt.show()