import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science','nature'])
from matplotlib.backends.backend_pdf import PdfPages

# 设置图像行列
fig, axs = plt.subplots(2, 3, figsize=(8, 6), dpi=150)

count = 0
color_list = ['#74AED4', '#7BDFF2', '#FBDD85', '#F46F43', '#CF3D3E']
color_list2 = [
    '#74AED4',  # 原始蓝色（误差小，深）
    '#A5CBE2',  # 更浅的蓝色
    '#CFE1EF',  # 非常浅的蓝色
    '#EAF3F9',  # 接近白色的蓝白
    '#FFFFFF'   # 白色（误差最大）
]
color_list3 = [
    '#3A7CA5',  # 更深蓝色（误差最小）
    '#74AED4',  # 原始蓝
    '#A5CBE2',  # 更浅蓝
    '#EAF3F9',  # 淡蓝白
    '#FFFFFF'   # 白色（误差最大）
]
color_list4 = [
    '#3A7CA5',  # 更深蓝色（误差最小）
    '#5B9BC9',  # 中间深浅过渡色
    '#74AED4',  # 原始蓝（中间）
    '#A5CBE2',  # 更浅蓝
    '#D2E4F1'   # 最浅蓝（误差最大，不用白色）
]
color_list5 = [
    '#1B4F72',  # 深海蓝：误差最小，最深
    '#3A7CA5',  # 深蓝（原深色）
    '#74AED4',  # 中蓝（原始蓝）
    '#B0D4E8',  # 淡蓝
    '#E2F0F9'   # 很浅蓝：误差最大
]
colors = plt.cm.colors.LinearSegmentedColormap.from_list('custom_cmap', color_list4, N=256)

data = 'XJTU'
batches = [0]
for batch in batches:
    pred_label = np.load('/Users/cruisin/Documents/BTM/SOH/results/XJTU/all_batches/best%20model/20250519_013045/pred_label.npy')
    true_label = np.load('/Users/cruisin/Documents/BTM/SOH/results/XJTU/all_batches/best%20model/20250519_013045/true_label.npy')
    # 计算MAPE
    from sklearn.metrics import mean_absolute_percentage_error
    mape = mean_absolute_percentage_error(true_label, pred_label)
    print(f'mape: {mape}')
    # 计算mse
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(true_label, pred_label)
    print(f'mse: {mse}')
    error = np.abs(pred_label - true_label)
    vmin, vmax = error.min(), error.max()
    vmax = 0.001
    """a1 = []
    a2 = []
    for i in range(len(pred_label)):
        error_a = abs(pred_label[i] - true_label[i])
        if error_a < 1:
            a1.append(pred_label[i]+0.0025)
            a2.append(true_label[i])
    b1 = []
    b2 = []
    for i in range(len(a1)):
        error_a = abs(a1[i] - a2[i])
        if error_a < 0.0005:
            b1.append(a1[i])
            b2.append(a2[i])
    pred_label = np.array(b1)
    true_label = np.array(b2)"""
    print(len(pred_label), len(true_label))
    error = np.abs(pred_label - true_label)
    print(f'max pred: {pred_label.max()}, min pred: {pred_label.min()}')
    print(f'max true: {true_label.max()}, min true: {true_label.min()}')

    # 自动设置坐标轴范围
    buffer = 0
    min_val = true_label.min() - buffer
    max_val = true_label.max() + buffer
    print(min_val, max_val)
    lims = (-0.002, 0.001)

    # 获取子图位置
    row = count // 3
    col = count % 3
    print(data, batch, row, col)
    ax = axs[row, col]

    # 画散点图
    sc = ax.scatter(true_label, pred_label, c=error, cmap=colors, s=0.3, alpha=0.7, vmin=vmin, vmax=vmax)

    # 对角线表示理想预测（即 prediction = truth）
    ax.plot(lims, lims, '--', c='#ff4d4e', linewidth=1)

    ax.set_aspect('equal')
    ax.set_xlabel('True SOH')
    ax.set_ylabel('Predicted SOH')

    # 设置坐标范围和刻度
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xticks(np.round(np.linspace(*lims, 5), 3))
    ax.set_yticks(np.round(np.linspace(*lims, 5), 3))

# 关闭右下角空子图
axs[1, 2].axis('off')

# 添加 colorbar 表示误差
cbar = fig.colorbar(
    plt.cm.ScalarMappable(cmap=colors, norm=plt.Normalize(vmin=vmin, vmax=0.001)),
    ax=axs[1, 2],
    label='Absolute error',
    fraction=0.46, pad=0.4
)

plt.tight_layout()
plt.show()