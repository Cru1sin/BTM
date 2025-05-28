import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
import numpy as np
# 配置字体、全局风格
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.size'] = 8
mpl.rcParams['axes.labelsize'] = 8
mpl.rcParams['axes.titlesize'] = 9
mpl.rcParams['legend.fontsize'] = 6
mpl.rcParams['xtick.labelsize'] = 7
mpl.rcParams['ytick.labelsize'] = 7

# 配色和样式设置
colors = ['#ff7f00', '#377eb8', '#4daf4a', '#e41a1c']
# 颜色分别为：橙色、蓝色、绿色、红色
markers = ['o', 'v', 'D', 'p', 's', '^']
line_width = 1.0
marker_size = 4.0

def plot_wind_power_data(save_path=None):
    from EnergyStorageSystem.TargetPower import WIND
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=200)
    colors = ['#377eb8', '#4daf4a', '#e41a1c', '#984ea3', '#ff7f00', '#ffdf00']
    line_width = 1.2
    time = np.arange(0, 24, 1)
    ax.plot(time, WIND, label='Wind Power', color=colors[0], linewidth=line_width)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (W)')
    #ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.7)
    ax.legend(loc='upper right', frameon=False)
    ax.set_ylim(0, max(WIND) * 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()

def plot_power_data(time, noisy_power, clean_power, threshold=0.1423, title='Power Data', save_path=None):
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=200)
    colors = ['#377eb8', '#4daf4a', '#e41a1c', '#984ea3', '#ff7f00', '#ffdf00']
    line_width = 0.8
    ax.plot(time, noisy_power, label='Power with Noise', color=colors[-2], linewidth=line_width)
    ax.plot(time, clean_power, label='Power without Noise', color=colors[0], linewidth=line_width)
    ax.axhline(y=threshold, color=colors[2], linestyle='--', linewidth=1.0, label='Electrolyzer Power')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (MW)')
    #ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.7)
    ax.legend(loc='upper right', frameon=False)
    ax.set_ylim(0, max(max(noisy_power), max(clean_power), threshold) * 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()

if __name__ == '__main__':
    from EnergyStorageSystem.TargetPower import P_RE_SECOND, POWER
    plot_power_data(np.arange(0, 86400, 1), POWER/1e6, P_RE_SECOND/1e6, title='Power Data', save_path='plot/power_data.png')
    #plot_wind_power_data(save_path='plot/wind_power_data.png')
