import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib as mpl
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

def load_dataI():
    MPCdata = np.loadtxt("results/caseI/MPC_data.csv", delimiter=',', skiprows=1)
    time = MPCdata[:, 0]
    comp_power_MPC = MPCdata[:, 1]
    temp_MPC = MPCdata[:, 2]
    SOH_loss_MPC = MPCdata[:, 3]

    RBdata = np.loadtxt("results/caseI/RB_data.csv", delimiter=',', skiprows=1)
    comp_power_RB = RBdata[:, 1]
    temp_RB = RBdata[:, 2]
    SOH_loss_RB = RBdata[:, 3]
    return time, comp_power_MPC, temp_MPC, SOH_loss_MPC, comp_power_RB, temp_RB, SOH_loss_RB

def load_dataII():
    MPCdata = np.loadtxt("results/caseII/MPC_data.csv", delimiter=',', skiprows=1)
    time = MPCdata[:, 0]
    comp_power_MPC = MPCdata[:, 1]
    temp_MPC = MPCdata[:, 2]
    SOH_loss_MPC = MPCdata[:, 3]

    RBdata = np.loadtxt("results/caseII/RB_data.csv", delimiter=',', skiprows=1)
    comp_power_RB = RBdata[:, 1]
    temp_RB = RBdata[:, 2]
    SOH_loss_RB = RBdata[:, 3]
    return time, comp_power_MPC, temp_MPC, SOH_loss_MPC, comp_power_RB, temp_RB, SOH_loss_RB



import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
def statics_temp(time, temp_MPC, temp_RB):
    max_temp_MPC = max(temp_MPC)
    min_temp_MPC = min(temp_MPC)
    avg_temp_MPC = np.mean(temp_MPC)
    delta_temp_MPC = max_temp_MPC - min_temp_MPC
    max_temp_RB = max(temp_RB)
    min_temp_RB = min(temp_RB)
    avg_temp_RB = np.mean(temp_RB)
    delta_temp_RB = max_temp_RB - min_temp_RB
    print(f"max_temp_MPC: {max_temp_MPC}, min_temp_MPC: {min_temp_MPC}, avg_temp_MPC: {avg_temp_MPC}, delta_temp_MPC: {delta_temp_MPC}")
    print(f"max_temp_RB: {max_temp_RB}, min_temp_RB: {min_temp_RB}, avg_temp_RB: {avg_temp_RB}, delta_temp_RB: {delta_temp_RB}")

def plot_temp(time, temp_MPC, temp_RB, save_path='plot/caseII/temp.png'):
    fig = plt.figure(figsize=(4, 2), dpi=200)
    custom_lines = []
    # 统一风格设置（如已有可省略）
    colors = ['#D55E00', '#0072B2']
    markers = ['o', 'v', 'D', 'p', 's', '^']
    line_width = 1
    marker_size = 1.5
    legend_fontsize = 6
    # 绘制 Rule-based 控制温度曲线
    plt.plot(
        time,
        temp_RB,
        linestyle='--',
        color=colors[0],
        linewidth=line_width,
        marker=markers[0],
        markersize=marker_size,
        markevery=max(len(time)//20, 1),
        label='Rule-based (RB)'
    )
    custom_lines.append(Line2D([0], [0], color=colors[0], linestyle='--',
                               linewidth=line_width, marker=markers[0], markersize=marker_size))

    # 绘制 MPC 控制温度曲线
    plt.plot(
        time,
        temp_MPC,
        linestyle='-',
        color=colors[1],
        linewidth=line_width,
        marker=markers[1],
        markersize=marker_size,
        markevery=max(len(time)//20, 1),
        label='Model Predictive Control (MPC)'
    )
    custom_lines.append(Line2D([0], [0], color=colors[1], linestyle='-',
                               linewidth=line_width, marker=markers[1], markersize=marker_size))

    # 坐标轴与标题
    plt.xlabel('Time (s)')
    plt.ylabel('Battery Temperature (°C)')
    #plt.title('Battery Temperature Comparison', fontsize=8)

    # 图例设置
    plt.legend(custom_lines, ['Rule-based (RB)', 'MPC'], loc='upper right', frameon=False, fontsize=legend_fontsize)

    # 网格 & 轴格式
    plt.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(600))  # 每 10 分钟一格（假设 time 单位是秒）

    # Y轴自动范围（留余量）
    y_min = min(min(temp_MPC), min(temp_RB)) - 0.3
    y_max = max(max(temp_MPC), max(temp_RB)) + 0.7
    plt.ylim(y_min, y_max)

    ax = plt.gca()  # 获取当前坐标轴
    # 启用小刻度
    ax.minorticks_on()

    # 布局与保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

def plot_comp_power(time, comp_power_MPC, comp_power_RB, save_path='plot/caseI/comp_power.png'):
    # 样式设置
    colors = ['#0072B2', '#D55E00']
    markers = ['o', 'v']
    line_width = 1
    marker_size = 0
    legend_fontsize = 6

    # 设置字体与坐标轴风格
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.0

    fig = plt.figure(figsize=(4, 2), dpi=200)
    custom_lines = []

    # 绘制 MPC 曲线
    plt.plot(
        time,
        comp_power_MPC,
        label='Model Predictive Control (MPC)',
        linestyle='-',
        color=colors[0],
        linewidth=line_width,
        marker=markers[0],
        markersize=marker_size,
        markevery=max(len(time)//20, 1)
    )
    custom_lines.append(Line2D([0], [0], color=colors[0], linestyle='-',
                               linewidth=line_width, marker=markers[0], markersize=marker_size))

    # 绘制 RB 曲线
    plt.plot(
        time,
        comp_power_RB,
        label='Rule-based (RB)',
        linestyle='--',
        color=colors[1],
        linewidth=line_width,
        marker=markers[1],
        markersize=marker_size,
        markevery=max(len(time)//20, 1)
    )
    custom_lines.append(Line2D([0], [0], color=colors[1], linestyle='--',
                               linewidth=line_width, marker=markers[1], markersize=marker_size))

    # 图例设置
    plt.legend(custom_lines, ['MPC', 'RB'], loc='upper right', frameon=False, fontsize=legend_fontsize, ncol=2)
    
    # 坐标轴设置
    plt.xlabel('Time (s)', fontsize=8)
    plt.ylabel('Compressor Power (W)', fontsize=8)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)
    plt.ylim(0, 3100)
    plt.xlim(time[0], time[-1])
    plt.grid(True, linestyle=':', linewidth=0.4, alpha=0.7)

    ax = plt.gca()  # 获取当前坐标轴
    # 启用小刻度
    ax.minorticks_on()

    # 紧凑布局与保存
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.show()

def delta_comp_power(comp_power_MPC):
    delta_comp_power = [abs(comp_power_MPC[i] - comp_power_MPC[i-1]) for i in range(1, len(comp_power_MPC))]
    print(f"delta_comp_power: {max(delta_comp_power)}")
    # 打印正数中的中位数
    print(f"正数中的中位数: {np.median(delta_comp_power)}")
    return delta_comp_power

def calculate_comp_power(comp_power_MPC, comp_power_RB):
    sum_of_comp_power_MPC = sum(comp_power_MPC)
    sum_of_comp_power_RB = sum(comp_power_RB)
    print(f"sum_of_comp_power_MPC: {sum_of_comp_power_MPC/3600} Wh")
    print(f"sum_of_comp_power_RB: {sum_of_comp_power_RB/3600} Wh")
    percent_of_comp_power_MPC = abs(sum_of_comp_power_MPC - sum_of_comp_power_RB) / sum_of_comp_power_RB
    print(f"percent_of_comp_power_MPC: {percent_of_comp_power_MPC}")
    return sum_of_comp_power_MPC, sum_of_comp_power_RB

def calculate_sum_SOH_loss(SOH_loss_MPC, SOH_loss_RB):
    sum_of_SOH_loss_MPC = sum(SOH_loss_MPC)
    sum_of_SOH_loss_RB = sum(SOH_loss_RB)
    print(f"sum_of_SOH_loss_MPC: {-sum_of_SOH_loss_MPC*1000} * 10^-3")
    print(f"sum_of_SOH_loss_RB: {-sum_of_SOH_loss_RB*1000} * 10^-3")
    percent_of_SOH_loss_MPC = abs(sum_of_SOH_loss_MPC - sum_of_SOH_loss_RB) / sum_of_SOH_loss_RB
    print(f"percent_of_SOH_loss_MPC: {percent_of_SOH_loss_MPC}")
    return -sum_of_SOH_loss_MPC, -sum_of_SOH_loss_RB

def calculate_efftive_SOH_loss(SOH_loss_MPC, SOH_loss_RB):
    # 提取其中非零的数
    effective_SOH_loss_MPC = [-SOH_loss_MPC[i] for i in range(len(SOH_loss_MPC)) if SOH_loss_MPC[i] != 0]
    effective_SOH_loss_RB = [-SOH_loss_RB[i] for i in range(len(SOH_loss_RB)) if SOH_loss_RB[i] != 0]
    # 计算每一步的累加
    effective_SOH_loss_MPC = [sum(effective_SOH_loss_MPC[:i]) for i in range(len(effective_SOH_loss_MPC))]
    effective_SOH_loss_RB = [sum(effective_SOH_loss_RB[:i]) for i in range(len(effective_SOH_loss_RB))]
    return effective_SOH_loss_MPC, effective_SOH_loss_RB


def plot_SOH_loss(time, SOH_loss_MPC_I, SOH_loss_RB_I, SOH_loss_MPC_II, SOH_loss_RB_II):
    # 提取有效值
    effective_SOH_loss_MPC_I, effective_SOH_loss_RB_I = calculate_efftive_SOH_loss(SOH_loss_MPC_I, SOH_loss_RB_I)
    effective_SOH_loss_MPC_II, effective_SOH_loss_RB_II = calculate_efftive_SOH_loss(SOH_loss_MPC_II, SOH_loss_RB_II)
    time_arange = [time[i] / 60 for i in range(0, len(time), 30)]  # 转为分钟

    # 美化配色与线型
    plt.figure(figsize=(6, 4), dpi=300)

    # Case I
    plt.plot(time_arange, effective_SOH_loss_MPC_I, label='Case I - MPC',
             linestyle='-', color='#0072B2', linewidth=0.8, marker='o', markersize=0.1)
    plt.plot(time_arange, effective_SOH_loss_RB_I, label='Case I - Rule-Based',
             linestyle='--', color='#56B4E9', linewidth=0.8, marker='s', markersize=0.1)

    # Case II
    plt.plot(time_arange, effective_SOH_loss_MPC_II, label='Case II - MPC',
             linestyle='-', color='#D55E00', linewidth=0.8, marker='^', markersize=0.1)
    plt.plot(time_arange, effective_SOH_loss_RB_II, label='Case II - Rule-Based',
             linestyle='--', color='#E69F00', linewidth=0.8, marker='v', markersize=0.1)

    # 标签与网格
    plt.xlabel('Time (min)', fontsize=10)
    plt.ylabel('SOH Loss', fontsize=10)
    plt.title('Comparison of SOH Loss Under Different Control Strategies', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # 图例
    plt.legend(fontsize=9, loc='upper left', frameon=False, ncol=2)

    # 布局与保存
    plt.tight_layout()
    plt.savefig("plot/SOH_loss_comparison.png", dpi=300)
    plt.show()

def plot_data():
    time_I, comp_power_MPC_I, temp_MPC_I, SOH_loss_MPC_I, comp_power_RB_I, temp_RB_I, SOH_loss_RB_I = load_dataI()
    time_II, comp_power_MPC_II, temp_MPC_II, SOH_loss_MPC_II, comp_power_RB_II, temp_RB_II, SOH_loss_RB_II = load_dataII()
    #statics_temp(time, temp_MPC, temp_RB)
    plot_temp(time_II, temp_MPC_II, temp_RB_II, save_path='plot/caseII/temp.png')
    plot_comp_power(time_II, comp_power_MPC_II, comp_power_RB_II, save_path='plot/caseII/comp_power.png')
    #plot_comparison(time, temp_MPC, temp_RB)
    #delta_comp_power_MPC = delta_comp_power(comp_power_MPC)
    #calculate_comp_power(comp_power_MPC, comp_power_RB)
    #calculate_SOH_loss(SOH_loss_MPC, SOH_loss_RB)
    #plot_comp_power_SOH_loss(time, comp_power_MPC, comp_power_RB, SOH_loss_MPC, SOH_loss_RB)
    #plot_SOH_loss(time_I, SOH_loss_MPC_I, SOH_loss_RB_I, SOH_loss_MPC_II, SOH_loss_RB_II)
if __name__ == "__main__":
    plot_data()