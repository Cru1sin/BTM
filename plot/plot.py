import numpy as np
import matplotlib.pyplot as plt

def load_data():
    MPCdata = np.loadtxt("results/MPC_data.csv", delimiter=',', skiprows=1)
    time = MPCdata[:, 0]
    comp_power_MPC = MPCdata[:, 1]
    temp_MPC = MPCdata[:, 2]
    SOH_loss_MPC = MPCdata[:, 3]

    RBdata = np.loadtxt("results/RB_data.csv", delimiter=',', skiprows=1)
    comp_power_RB = RBdata[:, 1]
    temp_RB = RBdata[:, 2]
    SOH_loss_RB = RBdata[:, 3]
    return time, comp_power_MPC, temp_MPC, SOH_loss_MPC, comp_power_RB, temp_RB, SOH_loss_RB

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def plot_temp(time, temp_MPC, temp_RB):
    plt.figure(figsize=(6, 4), dpi=300)  # 高分辨率，适合论文
    plt.rcParams['font.family'] = 'Times New Roman'

    # 曲线绘制
    plt.plot(time, temp_RB, label='Rule-based (RB)', linestyle='--', color='#D55E00', linewidth=2.0)
    plt.plot(time, temp_MPC, label='Model Predictive Control (MPC)', linestyle='-', color='#0072B2', linewidth=2.0)

    # 坐标轴标签
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Battery Temperature (°C)', fontsize=12)

    # 坐标刻度
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 网格线
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

    # 边框设置
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    # 图例设置
    plt.legend(fontsize=10, loc='upper right', frameon=False)

    # Y轴范围自动适应加小余量，防止曲线压边
    y_min = min(min(temp_MPC), min(temp_RB)) - 0.3
    y_max = max(max(temp_MPC), max(temp_RB)) + 1
    plt.ylim(y_min, y_max)

    # 格式控制器，让 x 轴以千为单位显示（更专业）
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(600))  # 每10分钟一个刻度
    plt.tight_layout()
    plt.savefig("plot/temp.png", dpi=300)
    plt.show()
    
def plot_comparison(time, temp_MPC, temp_RB):
    import matplotlib.pyplot as plt

    # 计算统计值
    metrics_MPC = [np.max(temp_MPC), np.min(temp_MPC), np.mean(temp_MPC)]
    metrics_RB = [np.max(temp_RB), np.min(temp_RB), np.mean(temp_RB)]

    labels = ['T_max (°C)', 'T_min (°C)', 'T_avg (°C)']
    x = np.arange(len(labels))  # [0, 1, 2]
    width = 0.35

    # 创建图形
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    # 柱状图
    bar1 = ax.bar(x - width/2, metrics_MPC, width, label='MPC', color='#0072B2')
    bar2 = ax.bar(x + width/2, metrics_RB, width, label='RB', color='#D55E00')

    # 添加数值标签
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 向上偏移3
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    add_labels(bar1)
    add_labels(bar2)

    # 设置标签和样式
    ax.set_ylabel('Temperature (℃)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    # 保存图像
    plt.savefig('temperature_metrics_comparison.pdf', dpi=300, bbox_inches='tight')
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
    print(f"sum_of_comp_power_MPC: {sum_of_comp_power_MPC}")
    print(f"sum_of_comp_power_RB: {sum_of_comp_power_RB}")
    percent_of_comp_power_MPC = abs(sum_of_comp_power_MPC - sum_of_comp_power_RB) / sum_of_comp_power_RB
    print(f"percent_of_comp_power_MPC: {percent_of_comp_power_MPC}")
    return sum_of_comp_power_MPC, sum_of_comp_power_RB

def calculate_SOH_loss(SOH_loss_MPC, SOH_loss_RB):
    sum_of_SOH_loss_MPC = sum(SOH_loss_MPC)
    sum_of_SOH_loss_RB = sum(SOH_loss_RB)
    print(f"sum_of_SOH_loss_MPC: {sum_of_SOH_loss_MPC}")
    print(f"sum_of_SOH_loss_RB: {sum_of_SOH_loss_RB}")
    percent_of_SOH_loss_MPC = abs(sum_of_SOH_loss_MPC - sum_of_SOH_loss_RB) / sum_of_SOH_loss_RB
    print(f"percent_of_SOH_loss_MPC: {percent_of_SOH_loss_MPC}")
    return -sum_of_SOH_loss_MPC, -sum_of_SOH_loss_RB

def plot_comp_power_SOH_loss(time, comp_power_MPC, comp_power_RB, SOH_loss_MPC, SOH_loss_RB):
    sum_of_comp_power_MPC, sum_of_comp_power_RB = calculate_comp_power(comp_power_MPC, comp_power_RB)
    sum_of_SOH_loss_MPC, sum_of_SOH_loss_RB = calculate_SOH_loss(SOH_loss_MPC, SOH_loss_RB)
    
    total_power = [sum_of_comp_power_MPC, sum_of_comp_power_RB]
    total_soh = [sum_of_SOH_loss_MPC, sum_of_SOH_loss_RB]

    # 指标组
    groups = ['Compressor Power (W·s)', 'SOH Loss (%)']
    x = np.arange(len(groups))  # [0, 1]

    # 每组中的两种方法
    mpc_values = [total_power[0], total_soh[0]]
    rb_values = [total_power[1], total_soh[1]]

    labels = ['MPC', 'RB']
    x = np.arange(len(labels))  # [0, 1]
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 6))

    # 左侧压缩机功率柱状图（主轴）
    color_power = '#1f77b4'
    bar1 = ax1.bar(x - width/2, total_power, width,
                   label='Compressor Power (W·s)', color=color_power,
                   edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Compressor Power (W·s)', fontsize=13, color=color_power)
    ax1.tick_params(axis='y', labelcolor=color_power)

    # 添加数值标签
    for rect in bar1:
        height = rect.get_height()
        ax1.text(rect.get_x() + rect.get_width()/2, height * 1.01,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=11, color=color_power)

    # 创建右侧Y轴（SOH loss）
    ax2 = ax1.twinx()
    color_soh = '#ff7f0e'
    bar2 = ax2.bar(x + width/2, total_soh, width,
                   label='SOH Loss (%)', color=color_soh,
                   edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('SOH Loss (%)', fontsize=13, color=color_soh)
    ax2.tick_params(axis='y', labelcolor=color_soh)

    # 添加数值标签
    for rect in bar2:
        height = rect.get_height()
        ax2.text(rect.get_x() + rect.get_width()/2, height * 1.01,
                 f'{height:.4f}', ha='center', va='bottom', fontsize=11, color=color_soh)

    # 设置x轴
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=12)
    ax1.set_title('Comparison of MPC and RB: Compressor Power vs SOH Loss',
                  fontsize=14, weight='bold')

    # 网格与布局美化
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    fig.tight_layout()

    # 图例合并
    # 图例合并
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles1 + handles2, labels1 + labels2, loc='upper right', fontsize=11)

    plt.show()

def plot_data():
    time, comp_power_MPC, temp_MPC, SOH_loss_MPC, comp_power_RB, temp_RB, SOH_loss_RB = load_data()
    #plot_temp(time, temp_MPC, temp_RB)
    #plot_comparison(time, temp_MPC, temp_RB)
    #delta_comp_power_MPC = delta_comp_power(comp_power_MPC)
    #calculate_comp_power(comp_power_MPC, comp_power_RB)
    #calculate_SOH_loss(SOH_loss_MPC, SOH_loss_RB)
    plot_comp_power_SOH_loss(time, comp_power_MPC, comp_power_RB, SOH_loss_MPC, SOH_loss_RB)
if __name__ == "__main__":
    plot_data()