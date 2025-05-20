import numpy as np
import matplotlib.pyplot as plt

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
    plt.savefig("plot/caseII/temp.png", dpi=300)
    plt.show()

def plot_comp_power(time, comp_power_MPC, comp_power_RB):
    plt.figure(figsize=(6, 4), dpi=300)
    
    # 设置字体
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴线宽
    
    # 绘制曲线
    plt.plot(time, comp_power_MPC, label='MPC', linestyle='-', color='#0072B2', linewidth=2.2)
    plt.plot(time, comp_power_RB, label='RB', linestyle='--', color='#D55E00', linewidth=2.2)

    # 坐标轴标签
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Compressor Power (W)', fontsize=12)
    
    # 坐标轴范围和刻度
    plt.ylim(0, 3100)
    plt.xlim(time[0], time[-1])
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)

    # 网格线
    plt.grid(which='major', linestyle=':', linewidth=0.6, alpha=0.8)

    # 去掉顶部和右侧边框
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 图例
    plt.legend(loc='upper right', fontsize=10, frameon=False)

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    plt.savefig("plot/caseII/comp_power.png", dpi=600, bbox_inches='tight')

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
    #plot_temp(time, temp_MPC, temp_RB)
    #plot_comp_power(time, comp_power_MPC, comp_power_RB)
    #plot_comparison(time, temp_MPC, temp_RB)
    #delta_comp_power_MPC = delta_comp_power(comp_power_MPC)
    #calculate_comp_power(comp_power_MPC, comp_power_RB)
    #calculate_SOH_loss(SOH_loss_MPC, SOH_loss_RB)
    #plot_comp_power_SOH_loss(time, comp_power_MPC, comp_power_RB, SOH_loss_MPC, SOH_loss_RB)
    plot_SOH_loss(time_I, SOH_loss_MPC_I, SOH_loss_RB_I, SOH_loss_MPC_II, SOH_loss_RB_II)
if __name__ == "__main__":
    plot_data()