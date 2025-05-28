import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_RB_data(name):
    root = '/Users/cruisin/Documents/BTM/results/harbor-RB'
    for file in os.listdir(root):
        if file.endswith(name+'.csv'):
            df = pd.read_csv(os.path.join(root, file))
            num_of_cooling_system = int(name)
            comp_power = df['Comp_Power(W)'].cumsum()*num_of_cooling_system
            temp = df['Temperature(℃)']
            return comp_power, temp

def load_MPC_data(name):
    root = '/Users/cruisin/Documents/BTM/results/harbor'
    for file in os.listdir(root):
        if file.endswith(name+'.csv'):
            df = pd.read_csv(os.path.join(root, file))
            num_of_cooling_system = int(name)
            comp_power = df['Control_Sequence'].cumsum()*num_of_cooling_system
            temp = df['Temperature(℃)']
            return comp_power, temp

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scienceplots
plt.style.use(['science', 'nature'])

# 配色和样式设置
colors = ['#ffdf00', '#ff7f00', '#377eb8', '#4daf4a', '#e41a1c', '#984ea3']
markers = ['o', 'v', 'D', 'p', 's', '^']
line_width = 0.5
marker_size = 3.0
legend_fontsize = 6
default_legend_labels = ['5 modules', '10 modules', '20 modules', '25 modules', '50 modules', '100 modules']
#plt.rcParams['font.family'] = 'Times New Roman'
def plot_temp_data(*temp_data, legends=None, ylabel='Temperature (°C)', xlabel='Time Step (s)', title='Temperature Data'):
    # 设置字体
    #from matplotlib import mpl
    #mpl.rcParams['font.family'] = 'Times New Roman'
    fig = plt.figure(figsize=(4, 2), dpi=200)
    custom_lines = []

    for i, temp in enumerate(temp_data):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(
            temp,
            color=color,
            alpha=0.95,
            linewidth=line_width,
            linestyle='-',
            marker=marker,
            markersize=marker_size,
            markevery=max(len(temp)//20, 1)
        )
        custom_lines.append(Line2D([0], [0], color=color, linestyle='-', linewidth=line_width,
                                   marker=marker, markersize=marker_size))

    # 设置图例 放在左上边
    if legends is None:
        legends = default_legend_labels[:len(temp_data)]
    plt.legend(custom_lines, legends, loc='upper left', frameon=False, fontsize=legend_fontsize, ncol=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title, fontsize=8)
    #plt.ylim(24, 30)
    plt.tight_layout()
    plt.savefig('temp_data.png', dpi=600)
    plt.show()

def plot_comp_power_data(*comp_power_data, legends=None, ylabel='Compressor Energy (MWh)', xlabel='Time Step (s)', title='Compressor Power Data'):
    fig = plt.figure(figsize=(4, 2), dpi=200)
    custom_lines = []

    for i, comp_power in enumerate(comp_power_data):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        plt.plot(
            comp_power/1e6/3600,
            color=color,
            alpha=0.95,
            linewidth=line_width,
            linestyle='-',
            marker=marker,
            markersize=marker_size,
            markevery=max(len(comp_power)//20, 1)
        )
        custom_lines.append(Line2D([0], [0], color=color, linestyle='-', linewidth=line_width,
                                   marker=marker, markersize=marker_size))

    # 设置图例
    if legends is None:
        legends = default_legend_labels[:len(comp_power_data)]
    plt.legend(custom_lines, legends, loc='upper left', frameon=False, fontsize=legend_fontsize, ncol=2)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.title(title, fontsize=8)
    plt.tight_layout()
    plt.savefig('comp_power_data.png', dpi=600)
    plt.show()

def plot_comp_temp_percent(percent_power_RB_5, percent_temp_RB_5,
                           percent_power_RB_10, percent_temp_RB_10,
                           percent_power_RB_20, percent_temp_RB_20,
                           percent_power_RB_25, percent_temp_RB_25,
                           percent_power_RB_50, percent_temp_RB_50,
                           percent_power_RB_100, percent_temp_RB_100):

    # 数据准备
    modules = ['5', '10', '20', '25', '50', '100']
    x = np.arange(len(modules))  # x坐标 [0, 1, 2, 3, 4, 5]

    percent_comp_power = [
        percent_power_RB_5, percent_power_RB_10, percent_power_RB_20,
        percent_power_RB_25, percent_power_RB_50, percent_power_RB_100
    ]

    percent_temp = [
        percent_temp_RB_5, percent_temp_RB_10, percent_temp_RB_20,
        percent_temp_RB_25, percent_temp_RB_50, percent_temp_RB_100
    ]

    # 样式设置
    width = 0.35  # 每个柱子的宽度
    #colors = ['#377eb8', '#e41a1c']  # 蓝色和红色（Tableau风格）
    #colors = ['#E69F00', '#56B4E9']  # 橙色 + 天蓝色
    #colors = ['#4daf4a', '#984ea3']  # 绿色 + 紫色
    colors = ['#3498db', '#e74c3c']  # 湖蓝 + 番茄红
    fig, ax = plt.subplots(figsize=(4, 2), dpi=300)

    # 绘图
    bars1 = ax.bar(x - width/2, percent_comp_power, width, label='Total Comp Power', color=colors[0])
    bars2 = ax.bar(x + width/2, percent_temp, width, label='Max Temp Range', color=colors[1])

    # 标注
    ax.set_xlabel('Number of Modules')
    ax.set_ylabel('Percent')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m} modules' for m in modules])
    ax.legend(frameon=False, fontsize=7, loc='upper left')
    ax.set_ylim(0, max(max(percent_comp_power), max(percent_temp)) * 1.2)
    # set_ylim变成百分比的形式
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))

    # 添加百分比数值标签（可选）
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height*100:.0f}\%', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 1), textcoords="offset points",
                    ha='center', va='bottom', fontsize=5)

    plt.tight_layout()
    root = '/Users/cruisin/Documents/BTM/results'
    plt.savefig(os.path.join(root, 'comp_temp_percent.png'), dpi=600)
    plt.show()

def main():
    comp_powerRB5, tempRB5 = load_RB_data('5')
    comp_powerRB10, tempRB10 = load_RB_data('10')
    comp_powerRB20, tempRB20 = load_RB_data('20')
    comp_powerRB25, tempRB25 = load_RB_data('25')
    comp_powerRB50, tempRB50 = load_RB_data('50')
    comp_powerRB100, tempRB100 = load_RB_data('100')

    comp_powerMPC5, tempMPC5 = load_MPC_data('5')
    comp_powerMPC10, tempMPC10 = load_MPC_data('10')
    comp_powerMPC20, tempMPC20 = load_MPC_data('20')
    comp_powerMPC25, tempMPC25 = load_MPC_data('25')
    comp_powerMPC50, tempMPC50 = load_MPC_data('50')
    comp_powerMPC100, tempMPC100 = load_MPC_data('100')

    # 统计总功耗减少
    print(comp_powerRB5.shape)
    # 取最后一个值
    total_power_RB_5 = comp_powerRB5.iloc[-1]
    total_power_MPC_5 = comp_powerMPC5.iloc[-1]
    percent_power_RB_5 = (total_power_MPC_5) / total_power_RB_5
    print(f'5 modules: {percent_power_RB_5*100}%')
    total_power_RB_10 = comp_powerRB10.iloc[-1]
    total_power_MPC_10 = comp_powerMPC10.iloc[-1]
    percent_power_RB_10 = (total_power_MPC_10) / total_power_RB_10
    print(f'10 modules: {percent_power_RB_10*100}%')
    total_power_RB_20 = comp_powerRB20.iloc[-1]
    total_power_MPC_20 = comp_powerMPC20.iloc[-1]
    percent_power_RB_20 = (total_power_MPC_20) / total_power_RB_20
    print(f'20 modules: {percent_power_RB_20*100}%')
    total_power_RB_25 = comp_powerRB25.iloc[-1]
    total_power_MPC_25 = comp_powerMPC25.iloc[-1]
    percent_power_RB_25 = (total_power_MPC_25) / total_power_RB_25
    print(f'25 modules: {percent_power_RB_25*100}%')
    total_power_RB_50 = comp_powerRB50.iloc[-1]
    total_power_MPC_50 = comp_powerMPC50.iloc[-1]
    percent_power_RB_50 = (total_power_MPC_50) / total_power_RB_50
    print(f'50 modules: {percent_power_RB_50*100}%')
    total_power_RB_100 = comp_powerRB100.iloc[-1]
    total_power_MPC_100 = comp_powerMPC100.iloc[-1]
    percent_power_RB_100 = (total_power_MPC_100) / total_power_RB_100
    print(f'100 modules: {percent_power_RB_100*100}%')
    
    temp_range_RB_5 = tempRB5.max() - tempRB5.min()
    temp_range_MPC_5 = tempMPC5.max() - tempMPC5.min()
    percent_temp_RB_5 = (temp_range_MPC_5) / temp_range_RB_5
    print(f'5 modules: {percent_temp_RB_5*100}%')
    temp_range_RB_10 = tempRB10.max() - tempRB10.min()
    temp_range_MPC_10 = tempMPC10.max() - tempMPC10.min()
    percent_temp_RB_10 = (temp_range_MPC_10) / temp_range_RB_10
    print(f'10 modules: {percent_temp_RB_10*100}%')
    
    temp_range_RB_20 = tempRB20.max() - tempRB20.min()
    temp_range_MPC_20 = tempMPC20.max() - tempMPC20.min()
    percent_temp_RB_20 = (temp_range_MPC_20) / temp_range_RB_20
    print(f'20 modules: {percent_temp_RB_20*100}%')
    
    
    temp_range_RB_25 = tempRB25.max() - tempRB25.min()
    temp_range_MPC_25 = tempMPC25.max() - tempMPC25.min()
    percent_temp_RB_25 = (temp_range_MPC_25) / temp_range_RB_25
    print(f'25 modules: {percent_temp_RB_25*100}%')

    temp_range_RB_50 = tempRB50.max() - tempRB50.min()
    temp_range_MPC_50 = tempMPC50.max() - tempMPC50.min()
    percent_temp_RB_50 = (temp_range_MPC_50) / temp_range_RB_50
    print(f'50 modules: {percent_temp_RB_50*100}%')

    temp_range_RB_100 = tempRB100.max() - tempRB100.min()
    temp_range_MPC_100 = tempMPC100.max() - tempMPC100.min()
    percent_temp_RB_100 = (temp_range_MPC_100) / temp_range_RB_100
    print(f'100 modules: {percent_temp_RB_100*100}%')

    plot_comp_temp_percent(percent_power_RB_5, percent_temp_RB_5, percent_power_RB_10, percent_temp_RB_10, percent_power_RB_20, percent_temp_RB_20, percent_power_RB_25, percent_temp_RB_25, percent_power_RB_50, percent_temp_RB_50, percent_power_RB_100, percent_temp_RB_100)
    
    #plot_temp_data(tempRB5, tempRB10, tempRB20, tempRB25, tempRB50, tempRB100)
    #plot_temp_data(tempMPC5, tempMPC10, tempMPC20, tempMPC25, tempMPC50, tempMPC100)
    #plot_comp_power_data(comp_powerRB5, comp_powerRB10, comp_powerRB20, comp_powerRB25, comp_powerRB50, comp_powerRB100)
    #plot_comp_power_data(comp_powerMPC5, comp_powerMPC10, comp_powerMPC20, comp_powerMPC25, comp_powerMPC50, comp_powerMPC100)
if __name__ == '__main__':
    main()
