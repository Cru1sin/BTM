import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

wind_root = '/Users/cruisin/Documents/BTM/EnergyStorageSystem/风、光、储、氢、岸电数据/wind.xlsx'
solar_root = '/Users/cruisin/Documents/BTM/EnergyStorageSystem/风、光、储、氢、岸电数据/solar.xls'

wind_data = pd.read_excel(wind_root)
solar_data = pd.read_excel(solar_root)

# 风电功率
wind_power = wind_data['10min内平均功率（MW）']

solar_power = solar_data['5min内平均发电功率（MW）']


print(solar_power.shape)
print(wind_power.shape)



# 插值风电到每5min一次（将147点 → 288点）
wind_power = np.interp(np.linspace(0, len(wind_power)-1, 288), np.arange(len(wind_power)), wind_power)

print(wind_power.shape)
print(solar_power.shape)
# 构造时间轴（294个5min间隔的点）
time = [f'{int(i/12):02d}:{int((i%12)*5):02d}' for i in range(288)]
total_power = solar_power[0:288] + wind_power[0:288]



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

def plot_solar_wind_power(time, solar_power, wind_power, total_power, electrolyzer_power=2.3, save_path=None):
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=200)

    ax.plot(time, solar_power, label='Solar Power', color=colors[0], linewidth=line_width)
    ax.plot(time, wind_power, label='Wind Power', color=colors[1], linewidth=line_width)
    ax.plot(time, total_power, label='Total Power', color=colors[2], linewidth=line_width)
    ax.axhline(y=electrolyzer_power, color=colors[3], linestyle='--', linewidth=1.0, label='Electrolyzer Power')

    ax.set_xlabel('Time (hh:mm)')
    ax.set_ylabel('Power (MW)')
    #ax.set_title('Power Output of Solar and Wind Energy in One Day')
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.6)

    # 设置 x 轴每 2 小时一个刻度（假设时间步长为 5 分钟，则每小时 12 个点）
    tick_interval = 24  # 每 2 小时
    xticks = np.arange(0, len(time), tick_interval)
    ax.set_xticks(xticks)
    ax.set_xticklabels([time[i] for i in xticks], rotation=45)

    ax.legend(loc='upper left', frameon=False)
    ax.set_ylim(-1, max(max(solar_power), max(wind_power), max(total_power), electrolyzer_power) * 1.05)
    ax.minorticks_on()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()

plot_solar_wind_power(time, solar_power[0:288], wind_power[0:288], total_power[0:288], electrolyzer_power=2.3, save_path='plot/solar_wind_power.png')



# 选择8～9小时的数据
total_power = total_power[96:110]
original_time = np.arange(0, 14*300, 300)
# 构造目标时间轴（每秒一个点）
target_time = np.arange(0, 13*300, 1)  # 3600 个点，1个小时
print(f'total_power.shape: {total_power.shape}, original_time.shape: {original_time.shape}, target_time.shape: {target_time.shape}')

# 创建插值函数（线性插值）
interp_func = interp1d(original_time, total_power, kind='linear')

# 得到每秒钟的功率数据
power_per_second = interp_func(target_time)
std_dev =  1* np.max(power_per_second)  # 例如波动范围约 ±20%
noise = np.random.normal(0, std_dev, size=power_per_second.shape)
from scipy.ndimage import gaussian_filter1d
smoothed_noise = gaussian_filter1d(noise, sigma=60)  # 控制波动频率
print(power_per_second)
noisy_power = power_per_second + smoothed_noise
POWER = np.clip(noisy_power, 0, None)  # 防止出现负功率
# MW -> W
#POWER = (2.5 - POWER) * 1e6 / 100
print(POWER)

colors = ['#377eb8', '#4daf4a', '#e41a1c', '#984ea3', '#ff7f00', '#ffdf00']
def plot_power_data(time, noisy_power, clean_power, threshold=2.3, title='Power Data', save_path=None):
    fig, ax = plt.subplots(figsize=(5, 2.5), dpi=200)

    ax.plot(time, noisy_power, label='Power with Noise', color=colors[0], linewidth=line_width)
    ax.plot(time, clean_power, label='Power without Noise', color=colors[1], linewidth=line_width)
    ax.axhline(y=threshold, color=colors[2], linestyle='--', linewidth=1.0, label='Electrolyzer Power')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Power (MW)')
    #ax.set_title(title)
    ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.7)
    ax.legend(loc='upper left', frameon=False)
    ax.set_ylim(1, max(max(noisy_power), max(clean_power), threshold) * 1.05)
    ax.minorticks_on()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.show()

print(len(time))
print(POWER.shape)
print(power_per_second.shape)
time = np.arange(0, 13*300, 1)
plot_power_data(time, POWER, power_per_second, title='Power Data', save_path='plot/power_data.png')








