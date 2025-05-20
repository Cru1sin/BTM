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
POWER = POWER * 1000000 /250
print(POWER)
"""# 可视化
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(target_time, POWER, label='Power with Noise', color='#008000')
plt.plot(target_time, power_per_second, label='Power without Noise', color='#56B4E9')
plt.axhline(y=2.3, color='red', linestyle='--', label='Electrolyzer Power')
plt.xlabel('Time (s)')
plt.ylabel('Power (MW)')
plt.title('Power with Noise and without Noise in 1 hour')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('plot/noise_power.png', dpi=300)
plt.show()"""


"""# 绘图
plt.figure(figsize=(10, 5), dpi=150)
plt.plot(time, solar_power, label='Solar Power', color='#E69F00')
plt.plot(time, wind_power, label='Wind Power', color='#56B4E9')
# 使用绿色画总的功率的曲线
plt.plot(time, total_power, label='Total Power', color='#008000')
# 画电解槽功率的直线
plt.axhline(y=2.3, color='red', linestyle='--', label='Electrolyzer Power')
plt.xlabel('Time (hh:mm)', fontsize=10)
plt.ylabel('Power (MW)', fontsize=10)
plt.title('Power Output of Solar and Wind Energy in One Day', fontsize=12)
plt.xticks(np.arange(0, 294, 24), time[::24], rotation=45)  # 每2小时一个tick
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig('plot/solar_wind_power1.png', dpi=300)
plt.show()"""











