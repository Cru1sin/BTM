"""
蓄电池容量为20kWh，测试时长为单天00：00到隔日00：00

不考虑自放电

锂电池充电I>0
锂电池放电I<0
"""
import numpy as np

# 从图中人工提取的归一化功率比例（单位化）
wind_shape = np.array([4200, 4500, 4700, 4800, 4700, 4600, 4500, 4400, 4500, 4400,
                       4300, 4600, 4700, 4100, 3600, 3000, 2400, 2300, 2000, 1800,
                       2200, 2700, 3100, 3500])
solar_shape = np.array([0, 0, 0, 0, 0, 0, 0, 0, -200, 1000, 3000, 5000,
                        6500, 7200, 7000, 6000, 5000, 3000, 1000, 0, 0, 0, 0, 0])

# 归一化
wind_shape = wind_shape / np.max(wind_shape)
solar_shape = solar_shape / np.max(solar_shape)

# 实际装机容量（单位：kW）
wind_capacity = 170
solar_capacity = 0.07

# 实际出力（单位：kW）
P_wind = wind_shape * wind_capacity
P_solar = solar_shape * solar_capacity
P_RE_HOUR = (P_wind + P_solar) * 1000
P_RE_HOUR_C = P_RE_HOUR[17:]

from scipy.interpolate import interp1d

# 原始时间点（单位：小时，0~23）
t_hours = np.arange(24)
t_hours_C = np.arange(17, 24)
# 新的时间点（单位：小时），每秒一个点，共 86400 点
t_seconds = np.linspace(0, 23, 86400)
t_seconds_C = np.linspace(17, 23, 86400)
# 插值函数（线性插值）
interp_func = interp1d(t_hours, P_RE_HOUR, kind='cubic')
interp_func_C = interp1d(t_hours_C, P_RE_HOUR_C, kind='cubic')

# 得到每秒功率数据
P_RE_SECOND = interp_func(t_seconds)  # 形状为 (86400,)
P_RE_SECOND_C = interp_func_C(t_seconds_C)  # 形状为 (86400,)

std_dev = 1 * np.max(P_RE_SECOND)  # 例如波动范围约 ±20%
noise = np.random.normal(0, std_dev, size=P_RE_SECOND.shape)
from scipy.ndimage import gaussian_filter1d
smoothed_noise = gaussian_filter1d(noise, sigma=60)  # 控制波动频率
noisy_power = P_RE_SECOND + smoothed_noise
POWER = np.clip(noisy_power, 0, None)  # 防止出现负功率

TARGET = 142300 - POWER
# 绘图
import matplotlib.pyplot as plt

"""plt.figure(figsize=(10, 5))
plt.plot(P_RE_SECOND, label='Original Power')
plt.plot(noisy_power, label='Noisy Power')
plt.plot(TARGET, label='Target Power')
plt.legend()
plt.show()"""

Q = [4.5, 7.5, 0, 0, 0, 0, 3, -11, -4, 0, 0, 0, 3, 8, 1, -1, -1, 10, 0, 0, 0, -10, -8, 0]

if __name__ == "__main__":
    print(f'length of Q: {len(Q)}')
    Q_total = []
    for i in range(len(Q)):
        if i == 0:
            Q_total.append(Q[i])
        else:
            Q_total.append(Q_total[i-1] + Q[i])
    print(f'Q_total: {Q_total}')
    print(f'max of Q_total: {max(Q_total)}')
    print(f'min of Q_total: {min(Q_total)}')
    print(f'P_RE = {P_RE_HOUR/1000} kW')
    print(f'R_RE_len = {len(P_RE_per_second)}')

