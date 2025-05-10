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
P_RE_HOUR = P_wind + P_solar

from scipy.interpolate import interp1d

# 原始时间点（单位：小时，0~23）
t_hours = np.arange(24)
# 新的时间点（单位：小时），每秒一个点，共 86400 点
t_seconds = np.linspace(0, 23, 86400)

# 插值函数（线性插值）
interp_func = interp1d(t_hours, P_RE_HOUR, kind='cubic')

# 得到每秒功率数据
P_RE_SECOND = interp_func(t_seconds)  # 形状为 (86400,)

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
    print(f'P_RE = {P_RE}')
    print(f'R_RE_len = {len(P_RE_per_second)}')

