from CoolProp.CoolProp import PropsSI
import numpy as np
from bin.math_utils import (
    exp, log, max, min, sqrt, power,
    if_else, logical_eq, logical_le
)
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import CubicSpline
import casadi as ca
PI = 3.14

from CoolingSystem.CoolingSystem import CoolingSystem


class SimpleCoolingSystem(CoolingSystem):
    """
    一个简单的液体冷却系统，假设传热方式为对流换热
    """
    def __init__(self, dt, T_amb):
        super().__init__(dt, T_amb)

    def battery_cooling(self, T_bat, n_pump):
        """
        计算冷却量 Q_cool (W)
        Q_cool = h * A * (T_bat - T_clnt_in)
        """
        self.n_pump = n_pump
        Q_cool = self.h_bat * self.A_bat * (T_bat - self.T_clnt_in)
        return Q_cool

    def power(self, n_pump):
        """
        计算泵的功率消耗 P_pump (W)
        P_pump = k_pump * n_pump^3
        """
        k_pump = 1e-6  # 假设泵的功率常数
        P_pump = k_pump * (n_pump ** 3)
        return P_pump


# **测试模型**
if __name__ == "__main__":
    dt = 0.1  # 采样时间
    T_amb = 25  # 环境温度
    cooling_system = SimpleCoolingSystem(dt, T_amb)

    T_bat = 40  # 当前电池温度 (℃)
    n_pump = 2000  # 泵转速 (rpm)

    Q_cool = cooling_system.battery_cooling(T_bat, n_pump)
    P_pump = cooling_system.power(n_pump)

    print(f"冷却量 Q_cool: {Q_cool:.2f} W")
    print(f"泵功率消耗 P_pump: {P_pump:.2f} W")