from CoolProp.CoolProp import PropsSI
import numpy as np
import math
from math_utils import sqrt
from BatteryModel import Battery
import casadi as ca

class Battery_Model(Battery):
    def __init__(self, dt):
        super().__init__(dt)

    def battery_thermal_model(self, current, Q_cool, T_bat):
        """
        根据电流和电池温度计算生成热
        :param current: 电池的电流 (A)
        :param T_bat: 电池的温度 (℃)
        :return: 生成热 (W), 电池工作功率
        """
        R_bat = self.R_int(current)
        Q_gen = current**2 * R_bat + ca.fabs(current) * (T_bat + 273.15) * self.entropy_coefficient
        T_bat_next = ((-Q_cool + Q_gen) / (self.M_bat * self.capacity_bat_thermal)) + T_bat

        P_bat = (self.U_oc - current * R_bat) * current

        self.update_SOC_OCV(current)

        return T_bat_next, P_bat


def generate_cooling_values(N, dt):
    """
    生成动态变化的冷却量 Q_cool 和冷却功率 P_cooling
    ----------------------------------------
    :param N: 时域长度
    :param dt: 时间步长 (s)
    :return: Q_cool (W), P_cooling (W)
    """
    t = np.arange(0, N * dt, dt)  # 时间数组
    Q_cool = -500 - 200 * np.sin(0.2 * t)  # 冷却量的动态变化
    P_cooling = 100 + 50 * np.cos(0.1 * t)  # 冷却功率的动态变化
    return Q_cool, P_cooling

def test_battery_model(battery, vehicle, N, dt):
    """
    测试电池热模型
    ----------------------------------------
    :param battery: 电池模型实例
    :param vehicle: 车辆模型实例
    :param N: 预测的时间步长数
    :param dt: 时间步长 (s)
    """
    # 初始化结果存储
    battery_temps = []  # 存储每个时间步长的电池温度
    Q_gen_values = []   # 存储每个时间步长的电池发热量

    compute_time = 10
    times = int(10 / dt)
    for episode in range(times):
        # 计算牵引功率
        P_traction = vehicle.traction()
        Q_cool, P_cooling = generate_cooling_values(N, dt)
        # 电池发热功率
        Q_gen = battery.battery_thermal_generation(P_cooling+P_traction)
        # 更新电池温度
        battery.battery_thermal_model(Q_cool, P_cooling+P_traction)

        print(f"Time step {episode}/{times}")
        print(f"Vehicle velocity: {vehicle.velocity:.2f} m/s")
        print(f"Battery temperature: {battery.T_bat:.2f} ℃")
        print("-----") 
        vehicle.update_velocity()
        battery_temps.append(battery.T_bat)
        Q_gen_values.append(Q_cool[0])

    return battery_temps, Q_gen_values

from vehicle_dynamics_model import Vehicle
if __name__ == '__main__':
    # 设置模型参数
    N, dt = 1, 0.1  # 时域长度和时间步长
    vehicle = Vehicle(N, dt)
    battery = Battery_Model(25, N, dt)

    vehicle.velocity = 10.5  # 初始化车辆速度

    # 动态生成冷却量和冷却功率
    

    # 运行测试
    battery_temps, Q_gen_values = test_battery_model(battery, vehicle, N, dt)

    # 输出测试结果
    print("Battery temperature evolution: ", battery_temps)
    print("Battery heat generation evolution: ", Q_gen_values)