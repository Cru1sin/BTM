from CoolProp.CoolProp import PropsSI
import numpy as np
import math
from utils.math_utils import sqrt
from Battery.BatteryCell import BatteryCell

class BatteryPack(BatteryCell):
    def __init__(self, dt):
        super().__init__(dt)
        self.N_series = 4
        self.N_parallel = 2
        self.N_cell = self.N_series * self.N_parallel

        self.M_bat = 40  # 电池热质量 (kg)
        self.capacity = self.Ah_cell * self.N_parallel
        self.I_max_limit = self.max_I_charge * self.N_parallel
        self.I_min_limit = self.max_I_discharge * self.N_parallel
        self.R_pack = self.R_cell * self.N_series / self.N_parallel

    def Current_Module2Cell(self, P_bat):
        """
        电池Cell的电流计算
        ----------------------------------------
        :param P_bat: 采样间隔时间内的电池Module需求响应功率 (W)
        :return I_cell: 模拟得到的电池Cell的电流
        """
        P_cell = P_bat / self.N_cell
        I_cell = (self.OCV - sqrt(self.OCV ** 2 - 4 * self.R_cell * P_cell)) / (2 * self.R_cell)
        return I_cell

    def Pack_thermal_generation(self, I_cell, T_bat):
        """
        计算电池在dt时间内的生成热(W)
        ----------------------------------------
        假设: 
        1. 电池的输出响应功率与需求功率一致
        2. 电池Cell的发热表现一致
        ----------------------------------------
        :param I_cell: 模拟得到的电池Cell的电流
        :param T_bat: 采样间隔时间内的电池的温度 (℃)
        :return: Q_gen: 时域N内电池Module在每个dt时间内的发热功率 (W)    
        """
        Q_gen_cell = I_cell**2 * self.R_cell + I_cell * (T_bat + 273.15) * self.entropy_coefficient
        Q_gen_module = Q_gen_cell * self.N_cell
        return Q_gen_module
    
    def battery_thermal_model(self, Q_cool, Power, T_bat):
        """
        电池热模型，用于控制变量是功率的情况
        ----------------------------------------
        假设:
        1. 电池dt时间内的发热功率恒为Q_gen > 0
        2. 电池dt时间内的冷却功率恒为Q_cool < 0
        Q_gen = self.battery_thermal_generation(P_cooling, P_traction)
        4. 电池的热容量不变
        ----------------------------------------
        :param Q_cool: 电池的冷却量 (W)
        :param P_cooling: 制冷冷却循环的冷却功率 (W)
        :param P_traction: 电池的牵引功率 (W)
        :param dt: 时间步长 (s)
        :return: T_bat_next: 电池在dt时间后的温度 (℃)
        """
        I_cell = self.Current_Module2Cell(Power)
        Q_gen = self.Module_thermal_generation(I_cell, T_bat) * self.dt
        T_bat_next = ((-Q_cool + Q_gen) / (self.M_bat * self.capacity_bat_thermal)) + T_bat
        self.update_module_parameters(I_cell, T_bat_next)
        return T_bat_next
    
    def Power_response(self, I_cell):
        """
        根据电池pack的电流计算电池pack的输出功率
        ----------------------------------------
        :param I_cell: 电池pack的电流 (A)
        :return: P_response: 电池pack的输出功率 (W)
        """
        P_response_cell = I_cell * (self.OCV - I_cell * self.R_cell)
        P_response = P_response_cell * self.N_cell
        return P_response
    
    def battery_model(self, Q_cool, I_pack, T_bat):
        """
        电池热电模型，用于控制变量是I的情况
        ----------------------------------------
        假设:
        1. 电池dt时间内的发热功率恒为Q_gen > 0
        2. 电池dt时间内的冷却功率恒为Q_cool <= 0
        Q_gen = self.battery_thermal_generation(P_cooling, P_traction)
        4. 电池的热容量不变
        ----------------------------------------
        :param Q_cool: 电池的冷却量 (W)
        :param I_pack: 电池pack的这一时刻的电流 (A)
        :param T_bat: 电池pack的上一时刻的温度 (℃)
        :return: T_bat_next: 电池在dt时间后的温度 (℃)
        :return: P_response: 电池pack的输出功率 (W)
        """
        I_cell = I_pack / self.N_parallel
        Q_gen = self.Pack_thermal_generation(I_cell, T_bat) * self.dt
        T_bat_next = ((-Q_cool + Q_gen) / (self.M_bat * self.capacity_bat_thermal)) + T_bat
        self.update_module_parameters(I_cell, T_bat_next)
        P_response = self.Power_response(I_cell)
        return T_bat_next, P_response, self.SOC, Q_gen

    def update_module_parameters(self, I_cell, T_bat):
        self.update_parameters(I_cell, T_bat)
        self.I_max_limit = self.max_I_charge * self.N_parallel
        self.I_min_limit = self.max_I_discharge * self.N_parallel

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