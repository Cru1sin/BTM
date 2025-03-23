from CoolProp.CoolProp import PropsSI
import numpy as np
import math
from math_utils import sqrt
from battery_model import Battery_Model

class Battery_Model_actual(Battery_Model):
    def __init__(self, dt):
        super().__init__(dt)
        self.Ah_bat_actual = self._apply_random_mask(self.Ah_bat_actual, std=0.01)
        self.SOH = self.Ah_bat_actual / self.Ah_bat_initial

    def R_int_actual(self, I_bat):
        return self._apply_random_mask(self.R_int(I_bat), std=0.1 * self.R_int(I_bat)) 

    def update_parameters(self):
        OCV_cell = (0.882 - 9.5 * 1e-5 * self.N_cycle) * self.SOC + 3.3
        self.U_oc = self.N_series * OCV_cell


    def battery_reset(self):
        self.M_bat = 40 # Battery thermal mass (kg) = 0.145 * 100 * 20 = 290
        self.capacity_bat_thermal = 1350  # Battery specific heat capacity (J/kg·K)
        self.entropy_coefficient = 0.06  # 电池的熵系数 (V/K)

        self.U_oc = 320 # 假定开路电压 (V) = 3.2 * 100
        self.R_bat = 6 * 1e-3  # 假定电池包内阻 (Ω)

        self.SOC = 1 # 电池初始的SOC
        self.capacity_bat_elec = 100 # 电池的电量 (Ah) 5 * 20 = 100 

    def battery_thermal_generation(self, P_bat, T_bat):
        """
        计算电池在dt时间内的生成热(W)
        ----------------------------------------
        假设: 
        1. 电池的输出功率为P_cooling + P_traion
        2. 动力电池内部温度一致
        3. 暂时不考虑摩擦制动动能回收
        4. 假设电池的输出电流能涵盖所有的牵引功率以及冷却功率，之后需要迭代考虑输出电流不够的情况，直接从P_max入手
        ----------------------------------------
        :param P_cooling: 时域N内的电池的冷却功率 (W)
        :param P_traction: 时域N内的电池的牵引功率 (W)
        :return: Q_gen: 时域N内电池在每个dt时间内的发热功率 (W)    
        """
        I_bat = (self.U_oc - sqrt(self.U_oc ** 2 - 4 * self.R_bat * P_bat)) / (2 * self.R_bat)

        #print('电池电流I = ', I_bat)
        #print('电池内阻发热量 = ', I_bat**2 * self.R_bat)
        #print('电池的可逆热 = ', I_bat * (T_bat + 273.15) * self.entropy_coefficient)

        Q_gen = I_bat**2 * self.R_bat + I_bat * (T_bat + 273.15) * self.entropy_coefficient
        # 电池在dt时间内的发热功率 (W)
        return Q_gen
    
    def battery_thermal_model(self, Q_cool, Power, T_bat):
        """
        电池热模型
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
        Q_gen = self.battery_thermal_generation(Power, T_bat) * self.dt
        # print('电池发热量: ', Q_gen)
        T_bat_next = ((-Q_cool + Q_gen) / (self.M_bat * self.capacity_bat_thermal)) + T_bat
        return T_bat_next
    
    def _apply_random_mask(self, value, std):
        """
        使用正态分布随机化参数
        :param value: 原始值
        :param std: 标准差
        :return: 随机化后的值
        """
        return np.random.normal(value, std)


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