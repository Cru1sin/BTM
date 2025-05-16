from utils.math_utils import sqrt, min, max
from Battery.BatteryCell import BatteryCell

class BatteryPack(BatteryCell):
    def __init__(self, dt, T_amb=25):
        super().__init__(dt)
        self.N_series = 60
        self.N_parallel = 10
        self.N_cell = self.N_series * self.N_parallel
        self.T_amb = T_amb
        self.h_bat = 10

        self.M_bat = self.m_cell * self.N_cell # 电池热质量 (kg)
        self.capacity = self.Ah_cell * self.N_parallel
        self.I_max_limit = self.max_I_discharge * self.N_parallel
        self.I_min_limit = self.max_I_charge * self.N_parallel
        self.R_pack = self.R_cell * self.N_series / self.N_parallel

    def Current_Pack2Cell(self, P_bat):
        """
        电池Cell的电流计算
        ----------------------------------------
        :param P_bat: 采样间隔时间内的电池Module需求响应功率 (W)
        :return I_cell: 模拟得到的电池Cell的电流
        """
        P_cell = P_bat / self.N_cell
        # P_cell <= OCV**2 / (4*R_cell)
        #P_cell = min(P_cell, self.OCV**2 / (4*self.R_cell)-1e-6)
        discriminant = self.OCV ** 2 - 4 * self.R_cell * P_cell
        safe_discriminant = max(discriminant, 1e-8)
        # 电流限制
        I_cell = (self.OCV - sqrt(safe_discriminant)) / (2 * self.R_cell)
        return I_cell * self.N_parallel

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
        Q_gen_cell = I_cell**2 * self.R_cell - I_cell * (T_bat + 273.15) * self.entropy_coefficient
        Q_gen_module = Q_gen_cell * self.N_cell
        return Q_gen_module
    
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
    
    def battery_ambient_heat_transfer(self, T_bat):
        """
        电池与环境的热交换
        ----------------------------------------
        :param T_bat: 电池pack的温度 (℃)
        :return: Q_ambient: 电池pack与环境的热交换功率 (W)
        """
        Q_ambient = self.h_bat * (self.T_amb - T_bat)
        return Q_ambient
    
    def battery_model(self, Q_cool, I_pack, T_bat, SOC):
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
        Q_ambient = self.battery_ambient_heat_transfer(T_bat) * self.dt
        T_bat_next = ((-Q_cool + Q_gen + Q_ambient) / (self.M_bat * self.capacity_bat_thermal)) + T_bat
        SOC = self.update_module_parameters(I_cell, T_bat_next, SOC)
        P_response = self.Power_response(I_cell)
        return T_bat_next, P_response, SOC, Q_gen, Q_ambient

    def update_module_parameters(self, I_cell, T_bat, SOC):
        SOC = self.update_parameters(I_cell, T_bat, SOC)
        self.I_max_limit = self.max_I_discharge * self.N_parallel
        self.I_min_limit = self.max_I_charge * self.N_parallel
        return SOC
    
    def get_SOH_loss(self, I_pack, T_bat):
        I_cell = I_pack / self.N_parallel
        U_cell = self.OCV - I_cell * self.R_cell
        return self.SOH_predictor.inference(abs(I_cell), U_cell, T_bat)