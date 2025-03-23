from abc import ABC, abstractmethod
import numpy as np

class Battery(ABC):
    def __init__(self, dt):
        self.dt = dt  # 时间步长 (s)
        self.M_bat = 40  # 电池热质量 (kg)
        self.capacity_bat_thermal = 1350  # 电池比热容 (J/kg·K)
        self.entropy_coefficient = 0.06  # 电池的熵系数 (V/K)
        self.U_oc = 360  # 标称开路电压 (V)
        self.R_bat = 6 * 1e-3  # 电池内阻 (Ω)
        self.SOC = 1  # 电池的SOC

        self.N_cycle = 0  # 电池的循环次数

        self.N_series = 100
        self.N_parallel = 22
        self.N_cell = self.N_series * self.N_parallel
        self.Ah_cell = 3
        self.Ah_bat_initial = 66
        self.Ah_bat_actual = 52

        self.U_charge_cutoff = 420 # 充电截止电压 (V)
        self.U_discharge_cutoff = 250 # 放电截止电压 (V)

    def update_internal_resistance(self, I_bat):
        """
        根据C-rate更新电池内阻，并应用随机分布模拟电池特性
        :param c_rate: 电池的C-rate
        """
        # 基础内阻计算
        c_rate = I_bat / self.Ah_bat
        base_r = self._calculate_base_internal_resistance(c_rate)
        # 应用随机分布模拟电池特性
        self.R_bat = self._apply_random_mask(base_r, std=0.1 * base_r)

    def _calculate_base_internal_resistance(self, c_rate):
        """
        计算基础内阻
        :param c_rate: 电池的C-rate
        :return: 基础内阻 (Ω)
        """
        return 6 * 1e-3 * (1 + 0.1 * c_rate)

    @abstractmethod
    def update_soc(self, current):
        """
        根据电流更新电池的SOC
        :param current: 电池的电流 (A)
        """
        pass

    @abstractmethod
    def thermal_generation(self, current, T_bat):
        """
        根据电流和电池温度计算生成热
        :param current: 电池的电流 (A)
        :param T_bat: 电池的温度 (℃)
        :return: 生成热 (W)
        """
        pass

    def _apply_random_mask(self, value, std):
        """
        使用正态分布随机化参数
        :param value: 原始值
        :param std: 标准差
        :return: 随机化后的值
        """
        return np.random.normal(value, std)
