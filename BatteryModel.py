from abc import ABC, abstractmethod
import numpy as np
import casadi as ca
from math import exp

class Battery(ABC):
    def __init__(self, dt):
        self.dt = dt  # 时间步长 (s)
        self.M_bat = 40  # 电池热质量 (kg)
        self.capacity_bat_thermal = 1350  # 电池比热容 (J/kg·K)
        self.entropy_coefficient = 0.06  # 电池的熵系数 (V/K)
        self.U_oc = 360  # 标称开路电压 (V)
        self.R_bat = 6 * 1e-2  # 电池内阻 (Ω)
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

        self.max_I_charge = (self.U_charge_cutoff - self.U_oc) / self.R_bat # 最大充电电流
        self.max_I_discharge = (self.U_oc - self.U_discharge_cutoff) / self.R_bat # 最大放电电流

        self.max_c_rate_charge = (self.U_charge_cutoff - self.U_oc) / self.R_bat / self.Ah_bat_initial# 最大充电倍率
        self.max_c_rate_discharge = (self.U_oc - self.U_discharge_cutoff) / self.R_bat / self.Ah_bat_initial # 最大放电倍率

        self.Ah_discharge_max = self.Ah_cell * self.N_parallel # 最大放电容量
        self.Q_discharged = 0 # 已放电容量
        
    def R_int(self, I_bat):
        """
        根据C-rate更新电池内阻，并应用随机分布模拟电池特性
        由于根据C-rate会有很大变化，所以需要实时更新
        :param c_rate: 电池的C-rate
        """
        # 基础内阻计算
        c_rate = I_bat / self.Ah_bat_initial / self.N_parallel
        self.R_bat = self._calculate_base_internal_resistance(c_rate)
        return self.R_bat
    
    def update_SOC_OCV_Ah(self, current, T_bat):
        """
        使用安倍法更新电池的SOC和OCV
        :param current: 电池的电流 (A)
        """
        delta_Q = current * self.dt / 3600
        self.Q_discharged += delta_Q

        delta_soc = delta_Q / self.Ah_bat_actual
        self.N_cycle += ca.fabs(delta_soc)
        self.SOC -= delta_soc

        OCV_cell = (0.882 - 9.5 * 1e-5 * self.N_cycle) * self.SOC + 3.3
        self.U_oc = self.N_series * OCV_cell
        self.Ah_discharge_max = self._calculate_discharge_capacity(T_bat) * self.N_parallel

    def _calculate_base_internal_resistance(self, C_rate):
        """
        计算基础内阻
        :param c_rate: 电池的C-rate
        :return: 基础内阻 (Ω)
        """
        R_int = 13*1e-3
        R1 = 3e-11 * self.N_cycle**4 - 9e-8 * self.N_cycle**3 + 7e-5 * self.N_cycle**2 - 0.0149 * self.N_cycle + R_int
        R2 = 2e-8 * self.N_cycle**3 - 3e-5 * self.N_cycle**2 + 0.0133 * self.N_cycle + R_int
        R3 = 1e-8 * self.N_cycle**3 - 1e-5 * self.N_cycle**2 + 0.0087 * self.N_cycle + R_int
        R_int = ca.if_else(C_rate <= 2.5, R1, ca.if_else(C_rate <= 6, R2, R3))
        self.R_bat = R_int * self.N_series / self.N_parallel

        self.max_I_charge = (self.U_charge_cutoff - self.U_oc) / self.R_bat # 最大充电电流
        self.max_I_discharge = (self.U_oc - self.U_discharge_cutoff) / self.R_bat # 最大放电电流
        self.max_c_rate_charge = (self.U_charge_cutoff - self.U_oc) / self.R_bat / self.Ah_bat_initial# 最大充电倍率
        self.max_c_rate_discharge = (self.U_oc - self.U_discharge_cutoff) / self.R_bat / self.Ah_bat_initial # 最大放电倍率

        return self.R_bat
    
    def update_soh(self, current, T_bat, SOH):
        """
        更新电池的SOH（基于循环老化和温度影响）
        :param current: 电池的电流 (A)
        :param T_bat: 电池的温度 (℃)
        """
        # 计算C-rate
        c_rate = abs(current) / self.Ah_bat_initial
        # 计算循环老化
        aging_rate = self._calculate_aging_rate(c_rate, T_bat)
        SOH -= aging_rate * self.dt / 100
        # 更新实际容量
        self.Ah_bat_actual = self.Ah_bat_initial * SOH
        self.SOH = SOH
        return SOH
    
    def _calculate_discharge_capacity(self, T_bat):
        """
        计算在特定温度下的最大放电容量 (Ahdischarge,max)
        :param T_bat: 电池温度 (℃)
        :return: 最大放电容量 (Ah)
        """
        Ah_discharge_max = -0.0002 * (T_bat ** 2) + 0.014 * T_bat + 2.69
        return Ah_discharge_max

    def _calculate_aging_rate(self, c_rate, T_bat):
        """
        计算电池老化速率
        :param c_rate: 电池的C-rate
        :param T_bat: 电池温度 (℃)
        :return: 老化速率 (1/day)
        """
        # 老化因子
        Af, B = ca.if_else(c_rate <= 2, (3814.7-44.6*2, 21681), ca.if_else(c_rate <= 4, (3814.7-44.6*4, 12934), ca.if_else(c_rate <= 10, (3814.7-44.6*10, 15512), (3814.7-44.6*20, 15512))))
        z = 0.66  # 幂律因子
        aging_rate = B * exp(-Af / (T_bat + 273.15)) * (self.capacity_bat_actual ** z)
        return aging_rate


    @abstractmethod
    def battery_thermal_model(self, *args, **kwargs):
        pass
