from abc import ABC, abstractmethod
import numpy as np
import casadi as ca
from utils.math_utils import exp
from SOH.inference import SOH_predictor

class BatteryCell(ABC):
    """
    Model of Single Battery Cell
    all variables input will be processed from Pack layer to Cell layer
    parameters will be updated after Model calculation 
    :func _update_parameter() 
    """
    def __init__(self, dt, T_bat=35, N_cycle=1500, SOH=1):
        
        # ==========================================电池Cell基本属性的初始化
        self.dt = dt  # 时间步长 (s)
        self.m_cell = 0.06  # 电池热质量 (kg)
        # =====================电池热系数
        self.capacity_bat_thermal = 1350  # 电池比热容 (J/kg·K)
        self.entropy_coefficient = 0.0003  # 电池的熵系数 (V/K)
        # =====================电池电系数
        self.OCV = 3.3
        self.R_cell = 200 * 1e-3  # 电池内阻 (Ω)
        self.T_bat = T_bat # 初始化电池温度
        self.N_cycle = N_cycle  # 电池的循环次数
        self.Ah_cell = 3
        self.U_charge_cutoff = 4.2 # 充电截止电压 (V)
        self.U_discharge_cutoff = 2.5 # 放电截止电压 (V)
        self.SOH = 0.8
        self.SOH_predictor = SOH_predictor(dt,0,100) # SOH模型，用于更新SOH
        self.Ah_cell_actual = self.Ah_cell * self.SOH
        self.worktime = 0 # 一个循环内的时间
        # =========================================电池Cell的约束
        self.max_I_charge = (self.OCV - self.U_charge_cutoff) / self.R_cell # 最大充电电流
        self.max_I_discharge = (self.OCV - self.U_discharge_cutoff) / self.R_cell # 最大放电电流
        self.Ah_discharge_max = self.Ah_cell  # 初始化最大放电容量
        self.Ah_discharged = 0 # 已放电容量
        self.SOC_min = 0 # 最小SOC

    def update_SOC(self, I_cell, SOC):
        """
        I > 0, if discharge
        I < 0, if charge
        """
        delta_Q = I_cell * self.dt / 3600
        delta_soc = delta_Q / self.Ah_cell_actual
        self.N_cycle += ca.fabs(delta_soc)
        SOC -= delta_soc
        return SOC

    def update_OCV(self,SOC):
        """
        After SOC
        """
        self.OCV = (0.882 - 9.5 * 1e-5 * self.N_cycle) * SOC + 3.3
   
    def update_I_constrains(self):
        """
        After OCV and R_cell
        """
        # max_I_charge <= I_cell <= max_I_discharge
        self.max_I_charge = (self.OCV - self.U_charge_cutoff) / self.R_cell # 最大充电电流
        self.max_I_discharge = (self.OCV - self.U_discharge_cutoff) / self.R_cell # 最大放电电流

    def update_discharge_capacity_constrains(self, T_bat):
        """
        约束
        SOC不能低于1 - Ah_discharge_max / self.Ah_cell
        After T_bat
        """
        self.Ah_discharge_max = -0.0002 * (T_bat ** 2) + 0.014 * T_bat + 2.69
        self.SOC_min = 0
    
    def update_parameters(self, I_cell, T_bat, SOC):
        """
        In the end of every iter
        """
        self.T_bat = T_bat
        SOC = self.update_SOC(I_cell, SOC)
        #self.update_R_cell()
        self.update_OCV(SOC)
        #self.update_SOH(I_cell)
        self.update_I_constrains()
        self.update_discharge_capacity_constrains(T_bat)
        self.worktime += self.dt
        return SOC
        

    @abstractmethod
    def battery_model(self, *args, **kwargs):
        pass
