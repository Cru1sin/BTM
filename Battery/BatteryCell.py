from abc import ABC, abstractmethod
import numpy as np
import casadi as ca
from utils.math_utils import exp
#from SOH.inference import SOH_model

class BatteryCell(ABC):
    """
    Model of Single Battery Cell
    all variables input will be processed from Pack layer to Cell layer
    parameters will be updated after Model calculation 
    :func _update_parameter() 
    """
    def __init__(self, dt, T_bat=25, N_cycle=300, SOH=1):
        
        # ==========================================电池Cell基本属性的初始化
        self.dt = dt  # 时间步长 (s)
        # =====================电池热系数
        self.capacity_bat_thermal = 1350  # 电池比热容 (J/kg·K)
        self.entropy_coefficient = 0.06  # 电池的熵系数 (V/K)
        # =====================电池电系数
        self.OCV = 3.3
        self.R_cell = 20 * 1e-3  # 电池内阻 (Ω)
        self.SOC = 0.5  # 电池的SOC
        self.T_bat = T_bat # 初始化电池温度
        self.N_cycle = N_cycle  # 电池的循环次数
        self.Ah_cell = 3
        self.U_charge_cutoff = 4.2 # 充电截止电压 (V)
        self.U_discharge_cutoff = 2.5 # 放电截止电压 (V)
        self.SOH = SOH
        #self.SOH_model = SOH_model() # SOH模型，用于更新SOH
        self.Ah_cell_actual = self.Ah_cell * self.SOH
        self.worktime = 0 # 一个循环内的时间
        # =========================================电池Cell的约束
        self.max_I_charge = (self.U_charge_cutoff - self.OCV) / self.R_cell # 最大充电电流
        self.max_I_discharge = (self.U_discharge_cutoff - self.OCV) / self.R_cell # 最大放电电流
        self.Ah_discharge_max = self.Ah_cell  # 初始化最大放电容量
        self.Ah_discharged = 0 # 已放电容量
        self.SOC_min = 0 # 最小SOC

    def update_SOC(self, I_cell):
        """
        I > 0, if charge
        I < 0, if discharge
        """
        delta_Q = I_cell * self.dt / 3600
        delta_soc = delta_Q / self.Ah_cell_actual
        self.N_cycle += ca.fabs(delta_soc)
        self.SOC += delta_soc

    def update_OCV(self):
        """
        After SOC
        """
        self.OCV = (0.882 - 9.5 * 1e-5 * self.N_cycle) * self.SOC + 3.3


    def update_SOH(self, I_cell):
        """
        现在只计算一个循环，因此没有重置Cycle_time
        如果Cycle_time重置，SOH_model的中间变量也需要重置
        Input: N_cycle, Cycle_time, Relative_time(dt),I, V_terminal, T, SOC  
        """
        V_terminal = I_cell * self.R_cell
        #self.SOH += self.SOH_model(self.N_cycle, self.worktime, self.dt, I_cell, V_terminal, self.T_bat, self.SOC)


    def update_R_cell(self):
        """
        After N_cycle
        """
        # 假设C-rate小于2.5
        self.R_cell = 3e-11 * self.N_cycle**4 - 9e-8 * self.N_cycle**3 + 7e-5 * self.N_cycle**2 - 0.0149 * self.N_cycle + 13.6
    
    def update_I_constrains(self):
        """
        After OCV and R_cell
        """
        # max_I_discharge <= I_cell <= max_I_charge
        self.max_I_charge = (self.U_charge_cutoff - self.OCV) / self.R_cell # 最大充电电流,(+)
        self.max_I_discharge = (self.U_discharge_cutoff - self.OCV) / self.R_cell # 最大放电电流, (-)

    def update_discharge_capacity_constrains(self, T_bat):
        """
        约束
        SOC不能低于1 - Ah_discharge_max / self.Ah_cell
        After T_bat
        """
        self.Ah_discharge_max = -0.0002 * (T_bat ** 2) + 0.014 * T_bat + 2.69
        self.SOC_min = ca.fmax(1 - self.Ah_discharge_max / self.Ah_cell_actual, 0)
    
    def update_parameters(self, I_cell, T_bat):
        """
        In the end of every iter
        """
        self.T_bat = T_bat
        self.update_SOC(I_cell)
        #self.update_R_cell()
        self.update_OCV()
        #self.update_SOH(I_cell)
        self.update_I_constrains()
        self.update_discharge_capacity_constrains(T_bat)
        self.worktime += self.dt

    @abstractmethod
    def battery_thermal_model(self, *args, **kwargs):
        pass
