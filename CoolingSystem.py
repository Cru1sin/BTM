from abc import ABC, abstractmethod

class CoolingSystem(ABC): 
    """
    冷却系统抽象类
    """
    def __init__(self, dt, T_amb):
        """
        Battery
        """
        self.A_bat = 12   # 电池包的冷却表面积(m^2) 假设电池包尺寸为 1.0 * 1.0 * 0.3
        self.h_bat = 300 # Heat transfer coefficient h (W/(m2 ℃))
        self.T_amb = T_amb # Ambient temperature (℃)

        '''
        Coolant
        40% 乙二醇 + 60% 水的混合液
        '''
        self.rho_clnt = 1069.5  # Density of coolant (kg/m^3)
        self.capacity_clnt = 3330 # Specific heat capacity of coolant (J/kg·K)
        self.mu_clnt = 1.1e-3      # 动力粘度 (Pa·s)
        self.V_pump = 33*1e-6 # Displacement volume of pump (m^3/rev)
        self.T_clnt_in = 20 # the coolant temperature at the inlet of the pipe in battery pack        

        '''
        Control variables
        '''
        self.massflow_clnt = 0  # Mass flow rate of coolant (Kg/s)
        self.n_pump = 0 # Pump speed (rpm)
        self.dt = dt # 采样时间 dt (s), example dt = 0.1
    
    @abstractmethod
    def battery_cooling(self, *args, **kwargs):
        """
        计算冷却系统的 Q_cool
        """
        pass

    def power(self, *args, **kargs):
        """
        计算冷却系统功率
        """
        pass
    