from CoolProp.CoolProp import PropsSI
from math_utils import custom_exp as exp
import numpy as np

PI = 3.14

class simple_cooling_model:
    def __init__(self, T_amb, dt):
        self.T_amb = T_amb
        self.dt = dt

        self.P_cond = 2100000
        self.P_eva = 420000

        self.h_rfg = 130000
        self.h_rfg_upper = 430000
        self.h_rfg_lower = 300000

        self.T_clnt_upper = T_amb
        self.T_clnt_lower = T_amb

        self.V_comp = 33 # Displacement volume of compressor (cm^3/rev)
        self.V_pump = 33 # Displacement volume of pump (cm^3/rev)

        self.m_rfg = None
        self.m_clnt = None

        self.rho_clnt = 1069.5  # Density of coolant (kg/m^3)
        self.capacity_clnt = 3330 # Specific heat capacity of coolant (J/kg·K)
        self.capacity_air = 1003 

        self.v_rfg_eva_out = 1 / PropsSI("D", "P", self.P_eva, "H", self.h_rfg_upper, 'R134a')

        self.coefficients_cond = None
        self.coefficients_eva = None
    
    def massflow_rfg(self, n_comp):
        self.v_rfg_eva_out = 1 / PropsSI("D", "P", self.P_eva, "H", self.h_rfg_upper, 'R134a')
        efficiency_vol = 0.966 * (1 - 0.05*((self.P_cond / self.P_eva)**(1/1.15) - 1))
        self.m_rfg = 1.66 * 1e-8 * self.V_comp * efficiency_vol * n_comp / self.v_rfg_eva_out

    def compressor_power(self):
        P_comp = self.P_eva * 1e3 * self.v_rfg_eva_out * 1.15 / 0.15 * ((self.P_cond / self.P_eva)**(0.15/1.15) - 1) * self.m_rfg
        return P_comp

    def massflow_clnt(self, n_pump):
        efficiency_vol = 0.95 # 泵的容积效率
        self.m_clnt = self.V_pump * 1e-6 * efficiency_vol * n_pump * self.rho_clnt / 60

    def pump_power(self):
        efficiency_power = 0.75 # 泵的能量转化效率
        Delta_P_pump = (0.927 * self.m_clnt ** 2 + 0.586 * self.m_clnt - 0.143) * 1000
        P_pump = Delta_P_pump * self.m_clnt / efficiency_power / self.rho_clnt
        return P_pump
    
    '''
    def thermal_para_initial(self):
        H_lower = 200000
        H_liquid = PropsSI("H", "P", self.P_cond, "Q", 0, "R134a")  # 液态焓值
        H_vapor = PropsSI("H", "P", self.P_cond, "Q", 1, "R134a")   # 气态焓值
        H_upper = 500000

        H_values_liquid = np.linspace(H_lower, H_liquid, 50)  # 液态区间
        H_values_liq_vap = np.linspace(H_liquid, H_vapor, 50)
        H_values_vapor = np.linspace(H_vapor, H_upper, 50)     # 气态区间

        # 获取对应温度值
        T_values_liquid = [PropsSI("T", "P", self.P_cond, "H", H, "R134a") for H in H_values_liquid]
        T_values_liq_vap = [PropsSI("T", "P", self.P_cond, "H", H, "R134a") for H in H_values_liq_vap]
        T_values_vapor = [PropsSI("T", "P", self.P_cond, "H", H, "R134a") for H in H_values_vapor]

        coefficients_cond_liquid = np.polyfit(H_values_liquid, T_values_liquid, deg=2)
        coefficients_cond_liq_vap = np.polyfit(H_values_liq_vap, T_values_liq_vap, deg=2)
        coefficients_cond_vapor = np.polyfit(H_values_vapor, T_values_vapor, deg=2)

        self.coefficients_cond = [
            {"range": (H_lower, H_liquid), "coefficients": coefficients_cond_liquid},
            {"range": (H_liquid, H_vapor), "coefficients": coefficients_cond_liq_vap},
            {"range": (H_vapor, H_upper), "coefficients": coefficients_cond_vapor},
        ]

        H_lower = 200000
        H_liquid = PropsSI("H", "P", self.P_eva, "Q", 0, "R134a")  # 液态焓值
        H_vapor = PropsSI("H", "P", self.P_eva, "Q", 1, "R134a")   # 气态焓值
        H_upper = 500000

        H_values_liquid = np.linspace(H_lower, H_liquid, 50)  # 液态区间
        H_values_liq_vap = np.linspace(H_liquid, H_vapor, 50)
        H_values_vapor = np.linspace(H_vapor, H_upper, 50)     # 气态区间

        # 获取对应温度值
        T_values_liquid = [PropsSI("T", "P", self.P_eva, "H", H, "R134a") for H in H_values_liquid]
        T_values_liq_vap = [PropsSI("T", "P", self.P_eva, "H", H, "R134a") for H in H_values_liq_vap]
        T_values_vapor = [PropsSI("T", "P", self.P_eva, "H", H, "R134a") for H in H_values_vapor]

        coefficients_eva_liquid = np.polyfit(H_values_liquid, T_values_liquid, deg=2)
        coefficients_eva_liq_vap = np.polyfit(H_values_liq_vap, T_values_liq_vap, deg=2)
        coefficients_eva_vapor = np.polyfit(H_values_vapor, T_values_vapor, deg=2)

        self.coefficients_eva = [
            {"range": (H_lower, H_liquid), "coefficients": coefficients_eva_liquid},
            {"range": (H_liquid, H_vapor), "coefficients": coefficients_eva_liq_vap},
            {"range": (H_vapor, H_upper), "coefficients": coefficients_eva_vapor},
        ]
    
    def temp_cond(self, H):
        for segment in self.coefficients_cond:
            H_min, H_max = segment["range"]
            if H_min <= H < H_max:
                poly_fit = np.poly1d(segment["coefficients"])
                return poly_fit(H) - 273.15
        raise ValueError("H value is out of defined range.")
    
    def temp_eva(self, H):
        for segment in self.coefficients_eva:
            H_min, H_max = segment["range"]
            if H_min <= H < H_max:
                poly_fit = np.poly1d(segment["coefficients"])
                return poly_fit(H) - 273.15
        raise ValueError("H value is out of defined range.")

    def cooling(self):

        K_cond, A_cond = 1000, 0.3
        K_eva, A_eva = 1000, 0.3
        K_bat, A_bat = 300, 12

        T_cond = self.temp_cond((self.h_rfg_lower+self.h_rfg_upper)/2)
        # 蒸发器容量影响Delta_h_max
        m_air = self.m_rfg * 100
        # T_air_out = K_cond * A_cond / (10 * m_air * self.capacity_air) * (T_cond - self.T_amb) + self.T_amb
        T_air_out = (T_cond - self.T_amb)*np.exp(-(K_cond*A_cond)/(m_air*self.capacity_air)) + self.T_amb
        # self.h_rfg_lower = K_cond * A_cond / self.m_rfg * ((T_air_out+self.T_amb)/2 - T_cond) + self.h_rfg_upper
        self.h_rfg_lower = -m_air*self.capacity_air*(T_air_out - self.T_amb) + self.h_rfg_upper

        T_eva = self.temp_eva(self.h_rfg_lower)
        C_min = min(self.m_rfg*PropsSI("C","P",self.P_eva,"H",(self.h_rfg_lower+self.h_rfg_upper)/2,"R134a"),self.m_clnt*self.capacity_clnt)
        C_max = max(self.m_rfg*PropsSI("C","P",self.P_eva,"H",(self.h_rfg_lower+self.h_rfg_upper)/2,"R134a"),self.m_clnt*self.capacity_clnt)
        C_r = C_min / C_max
        NTU = K_eva*A_eva/C_min
        epsilon = 1 - np.exp(-NTU)  # NTU 方法效率公式（逆流）
        Q = epsilon*C_min*(self.T_clnt_upper - T_eva)
        
        self.h_rfg_upper = Q/self.m_rfg + self.h_rfg_lower

        self.T_clnt_lower = -Q/self.m_clnt/self.capacity_clnt + self.T_clnt_upper
        self.T_clnt_upper = (self.T_clnt_lower - self.T_bat)*np.exp(-(K_bat*A_bat)/(self.m_clnt*self.capacity_clnt)) + self.T_bat
        # self.T_clnt_upper = K_bat * A_bat / (self.m_clnt * self.capacity_clnt) * (self.T_bat - (self.T_clnt_upper + self.T_clnt_lower)/2) + self.T_clnt_lower
        # Q_cooling = (self.m_clnt * self.capacity_clnt) * (self.T_bat - (self.T_clnt_upper + self.T_clnt_lower)/2)
        Q_cooling = (self.m_clnt * self.capacity_clnt) * (self.T_clnt_lower - self.T_clnt_upper)

        return Q_cooling
    '''
    def Q_cond(self, K_cond=1000, A_cond=0.3):
        T_cond = 69.64
        m_air = self.m_rfg * 10
        T_air_out = (self.T_amb - T_cond)* exp(-(K_cond*A_cond)/(m_air*self.capacity_air)) + T_cond
        Q_cond = m_air*self.capacity_air * (T_air_out - self.T_amb)
        return Q_cond
    
    def Q_eva(self, K_eva = 1000, A_eva = 0.3):
        T_eva = 10.39
        self.T_clnt_lower = (self.T_clnt_upper - T_eva) * exp(-(K_eva*A_eva)/(self.m_clnt*self.capacity_clnt)) + T_eva
        Q_eva = self.m_clnt*self.capacity_clnt * (self.T_clnt_lower - self.T_clnt_upper)
        return Q_eva
    
    def Q_cooling(self, T_bat, T_eva = 10.39, K_eva = 1000, A_eva = 0.3, K_bat = 300, A_bat = 12):
        self.T_clnt_lower = (self.T_clnt_upper - T_eva) * exp(-(K_eva*A_eva)/(self.m_clnt*self.capacity_clnt)) + T_eva
        self.T_clnt_upper = (self.T_clnt_lower - T_bat)*exp(-(K_bat*A_bat)/(self.m_clnt*self.capacity_clnt)) + T_bat
        Q_cooling = (self.m_clnt*self.capacity_clnt) * (self.T_clnt_upper - self.T_clnt_lower)
        return Q_cooling

    def dynamic_h_rfg(self):
        Q_cond = self.Q_cond()
        Q_eva = self.Q_eva()
        Delta_h_rfg = (Q_cond + Q_eva) / self.m_rfg
        return Delta_h_rfg
    
    def dynamic_T_clnt(self):
        Q_eva = self.Q_eva()
        Q_cooling = self.Q_cooling()
        Delta_h_clnt = (Q_eva + Q_cooling) / self.m_clnt
        return Delta_h_clnt

    def cooling(self, T_bat):
        K_cond, A_cond = 1000, 0.3
        K_eva, A_eva = 1000, 0.3
        K_bat, A_bat = 300, 12

        T_cond = 69.64
        T_eva = 10.39
        m_air = self.m_rfg * 10
        T_air_out = (self.T_amb - T_cond)* exp(-(K_cond*A_cond)/(m_air*self.capacity_air)) + T_cond
        Q_cond = m_air*self.capacity_air * (T_air_out - self.T_amb)
        self.T_clnt_lower = (self.T_clnt_upper - T_eva) * exp(-(K_eva*A_eva)/(self.m_clnt*self.capacity_clnt)) + T_eva
        Q_eva = self.m_clnt*self.capacity_clnt * (self.T_clnt_lower - self.T_clnt_upper)
        self.T_clnt_upper = (self.T_clnt_lower - T_bat)*exp(-(K_bat*A_bat)/(self.m_clnt*self.capacity_clnt)) + T_bat
        self.h_rfg = (Q_cond + Q_eva) / self.m_rfg
        Q_cooling = (self.m_clnt*self.capacity_clnt) * (self.T_clnt_upper - self.T_clnt_lower)
        self.h_clnt = (Q_eva + Q_cooling) / self.m_clnt
        return Q_cooling

import matplotlib.pyplot as plt
def test_cooling_system(cooling_system, N=30, dt=0.1):
    """
    测试冷却系统模型
    参数：
        cooling_system: 冷却系统模型对象
        N: 预测时域内的采样点数
        dt: 每个采样点的时间步长 (s)
    """
    # 初始化
    n_comp_values = np.random.uniform(1000, 5000, N)  # 随机生成压缩机转速 (rpm)
    n_pump_values = np.random.uniform(1000, 5000, N)  # 随机生成泵转速 (rpm)
    Q_cooling_values = []  # 存储每个时间步的电池冷却量
    h_rfg_values = []  # 制冷剂焓值变化
    P_rfg_values = []  # 制冷剂压力变化
    T_clnt_lower_values = []  # 冷却液出口温度
    T_clnt_upper_values = []  # 冷却液入口温度

    #cooling_system.massflow_rfg(n_comp_values)
    #cooling_system.massflow_clnt(n_pump_values)
    #Q_cooling = cooling_system.cooling()
    #cooling_system.thermal_para_initial()

    # 时域模拟
    for i in range(N):
        n_comp = n_comp_values[i]
        n_pump = n_pump_values[i]

        # 更新制冷剂和冷却液的质量流量
        cooling_system.massflow_rfg(n_comp)
        cooling_system.massflow_clnt(n_pump)

        # 计算冷却量
        Q_cooling = cooling_system.cooling(T_bat)
        Q_cooling_values.append(Q_cooling)

        # 存储状态变量
        h_rfg_values.append(cooling_system.h_rfg_upper)
        P_rfg_values.append(cooling_system.P_eva)
        T_clnt_lower_values.append(cooling_system.T_clnt_lower)
        T_clnt_upper_values.append(cooling_system.T_clnt_upper)

    # 绘制制冷循环焓值-压力变化图
    plt.figure(figsize=(10, 5))
    plt.plot(h_rfg_values, P_rfg_values, marker='o')
    plt.title("制冷循环中的焓值-压力变化")
    plt.xlabel("焓值 (J/kg)")
    plt.ylabel("压力 (Pa)")
    plt.grid()
    plt.show()

    # 绘制冷却循环温度变化图
    time = np.arange(0, N * dt, dt)
    plt.figure(figsize=(10, 5))
    plt.plot(time, T_clnt_lower_values, label="冷却液出口温度 (T_clnt_lower)")
    plt.plot(time, T_clnt_upper_values, label="冷却液入口温度 (T_clnt_upper)")
    plt.title("冷却循环温度变化")
    plt.xlabel("时间 (s)")
    plt.ylabel("温度 (K)")
    plt.legend()
    plt.grid()
    plt.show()

    # 输出电池冷却量总和
    total_cooling = np.sum(Q_cooling_values) * dt
    print(f"未来时域内电池总冷却量: {total_cooling:.2f} J")

    return Q_cooling_values, h_rfg_values, P_rfg_values, T_clnt_lower_values, T_clnt_upper_values

if __name__ == "__main__":
    cooling_system = simple_cooling_model(T_amb=25, dt=0.1, N=30)
    T_bat = 35
    Q_cooling_values, h_rfg_values, P_rfg_values, T_clnt_lower_values, T_clnt_upper_values = test_cooling_system(cooling_system)