from CoolProp.CoolProp import PropsSI
import numpy as np
from math_utils import (
    exp, log, max, min, sqrt, power,
    if_else, logical_eq, logical_le
)
from sklearn.metrics import mean_squared_error, r2_score
from scipy.interpolate import CubicSpline
import casadi as ca
PI = 3.14



class CoolingSystem: 
    """
    冷却循环: 换热器 -> 电池 -> 泵
    ----------------------------
    控制变量:
    1. 泵转速
    ----------------------------
    假设:
    1. 系统能够快速达到稳定状态，没有温度压力的剧烈波动; 系统初始状态为稳定状态
    2. 不考虑电池包内部温度分布
    3. Q_cooling用冷却水进口温度T_clnt_in和T_clnt_out, 以及m_clnt计算
    -----------------------------
    """
    def __init__(self, dt, T_amb):
        """
        Battery
        """
        self.A_bat = 12   # 电池包的冷却表面积(m^2) 假设电池包尺寸为 1.0 * 1.0 * 0.3
        self.h_bat = 300 # Heat transfer coefficient h (W/(m2 ℃))

        '''
        Coolant
        40% 乙二醇 + 60% 水的混合液
        '''
        self.rho_clnt = 1069.5  # Density of coolant (kg/m^3)
        self.capacity_clnt = 3330 # Specific heat capacity of coolant (J/kg·K)

        self.mu_clnt = 1.1e-3      # 动力粘度 (Pa·s)

        self.V_pump = 33*1e-6 # Displacement volume of pump (m^3/rev)
        
        self.T_clnt_in = T_amb # the coolant temperature at the inlet of the pipe in battery pack        

        '''
        Control variables
        '''
        self.massflow_clnt = 0  # Mass flow rate of coolant (Kg/s)

        self.dt = dt # 采样时间 dt (s), example dt = 0.1
        
        # 管路系统参数
        self.pipe_length = 3.5     # 总管道长度 (m)
        self.pipe_diameter = 0.016 # 管道内径 (m)
        self.K_factors = 2.4       # 局部阻力系数总和
        
        # 效率模型参数
        self.eta_vol_coeff = [0.91, -0.18, 0.07]  # 容积效率多项式系数
        self.eta_hyd_coeff = [0.68, 0.25, -0.12]  # 水力效率多项式系数

    def battery_cooling(self, n_pump, T_bat):
        massflow_clnt, _ = self.get_massflow(n_pump)
        T_clnt_out = if_else(
            logical_eq(massflow_clnt, 0),
            T_bat,
            (self.T_clnt_in - T_bat) * exp(-(self.h_bat * self.A_bat) / (massflow_clnt * self.capacity_clnt)) + T_bat
        )
        self.T_clnt_eva_in = T_clnt_out

        Q_cool = massflow_clnt * self.capacity_clnt * (self.T_clnt_in - T_clnt_out)
        return Q_cool

    def get_efficiency(self, n_norm):
        eta_vol = sum(c * power(n_norm, i) for i, c in enumerate(self.eta_vol_coeff))
        eta_hyd = sum(c * power(n_norm, i) for i, c in enumerate(self.eta_hyd_coeff))
        # 使用符号化max/min限制范围
        eta_vol = min(max(eta_vol, 0.65), 0.95)
        eta_hyd = min(max(eta_hyd, 0.6), 0.92)
        return eta_vol, eta_hyd

    def calculate_delta_P(self, massflow):
        A = PI * power(self.pipe_diameter/2, 2)
        v = massflow / (self.rho_clnt * A)
        Re = self.rho_clnt * v * self.pipe_diameter / self.mu_clnt

        # 符号化条件替换：Re == 0 ?
        is_Re_zero = logical_eq(Re, 0)
        a = power(2.457 * log(1/(power(7/Re, 0.9) + 0.27e-6/self.pipe_diameter)), 16)
        b = power(37530/Re, 16)
        f = 8 * power(power(8/Re, 12) + 1/power(a + b, 1.5), 1/12)
        delta_P = if_else(
            is_Re_zero,
            0.0,
            (f * self.pipe_length/self.pipe_diameter * 0.5*self.rho_clnt*power(v,2) 
            + self.K_factors * 0.5*self.rho_clnt*power(v,2))
        )
        return delta_P

    def get_massflow(self,n_pump):
        n_norm = n_pump / 5000.0
        eta_vol, eta_hyd = self.get_efficiency(n_norm)
        theoretical_flow = self.V_pump * n_pump * self.rho_clnt / 60
        actual_flow = theoretical_flow * eta_vol
        massflow_clnt = actual_flow
        return massflow_clnt, eta_hyd

    def pump_power(self, n_pump):
        actual_flow, eta_hyd = self.get_massflow(n_pump)
        delta_P = self.calculate_delta_P(actual_flow)
        
        # 符号化条件替换：actual_flow <= 1e-6 ?
        P_shaft = if_else(
            logical_le(actual_flow, 1e-6),
            0.0,
            (delta_P * actual_flow) / (eta_hyd * self.rho_clnt)
        )
        return P_shaft
    

def test_pump_model():
    """
    测试泵模型的功能和结果是否合理
    """
    # 初始化泵模型参数
    pump_model = CoolingSystem(dt=0.1, T_amb=25)

    # 测试不同转速下的结果
    for n_pump in [1000, 2000, 3000, 4000, 5000]:  # 转速 (rev/min)
        P_pump = pump_model.pump_power(n_pump)
        massflow_clnt = pump_model.massflow_clnt
        Q_cool = pump_model.battery_cooling(30)

        # 输出结果
        print(f"转速: {n_pump} rpm")
        print(f"质量流速: {massflow_clnt:.6f} kg/s")
        print(f"泵功率: {P_pump:.2f} W")
        print(f"制冷量: {Q_cool:.2f} W")
        print("-" * 30)


# 运行测试
# test_evaporator()

if __name__ == "__main__":
    test_pump_model()