from CoolProp.CoolProp import PropsSI
import numpy as np
PI = 3.14



class CoolingSystem: 
    """
    包括液体制冷循环和液体冷却循环
    制冷循环: 压缩机 -> 冷凝器 -> 膨胀阀 -> 换热器
    冷却循环: 换热器 -> 电池 -> 泵
    ----------------------------
    控制变量:
    1. 压缩机转速
    2. 泵转速
    ----------------------------
    假设:
    1. 系统能够快速达到稳定状态，没有温度压力的剧烈波动; 系统初始状态为稳定状态
    2. 压缩机: 恒压比压缩机，控制压缩机转速仅控制制冷循环质量流速, 各部件前后压力值和焓值均设为常数
    3. 冷凝板: 不考虑风扇功率，能将冷凝板液体出口焓值控制在设定值
    4. 膨胀阀: 未建模，考虑为等焓压降过程
    5. 换热器: 
    -----------------------------
    参考: Battery thermal management of intelligent-connected electric vehicles at low temperature based on NMPC
    """
    def __init__(self, N, dt, T_amb):
        self.T_bat = T_amb # 初始化电池温度
        self.A_bat = 12   # 电池包的冷却表面积(m^2) 假设电池包尺寸为 1.0 * 1.0 * 0.3
        self.h_bat = 300 # Heat transfer coefficient h (W/(m2 ℃))

        '''
        Coolant
        40% 乙二醇 + 60% 水的混合液
        '''
        self.rho_clnt = 1069.5  # Density of coolant (kg/m^3)
        self.capacity_clnt = 3330 # Specific heat capacity of coolant (J/kg·K)
        self.V_pump = 33*1e-6 # Displacement volume of pump (m^3/rev)
        self.h_bat = 300 # Heat transfer coefficient between the battery and the coolant (W/m^2/K)
        self.A_bat = 1 # Heat transfer sectional area between the battery and the coolant (m^2)
        self.T_wi = T_amb # the coolant temperature at the inlet of the pipe in battery pack
        self.T_wo = T_amb # the coolant temperature at the outlet of the pipe in battery pack
        self.T_wo_next = T_amb # 用于计算 出口温度的变化量, 在换热器计算中更新，在电池传热计算后更新成T_wo

        '''
        Refrigerant circle
        R-134a
        Compressor, Condenser, Expansion valve, Evaporator
        '''
        self.rho_rfg = 27.8   # Density of refrigerant (kg/m^3)
        self.capacity_rfg = 1117 # Specific heat capacity of refrigerant (J/kg·K)
        self.V_comp = 33 # Displacement volume of compressor (cm^3/rev)
        self.h_eva_rfg = 3000 # Heat transfer coefficient between evaporator and refrigerant/coolant (W/m^2/K)
        self.h_eva_clnt = 500 # Heat transfer coefficient between evaporator and refrigerant/coolant (W/m^2/K)
        self.A_eva = 0.3 # Heat transfer sectional area between evaporator and refrigerant/coolant (m^2)
        self.PR = 5 # Compression ratio of the compressor

        self.h_comp_out = 430 # Enthalpy at the outlet of compressor (kJ/Kg)
        self.P_comp_out = 2100  # Pressure at the outlet of compressor (kPa)

        self.h_eva_out = 410  # Enthalpy at the outlet of evaporator (kJ/Kg)
        self.P_eva = 600 # 换热器内气态制冷剂内的压力 (kPa)
        self.T_eva_rfg = T_amb # 初始化换热器内制冷剂平均温度

        self.h_cond_out = 300 # Enthalpy at the outlet of condenser (kJ/Kg)
        self.P_cond_out = 2100  # Pressure at the outlet of condenser (kPa)
        self.T_cond = T_amb    # 初始化冷凝器内制冷剂平均温度(℃)
        self.T_amb = T_amb        # 外界空气温度 (℃) , 冷凝器和外界空气换热
        self.capacity_air = 1005   # 空气比热容 (J/kg·°C)

        

        '''
        Control variables
        '''
        self.massflow_rfg = 0   # Mass flow rate of refrigerant (Kg/s)
        self.massflow_clnt = 0  # Mass flow rate of coolant (Kg/s)
        self.T_clnt_eva_in = self.T_amb # 假定的初始的进入蒸发器的冷却液温度 (℃)

        self.N = N # 时域 N , example: N = 30
        self.dt = dt # 采样时间 dt (s), example dt = 0.1

    def compressor(self, n_comp):
        """
        压缩机功率
        在计算出制冷循环质量流速之后
        低压气体 -> 高压气体
        """
        v1 = 1 / PropsSI("D", "P", self.P_eva * 1e3, "H", self.h_eva_out * 1e3, 'R134a')
        efficiency_vol = 0.966 * (1 - 0.05*((self.P_comp_out / self.P_eva)**(1/1.15) - 1))
        self.massflow_rfg = 1.66 * 1e-8 * self.V_comp * efficiency_vol * n_comp / v1
        P_comp = self.P_eva * 1e3 * v1 * 1.15 / 0.15 * ((self.P_comp_out / self.P_eva)**(0.15/1.15) - 1) * self.massflow_rfg
        return P_comp
    
    def condenser(self):
        A_cond = 14 # Heat transfer area (m2)
        V_cond = 0.0045 # Capacity of refrigerant
        c_air = 1003 # Specific heat capacity of air ca (J/(kg ,℃))
        K_cond = 55 # Heat transfer coefficient Ke (W/(m2 ,℃))

        m_cond = V_cond * self.rho_rfg
        mc = m_cond * self.capacity_rfg
        KA = K_cond * A_cond

        T_air_in = self.T_amb
        q0 = PropsSI("H", "P", self.P_eva_in, "Q", 1, "R134a") - PropsSI("H", "P", self.P_eva_in, "Q", 0, "R134a")
        massflow_air = self.massflow_rfg
        T_air_out = -(self.massflow_rfg * q0) / (massflow_air * c_air) + T_air_in
        self.T_cond = self.T_cond + ((q0/mc) * self.massflow_rfg
                               + (KA/mc/2) * T_air_in
                               + (KA/mc/2) * T_air_out
                               - (KA/mc) * self.T_cond)

    def pump(self, n_pump):
        '''
        假设: 
        1. 冷却循环中，冷却剂无状态变化，假设密度不变
        2. Delta_P_pump 公式由https://doi.org/10.1016/j.energy.2021.122571给出
        3. 泵的容积效率和能量转化效率定为常数
        4. 泵不产生热量, 工质通过泵后温度不变, 整个冷却循环流速处处相同
        '''
        #omega_pump = 2 * PI / 60 * n_pump
        # eta_pump = 1e-5 * omega_pump + 0.9
        efficiency_vol = 0.95 # 泵的容积效率
        self.massflow_clnt = self.V_pump * efficiency_vol * n_pump * self.rho_clnt / 60

        '''Delta_pump = self.massflow_clnt ** 2 * 1000 # Pressure Drop (Pa)
        torque_pump = Delta_pump * self.massflow_clnt / self.rho_clnt / (omega_pump * (2 * math.pi / 60))
        eta_p = min(1, 0.02 * torque_pump + 0.6)
        P_pump = torque_pump * (omega_pump * (2 * math.pi / 60)) / eta_p'''

        efficiency_power = 0.75 # 泵的能量转化效率
        Delta_P_pump = (0.927 * self.massflow_clnt ** 2 + 0.586 * self.massflow_clnt - 0.143) * 1000
        P_pump = Delta_P_pump * self.massflow_clnt / efficiency_power / self.rho_clnt
        return P_pump

    def evaporator(self):
        """
        换热器: 制冷剂与冷却液交换热量，制冷剂吸热气化，冷却液物态不变
        -------------------------------------
        显式欧拉法: 需要选取合适的采样时间步长
        制冷剂 -> 显热 + 潜热 = 传热量
        冷却液 -> 
        -------------------------------------
        """
        A_c = 0.3  # Heat transfer area Ac (m2)
        V_c_rfg = 0.0071 # Capacity of refrigerant (m3)
        V_c_clnt = 0.0085 # Capacity of coolant (m3)
        K_c = 1000 # Heat transfer coefficient Kc (W/(m2 ,℃))

        rho_rfg = PropsSI("D", "P", self.P_eva * 1e3, "T", self.T_eva_rfg, "R134a") # 换热器制冷剂侧的气态制冷剂平均密度
        m_rfg = V_c_rfg * rho_rfg  # 制冷剂质量 (kg)
        m_clnt = V_c_clnt * self.rho_clnt # 冷却液质量 (kg)
        
        # 制冷剂在换热器中完全气化，气化潜热 = 气态焓值 - 液态焓值
        latent_heat = PropsSI("H", "P", self.P_eva_in, "Q", 1, "R134a") - PropsSI("H", "P", self.P_eva_in, "Q", 0, "R134a")

        # 换热器内制冷剂的平均温度
        mc_rfg = self.capacity_rfg * m_rfg
        mc_clnt = self.capacity_clnt * m_clnt
        KA = K_c * A_c
        self.T_eva_rfg = self.T_eva_rfg + ((latent_heat / mc_rfg) * self.massflow_rfg
                            + (KA/mc_rfg/2) * self.T_wo
                            + (KA/mc_rfg/2) * self.T_wi
                            - (KA/mc_rfg) * self.T_eva_rfg) * self.dt
        
        A_bat = 12 # 电池传热表面积
        h_bat = 300 # 电池与冷却液之间的传热系数
        heat_transfer_efficiency = np.exp(-(h_bat * A_bat)/(self.massflow_clnt * self.capacity_clnt))

        self.T_wo_next = (self.T_wi - self.T_bat) * heat_transfer_efficiency + self.T_bat
        Delta_T_wo = self.T_wo_next - self.T_wo
        
        self.T_wi = self.T_wi + (((2*self.capacity_clnt * self.massflow_clnt - KA)/(mc_clnt)) * self.T_wo
                                 - ((2*self.capacity_clnt * self.massflow_clnt + KA)/(mc_clnt)) * self.T_wi
                                 ((2*KA)/(mc_clnt)) * self.T_eva_rfg
                                 - Delta_T_wo) * self.dt
    
    def battery_cooling(self, T_bat):
        T_clnt_in = self.evaporator()
        if self.massflow_clnt == 0:
            T_clnt_out = T_bat
        else:
            T_clnt_out = (T_clnt_in - T_bat) * math.exp(-(self.h_bat * self.A_bat) / (self.massflow_clnt * self.capacity_clnt)) + T_bat
        self.T_clnt_eva_in = T_clnt_out  # 更新冷却循环中冷却液到电池吸热后出口的温度
        print("冷却液进出口温度分别为 = ", T_clnt_in, T_clnt_out)
        Q_cool = self.massflow_clnt * self.capacity_clnt * (T_clnt_in - T_clnt_out)
        return Q_cool

def test_compressor_model():
    """
    测试压缩机模型
    """
    # 创建压缩机模型实例，初始化参数
    compressor = CoolingSystem(N=30, dt=0.1, T_amb=25)
    # 定义测试转速范围 (rpm)
    test_speeds = [1000, 2000, 3000, 4000, 5000]

    # 遍历测试转速并打印结果
    for n_comp in test_speeds:
        P_comp = compressor.compressor(n_comp)
        print(f"转速: {n_comp} rpm")
        print(f"质量流速: {compressor.massflow_rfg:.6f} kg/s")
        print(f"压缩机功率: {P_comp:.2f} W")
        print("-" * 30)

def test_pump_model():
    """
    测试泵模型的功能和结果是否合理
    """
    # 初始化泵模型参数
    pump_model = CoolingSystem(N=30, dt=0.1, T_amb=25)

    # 测试不同转速下的结果
    for n_pump in [1000, 2000, 3000, 4000, 5000]:  # 转速 (rev/min)
        P_pump = pump_model.pump(n_pump)
        massflow_clnt = pump_model.massflow_clnt

        # 输出结果
        print(f"转速: {n_pump} rpm")
        print(f"质量流速: {massflow_clnt:.6f} kg/s")
        print(f"泵功率: {P_pump:.2f} W")
        print("-" * 30)

def test_evaporator():
    """
    测试换热器模型，检查 T_wo_next, T_wi, T_c 是否异常。
    """
    # 初始化参数
    Cooling_System = CoolingSystem(N=30, dt=0.1, T_amb=25)
    P_comp = Cooling_System.pump(n_pump=3000)
    P_pump = Cooling_System.compressor(n_comp=3000)
    # 执行换热器模拟
    for step in range(10):  # 模拟 10 个时间步长
        exchanger.evaporator()
        print(f"Step {step + 1}:")
        print(f"  T_c = {exchanger.T_c:.2f} ℃")
        print(f"  T_wi = {exchanger.T_wi:.2f} ℃")
        print(f"  T_wo_next = {exchanger.T_wo_next:.2f} ℃")

        # 检查异常
        if not (0 <= exchanger.T_c <= 100):
            print("Error: T_c out of range!")
            break
        if not (0 <= exchanger.T_wi <= 100):
            print("Error: T_wi out of range!")
            break
        if not (0 <= exchanger.T_wo_next <= 100):
            print("Error: T_wo_next out of range!")
            break

# 运行测试
# test_evaporator()

if __name__ == "__main__":
    test_compressor_model()
    test_pump_model()
    scm = simple_cooling_model(T_amb=25, dt=0.1, N=30)
    scm.thermal_para_rfg()