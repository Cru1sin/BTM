import numpy as np
import casadi as ca
from battery_model_theo import Battery_Model
from CS_for_ES import SimpleCoolingSystem
from vehicle_dynamics_model import Vehicle
from grid_price import get_electricity_price

class MPCController:
    """
    使用CasADi的Opti()实现电池热管理MPC控制器。
    
    主要变量：
    - 状态变量: 电池温度 T_bat
    - 控制变量: 泵转速 n_pump
    - 干扰变量: 车辆牵引功率 P_trac
    
    目标：最小化电池温度偏差（T_bat - T_opt）和控制能耗（n_pump）
    """
    
    def __init__(self, battery_model, cooling_system, 
                initial_temp=30, T_opt=25, N=10, dt=1, P_comp_limits=(0, 4500), Q = 1e5, R = 1e-2):
        """
        初始化电池温度MPC控制器
        - param battery_model: 电池热模型实例
        - param cooling_system: 冷却系统模型实例
        - param initial_temp: 初始电池温度 (℃)
        - param T_opt: 目标电池温度（优化目标）
        - param N: 预测时域步数
        - param dt: 离散时间步长(s)
        - param P_comp_limits: 压缩机功率范围 (min, max)
        """
        self.bm = battery_model
        self.cs = cooling_system

        self.initial_temp = initial_temp
        self.T_opt = T_opt
        self.N = N
        self.dt = dt
        self.P_min, self.P_max = P_comp_limits

        # CasADi优化器
        self.opti = ca.Opti()

        self.Q = 1.0  # 温度误差权重
        self.R = 5e-7  # 控制能耗权重
        self.S = 5e-6 # 控制变化率权重
        self.w = 1e-3  # 电价权重

        # 状态变量：电池温度, SOH, 功率
        self.opt_states = self.opti.variable(N + 1, 2)  # (N+1) x 1
        self.temp = self.opt_states[:, 0]  # 电池温度
        #self.SOH = self.opt_states[:, 1]  # 电池SOH
        self.P = self.opt_states[:, 1]  # 电池功率

        # 控制变量：压缩机功率, 电流
        self.opt_controls = self.opti.variable(N, 2)  # N x 1
        self.comp_power = self.opt_controls[:, 0]  # 压缩机功率
        self.current = self.opt_controls[:, 1]  # 电流

        # 参数：参考温度、牵引功率、车辆速度
        self.initial_state = self.opti.parameter(1, 2)
        self.opt_temp_ref = self.opti.parameter(N + 1, 1)  # 参考温度

        # 定义状态、控制和干扰变量
        self._build_optimization_problem()

    def _build_optimization_problem(self):
        """使用Opti()定义MPC优化问题，包括状态转移约束、目标函数和优化求解器。"""
        # === 1. 初始条件约束 ===
        self.opti.subject_to(self.opt_states[0, :] == self.initial_state[0, :])

        # === 2. 目标函数 ===
        obj = 0  # 初始化目标函数

        for i in range(self.N):
            # 获取当前时间步的电价
            p_buy, p_sell = self.get_electricity_price(i)

            # 计算冷却功率: battery_cooling(self, T_bat, P_comp, v_veh)
            Q_cool = self.cs.battery_cooling(self.temp[i], self.comp_power[i])

            # 计算下一步电池温度: battery_thermal_model(self, Q_cool, P_trac+P_cool+200, T_bat)
            #temp_next, P_next, SOH_next = self.bm.battery_thermal_model(self.current[i], Q_cool, self.temp[i], self.SOH[i])
            temp_next, P_next = self.bm.battery_thermal_model2(self.current[i], Q_cool, self.temp[i])


            # 状态转移约束
            self.opti.subject_to(self.opt_states[i + 1, 0] == temp_next)  # 温度
            self.opti.subject_to(self.opt_states[i + 1, 1] == P_next)  # 功率

            # 目标: 最小化温度偏差和控制能耗
            grid_cost = p_buy * ca.fmax(self.P[i], 0) + p_sell * ca.fmin(self.P[i], 0)
            obj += self.Q * (self.temp[i] - self.T_opt) ** 2 - self.w * grid_cost
            
        # 目标函数
        self.opti.minimize(obj)

        # === 3. 约束条件 ===
        self.opti.subject_to(self.opti.bounded(self.P_min, self.comp_power, self.P_max))  # 压缩机功率限制
        self.opti.subject_to(self.opti.bounded(15, self.temp, 45))  # 温度范围限制
        self.opti.subject_to(self.opti.bounded(-self.bm.max_I_charge, self.current, self.bm.max_I_discharge))  # 电流范围限制
        #self.opti.subject_to(self.opti.bounded(0, self.SOH, 1))  # SOH范围限制

        # === 4. 设定求解器 ===
        opts_setting = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6,
            },
            'print_time': False
        }
        self.opti.solver('ipopt', opts_setting)

    def solve(self, current_temp):
        """
        求解MPC优化问题，获取最优控制输入序列。
        :param current_temp: 当前电池温度
        :return: 最优控制输入 (压缩机功率)
        """
        # 构建参考温度轨迹 (N x 1)
        opt_temp_ref = np.full((self.N + 1, 1), self.T_opt)  # (N+1,1) 直接赋值目标温度

        # 设置参数
        self.opti.set_value(self.opt_temp_ref, opt_temp_ref)
        self.opti.set_value(self.initial_state, np.array([current_temp, 0]))

        # 设置初始猜测
        self.opti.set_initial(self.opt_states[:, 0], np.full((self.N + 1, 1), current_temp))  # 初始温度轨迹
        self.opti.set_initial(self.opt_states[:, 1], np.full((self.N + 1, 1),0))  # 其他状态初始化

        self.opti.set_initial(self.opt_controls[:, 0], np.full((self.N, 1), (self.P_min + self.P_max) / 2))
        self.opti.set_initial(self.opt_controls[:, 1], np.full((self.N, 1), 0))

        # 求解优化问题
        sol = self.opti.solve()

        # 解析解
        return {
            'control_sequence': sol.value(self.opt_controls),
            'state_trajectory': sol.value(self.opt_states)
        }
    
    def get_electricity_price(self, time_index):
        """根据不同时间段返回买电和卖电价格"""
        # 计算当前时间对应的小时
        hour = (time_index * self.dt) / 3600  # 秒数转换为小时
        hour = int(hour % 24)  # 确保 hour 在 0-23 之间

        if 9 <= hour < 11 or 15 <= hour < 17:  # 尖峰时段
            p_buy = 1.2  # 买电价格 (元/kWh)
        elif 8 <= hour < 9 or 17 <= hour < 23:  # 高峰时段
            p_buy = 1.0
        elif 13 <= hour < 15 or 23 <= hour < 24:  # 平段时段
            p_buy = 0.8
        else:  # 低谷时段
            p_buy = 0.5
        
        p_sell = 0.8 * p_buy  # 假设卖电价是买电价的80%

        return p_buy, p_sell

# === 使用示例 ===
if __name__ == "__main__":
    dt = 1
    bm = Battery_Model(dt)
    cs = SimpleCoolingSystem(dt, T_amb=26)

    mpc = MPCController(bm, cs, N=100, dt=1)

    T_current = 25.0  # 当前温度

    # 求解最优控制
    solution = mpc.solve(T_current)
    
    print("最优压缩机功耗序列:", solution['control_sequence'])
    print("预测温度轨迹:", solution['state_trajectory'])