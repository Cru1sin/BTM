import numpy as np
import casadi as ca
from battery_model_theo import Battery_Model
from sample import SimpleCoolingSystem
from vehicle_dynamics_model import Vehicle

class MPCController:
    """
    使用CasADi的Opti()实现电池热管理MPC控制器。
    
    主要变量：
    - 状态变量: 电池温度 T_bat
    - 控制变量: 泵转速 n_pump
    - 干扰变量: 车辆牵引功率 P_trac
    
    目标：最小化电池温度偏差（T_bat - T_opt）和控制能耗（n_pump）
    """
    
    def __init__(self, battery_model, cooling_system, vehicle_model, 
                initial_temp=30, T_opt=25, N=10, dt=1, P_comp_limits=(0, 4500), Q = 1e5, R = 1e-2):
        """
        初始化电池温度MPC控制器
        - param battery_model: 电池热模型实例
        - param cooling_system: 冷却系统模型实例
        - param vehicle_model: 车辆动力学模型实例
        - param initial_temp: 初始电池温度 (℃)
        - param T_opt: 目标电池温度（优化目标）
        - param N: 预测时域步数
        - param dt: 离散时间步长(s)
        - param P_comp_limits: 压缩机功率范围 (min, max)
        """
        self.bm = battery_model
        self.cs = cooling_system
        self.vc = vehicle_model

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

        # 状态变量：电池温度
        self.opt_states = self.opti.variable(N + 1, 1)  # (N+1) x 1
        self.temp = self.opt_states[:, 0]  # 电池温度
        # 控制变量：泵功率
        self.opt_controls = self.opti.variable(N, 1)  # N x 1
        self.comp_power = self.opt_controls[:, 0]  # 泵功率
        # 参数：参考温度、牵引功率、车辆速度
        self.opt_temp_ref = self.opti.parameter(N + 1, 1)  # 参考温度
        self.opt_traction_power = self.opti.parameter(N, 1)  # 牵引功率
        self.opt_vehicle_speed = self.opti.parameter(N, 1)  # 车辆速度

        # 定义状态、控制和干扰变量
        self._build_optimization_problem()

    def _build_optimization_problem(self):
        """使用Opti()定义MPC优化问题，包括状态转移约束、目标函数和优化求解器。"""
        # === 1. 初始条件约束 ===
        self.opti.subject_to(self.opt_states[0, :] == self.opt_temp_ref[0, :])

        # === 2. 目标函数 ===
        obj = 0  # 初始化目标函数

        for i in range(self.N):
            # 计算冷却功率: battery_cooling(self, T_bat, P_comp, v_veh)
            Q_cool = self.cs.battery_cooling(self.temp[i], self.comp_power[i], self.opt_vehicle_speed[i, 0])

            # 计算下一步电池温度: battery_thermal_model(self, Q_cool, P_trac+P_cool+200, T_bat)
            temp_next = self.bm.battery_thermal_model(Q_cool, self.opt_traction_power[i, 0] + self.comp_power[i] + 200, self.temp[i])

            # 状态转移约束
            self.opti.subject_to(self.temp[i + 1] == temp_next)

            # 目标: 最小化温度偏差和控制能耗
            obj += self.Q * (self.temp[i] - self.T_opt) ** 2 + self.R * (self.comp_power[i] ** 2)
            if i > 0:
                obj += self.S * (self.comp_power[i] - self.comp_power[i-1]) ** 2

        # 目标函数
        self.opti.minimize(obj)

        # === 3. 约束条件 ===
        self.opti.subject_to(self.opti.bounded(self.P_min, self.comp_power, self.P_max))  # 压缩机功率限制
        self.opti.subject_to(self.opti.bounded(15, self.temp, 45))  # 温度范围限制

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
        Temp_opt = np.full((self.N, 1), self.T_opt)  # 生成所有元素均为 T_opt 的数组
        opt_temp_ref = np.vstack((current_temp, Temp_opt)).reshape(-1, 1) 

        # 生成牵引功率和车辆速度的扰动序列
        traction_data = [self.vc.traction() for _ in range(self.N)]  # 生成 (P_trac, v_veh) 的列表
        traction_powers = np.array([data[0] for data in traction_data]).reshape(-1, 1)  # 提取 P_trac
        vehicle_speeds = np.array([data[1] for data in traction_data]).reshape(-1, 1)   # 提取 v_ve

        # 设置参数
        self.opti.set_value(self.opt_temp_ref, opt_temp_ref)
        self.opti.set_value(self.opt_traction_power, traction_powers)
        self.opti.set_value(self.opt_vehicle_speed, vehicle_speeds)

        # 设置初始猜测
        self.opti.set_initial(self.opt_states, np.tile(current_temp, (self.N + 1, 1)))
        self.opti.set_initial(self.opt_controls, np.full((self.N, 1), (self.P_min + self.P_max) / 2))

        # 求解优化问题
        sol = self.opti.solve()

        # 解析解
        return {
            'control_sequence': sol.value(self.opt_controls),
            'state_trajectory': sol.value(self.opt_states)
        }

# === 使用示例 ===
if __name__ == "__main__":
    dt = 1
    bm = Battery_Model(dt)
    cs = SimpleCoolingSystem(dt, T_amb=26)
    ev = Vehicle(dt)

    mpc = MPCController(bm, cs, ev, N=100, dt=1)

    T_current = 25.0  # 当前温度

    # 求解最优控制
    solution = mpc.solve(T_current)
    
    print("最优压缩机功耗序列:", solution['control_sequence'])
    print("预测温度轨迹:", solution['state_trajectory'])