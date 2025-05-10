import numpy as np
import casadi as ca
from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem as CoolingSystem
from EnergyStorageSystem.TargetPower import P_RE_SECOND as P_RE

class MPCController:
    """
    使用CasADi的Opti()实现电池热管理MPC控制器。
    
    主要变量：
    - 状态变量: 电池温度 T_bat, 系统输出功率 P_out, 电池SOC
    - 控制变量: 压缩机功率 P_comp, 电池电流 I_bat
    - 干扰变量: 可再生能源发电功率 P_RE
    
    目标：
    1. 最小化电池温度偏差（T_bat - T_opt）
    2. 最小化控制能耗（P_comp）
    3. 最大化可再生能源利用率
    4. 保证电解槽恒定功率供给
    """
    
    def __init__(self, battery_model, cooling_system, 
                initial_temp=30, T_opt=25, P_target=142.3, N=24, dt=3600, P_comp_limits=(0, 4500)):
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
        self.P_target = P_target
        self.N = N
        self.dt = dt
        self.P_min, self.P_max = P_comp_limits

        # CasADi优化器
        self.opti = ca.Opti()

        # 权重系数
        self.Q = 1.0  # 温度误差权重
        self.R = 5e-7  # 控制能耗权重
        self.S = 5e-6  # 压缩机稳定性权重
        self.H = 1 # 制氢功率权重
        self.W = 1 # 弃电功率权重

        # 状态变量：电池温度, 系统输出功率, SOC
        self.opt_states = self.opti.variable(N + 1, 3)  # (N+1) x 3
        self.temp = self.opt_states[:, 0]  # 电池温度
        self.P = self.opt_states[:, 1]  # 系统输出功率
        self.SOC = self.opt_states[:, 2] # 电池pack的SOC

        # 控制变量：压缩机功率, 电流(charge, >0, discharge, <0)
        self.opt_controls = self.opti.variable(N, 2)  # N x 2
        self.comp_power = self.opt_controls[:, 0]  # 压缩机功率
        self.current = self.opt_controls[:, 1]  # 电流

        # 参数：参考温度、风光发电功率
        self.initial_state = self.opti.parameter(1, 3)
        self.opt_temp_ref = self.opti.parameter(N + 1, 1)  # 参考温度，暂时没用到
        self.P_RE = P_RE # 每小时可再生能源发电功率

        # 定义状态、控制和干扰变量
        self._build_optimization_problem()

    def _build_optimization_problem(self):
        """使用Opti()定义MPC优化问题，包括状态转移约束、目标函数和优化求解器。"""
        # === 1. 初始条件约束 ===
        self.opti.subject_to(self.opt_states[0, :] == self.initial_state[0, :])

        # === 2. 目标函数 ===
        obj = 0  # 初始化目标函数

        for i in range(self.N):
            # 计算冷却量和冷却功率
            Q_cool = self.cs.battery_cooling(self.temp[i], self.comp_power[i])
            P_cool = self.comp_power[i]+200

            # 计算下一步电池温度, 电池功率响应和SOC
            temp_next, P_response, SOC_next, Q_gen = self.bm.battery_model(Q_cool=Q_cool, I_pack=self.current[i], T_bat=self.temp[i])
            P_output = self.P_RE[i] - P_response # 系统输出功率 = 制氢功率（有效） + 弃电功率（无效）
            P_system = ca.fmin(P_output, self.P_target) # 制氢功率
            P_waste = ca.fmax(P_output - P_system, 0) # 发电功率超过电池载荷，弃电功率
            percent = (P_system - self.P_target + P_waste) / self.P_target

            # 状态转移约束
            self.opti.subject_to(self.opt_states[i + 1, 0] == temp_next)  # 温度
            self.opti.subject_to(self.opt_states[i + 1, 1] == percent)  # 系统输出功率
            self.opti.subject_to(self.opt_states[i + 1, 2] == SOC_next) # SOC

            # 动态约束
            self.opti.subject_to(self.opti.bounded(self.bm.I_min_limit, self.current[i], self.bm.I_max_limit))  # 电流范围限制
            self.opti.subject_to(self.opti.bounded(self.bm.SOC_min, self.SOC[i], 1))  # SOC范围限制
            if i>0:
                self.opti.subject_to(self.opti.bounded(-200, self.comp_power[i] - self.comp_power[i-1], 200))  # 压缩机功率范围限制

            # 目标函数各项:
            # 1. 温度控制目标
            temp_obj = self.Q * (self.temp[i] - self.T_opt) ** 2
            
            # 2. 控制能耗目标
            cool_obj = self.R * (self.comp_power[i] + 200) ** 2
            if i > 0:
                cool_obj += self.S * (self.comp_power[i] - self.comp_power[i-1]) ** 2
            
            # 3. 电解槽供电目标
            output_obj = self.H * (P_system - self.P_target)**2 + self.W * (P_waste)**2
            
            obj += temp_obj + cool_obj + output_obj
            
        # 目标函数
        self.opti.minimize(obj)

        # === 3. 约束条件 ===
        self.opti.subject_to(self.opti.bounded(0, self.comp_power, 4500))  # 压缩机功率限制
        self.opti.subject_to(self.opti.bounded(15, self.temp, 45))  # 温度范围限制
        self.opti.subject_to(self.opti.bounded(0, self.P, ca.inf))  # 系统输出功率限制


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
        self.opti.set_value(self.initial_state, np.array([current_temp, self.P_target, 0.5]))

        # 设置初始猜测
        self.opti.set_initial(self.opt_states[:, 0], np.full((self.N + 1, 1), current_temp))  # 初始温度轨迹
        self.opti.set_initial(self.opt_states[:, 1], np.full((self.N + 1, 1), self.P_target))  # 初始功率轨迹
        self.opti.set_initial(self.opt_states[:, 2], np.full((self.N + 1, 1), 0.5))  # 初始SOC轨迹

        self.opti.set_initial(self.opt_controls[:, 0], np.full((self.N, 1), 2000))
        self.opti.set_initial(self.opt_controls[:, 1], np.full((self.N, 1), 0))

        # 求解优化问题

        try:
            sol = self.opti.solve()
                    # 解析解
            
            return {
                'control_sequence': sol.value(self.opt_controls),
                'state_trajectory': sol.value(self.opt_states)
            }
        except RuntimeError as e:
            print("Solver failed:", e)
            # 显式调试每个变量
            print("u =", self.opti.debug.value(self.opt_controls))
            print("x =", self.opti.debug.value(self.opt_states))



# === 使用示例 ===
if __name__ == "__main__":
    dt = 1
    bm = Battery(dt)
    cs = CoolingSystem(dt, T_amb=26)

    mpc = MPCController(bm, cs, N=24, dt=3600)

    T_current = 25.0  # 当前温度

    # 求解最优控制
    solution = mpc.solve(T_current)
    
    print("最优控制序列:", solution['control_sequence'])
    print("状态轨迹:", solution['state_trajectory'])