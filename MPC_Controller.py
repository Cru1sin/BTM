import numpy as np
import casadi as ca
from battery_model import Battery_Model
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
                 T_opt=25, N=10, dt=1, n_pump_limits=(0, 3000)):
        """
        初始化MPC控制器。

        参数:
        - battery_model: 电池热模型实例
        - cooling_system: 冷却系统模型实例
        - vehicle_model: 车辆动力学模型实例
        - T_opt: 目标电池温度（优化目标）
        - N: 预测时域步数
        - dt: 离散时间步长
        - n_pump_limits: 泵的转速范围 (min, max)
        """
        self.bm = battery_model
        self.cs = cooling_system
        self.vc = vehicle_model
        self.T_opt = T_opt
        self.N = N
        self.dt = dt
        self.n_min, self.n_max = n_pump_limits

        # CasADi优化器
        self.opti = ca.Opti()

        # 定义状态、控制和干扰变量
        self._build_optimization_problem()

    def _build_optimization_problem(self):
        """使用Opti()定义MPC优化问题，包括状态转移约束、目标函数和优化求解器。"""
        
        # === 1. 定义变量 ===
        # 状态变量: 电池温度 T_bat (N+1步)
        opt_states = self.opti.variable(self.N+1, 1)  
        T_bat = opt_states[:, 0]  

        # 控制变量: 泵转速 n_pump (N步)
        opt_controls = self.opti.variable(self.N, 1)  
        n_pump = opt_controls[:, 0]  

        # 车辆牵引功率（干扰变量）(N+1步)
        opt_P_trac = self.opti.parameter(self.N+1, 1)  

        # 初始电池温度参数
        opt_T0 = self.opti.parameter(1, 1)

        # === 2. 目标函数 ===
        Q = 1e5  # 温度误差权重
        R = 1e-2  # 控制能耗权重
        obj = 0  # 初始化目标函数

        for i in range(self.N):
            # 计算冷却功率 (模型需提供 battery_cooling(n_pump, T_bat) 接口)
            Q_cool = self.cs.battery_cooling(n_pump[i], T_bat[i])
            P_cool = self.cs.power(n_pump[i])

            # 计算下一步电池温度 (模型需提供 battery_thermal_model(Q_cool, P_trac, T_bat) 接口)
            T_next = self.bm.battery_thermal_model(Q_cool, opt_P_trac[i] + P_cool, T_bat[i])

            # 约束: 状态更新
            self.opti.subject_to(T_bat[i+1] == T_next)

            # 目标: 最小化温度偏差和控制能耗
            obj += Q * (T_bat[i] - self.T_opt) ** 2 + R * (n_pump[i] ** 2)

        # 目标函数
        self.opti.minimize(obj)

        # === 3. 约束条件 ===
        self.opti.subject_to(self.opti.bounded(self.n_min, n_pump, self.n_max))  # 泵转速限制
        self.opti.subject_to(self.opti.bounded(20, T_bat, 45))  # 温度范围限制
        self.opti.subject_to(T_bat[0] == opt_T0)  # 初始条件约束

        # === 4. 设定求解器 ===
        opts_setting = {
            'ipopt': {
                'max_iter': 5000,
                'print_level': 0,
                'acceptable_tol': 1e-8,
                'linear_solver': 'mumps'
            },
            'print_time': False
        }
        self.opti.solver('ipopt', opts_setting)

    def solve(self, current_state):
        """
        求解MPC优化问题，获取最优控制输入序列。

        参数:
        - current_state: 当前电池温度
        
        返回:
        - 最优控制输入序列 n_pump
        - 预测的温度轨迹 T_bat
        """
        # 生成牵引功率扰动序列 (假设车辆模型提供 traction() 方法)
        traction_powers = np.array([self.vc.traction() for _ in range(self.N+1)]).reshape(-1, 1)

        # 设置参数值
        self.opti.set_value(self.opti.parameter(self.N+1, 1), traction_powers)  # 设置车辆牵引功率
        self.opti.set_value(self.opti.parameter(1, 1), current_state)  # 设置当前温度

        # 变量约束
        lbx = np.concatenate([np.full((self.N, 1), self.n_min), np.full((self.N+1, 1), 20)])
        ubx = np.concatenate([np.full((self.N, 1), self.n_max), np.full((self.N+1, 1), 45)])

        # 求解优化问题
        sol = self.opti.solve()

        # 解析解
        return {
            'control_sequence': sol.value(self.opti.variable(self.N, 1)).flatten(),
            'state_trajectory': sol.value(self.opti.variable(self.N+1, 1)).flatten()
        }

# === 使用示例 ===
if __name__ == "__main__":
    dt = 1
    bm = Battery_Model(dt)
    cs = SimpleCoolingSystem(dt, T_amb=25)
    ev = Vehicle(dt)

    mpc = MPCController(bm, cs, ev, N=10, dt=1)

    T_current = 30.0  # 当前温度

    # 求解最优控制
    solution = mpc.solve(T_current)
    
    print("最优泵转速序列:", solution['control_sequence'])
    print("预测温度轨迹:", solution['state_trajectory'])