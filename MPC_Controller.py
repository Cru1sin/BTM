import numpy as np
import casadi as ca
from Battery.battery_model import Battery_Model
from CoolingSystem.liquid_cooling_system import CoolingSystem
from Vehicle.vehicle_dynamics_model import Vehicle

class MPCController:
    """符合CasADi符号计算规范的MPC控制器"""
    def __init__(self, battery_model, cooling_system, vehicle_model,
                 T_opt=25, N=10, dt=1, n_pump_limits=(0, 3000)):
        self.bm = battery_model
        self.cs = cooling_system
        self.vc = vehicle_model
        self.T_opt = T_opt
        self.N = N
        self.dt = dt
        self.n_min, self.n_max = n_pump_limits
        
        # 构建CasADi优化问题
        self.opti = ca.Opti()
        self.n_states = 1  # 仅考虑 T_bat 作为状态变量
        self.n_controls = 1  # 控制变量 n_pump
        self.n_disturbance = 1  # v_veh 作为不可控变量

        self._build_optimization_problem()

    def _build_optimization_problem(self):
        # === 状态、控制和干扰变量 ===
        T_bat = ca.MX.sym("T_bat")  # 电池温度
        states = ca.vertcat(T_bat)

        n_pump = ca.MX.sym("n_pump")  # 泵转速
        controls = ca.vertcat(n_pump)

        P_trac = ca.MX.sym("P_trac")  # 车辆牵引功率（替代 v_t）
        disturbance = ca.vertcat(P_trac)

        # === 预测变量 ===
        U = self.opti.variable(self.n_controls, self.N)
        X = self.opti.variable(self.n_states, self.N + 1)
        P = self.opti.parameter(self.n_disturbance * (self.N + 1) + self.n_states)  # P_trac 作为干扰参数

        # === 目标函数与约束 ===
        obj = 0
        g = []

        # 初始状态约束
        g.append(X[:, 0] - P[self.n_disturbance * (self.N + 1):])

        # 滚动优化
        for k in range(self.N):
            P_trac = P[k]  # 直接从 disturbance 读取牵引功率
            T_curr = X[:, k]
            n_curr = U[:, k]

            # 计算冷却功率
            Q_cool = self.cs.battery_cooling(n_curr, T_curr)

            # 计算电池温度更新
            T_next = self.bm.battery_thermal_model(Q_cool, P_trac, T_curr)

            # 状态转移方程
            g.append(X[:, k+1] - T_next)

            # 目标函数：最小化温度偏差和功耗
            obj += 1e4 * (T_curr - self.T_opt)**2 + 1e-3 * (n_curr**2)

        # === 约束设置 ===
        opt_variables = ca.vertcat(U.reshape((-1, 1)), X.reshape((-1, 1)))

        nlp_prob = {
            'f': obj,
            'x': opt_variables,
            'g': ca.vertcat(*g),
            'p': P
        }

        solver_opts = {
            'ipopt': {
                'max_iter': 2000,
                'print_level': 0,
                'acceptable_tol': 1e-6,
                'linear_solver': 'mumps'
            },
            'print_time': False
        }

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, solver_opts)
    
    def solve(self, current_state):
        """求解最优控制序列"""

        # 生成随机牵引功率
        traction_powers = [self.vc.traction() for _ in range(self.N+1)]

        p = np.concatenate([
            traction_powers,  # 车辆牵引功率序列
            [current_state]  # 初始温度
        ])

        # 验证参数维度
        assert len(p) == self.n_disturbance*(self.N+1) + self.n_states

        # 变量约束
        lbx = [self.n_min] * self.N + [20] * (self.N+1)  # 最小约束
        ubx = [self.n_max] * self.N + [45] * (self.N+1)  # 最大约束

        # 求解
        sol = self.solver(
            x0=self._get_initial_guess(),
            lbx=lbx,
            ubx=ubx,
            p=p
        )

        return self._parse_solution(sol)

    def _get_initial_guess(self):
        """生成合理的初始猜测"""
        return np.concatenate([
            np.ones(self.N) * (self.n_min + self.n_max) / 2,
            np.ones(self.N+1) * self.T_opt
        ])

    def _parse_solution(self, sol):
        """解析求解结果"""
        return {
            'control_sequence': sol['x'][:self.N].full().flatten(),
            'state_trajectory': sol['x'][self.N:].full().flatten()
        }

# === 使用示例 ===
if __name__ == "__main__":
    dt = 1
    bm = Battery_Model(dt)
    cs = CoolingSystem(dt, T_amb=25)
    ev = Vehicle(dt)

    mpc = MPCController(bm, cs, ev, N=10, dt=1)

    T_current = 30.0  # 当前温度

    # 求解最优控制
    solution = mpc.solve(T_current)

    print(f"最优泵转速序列: {solution['control_sequence']}")
    print(f"预测温度轨迹: {solution['state_trajectory']}")