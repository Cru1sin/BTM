import numpy as np
import casadi as ca
from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem as CoolingSystem
import time

class MPCController:
    """
    使用CasADi的Opti()实现电池热管理MPC控制器。
    
    主要变量：
    - 状态变量: 电池温度 T_bat, SOC
    - 控制变量: 压缩机功率 P_comp
    - 干扰变量: 可再生能源发电功率 P_RE
    
    目标：
    1. 最小化电池温度偏差（T_bat - T_opt）
    2. 最小化控制能耗（P_comp）
    3. 保证电解槽恒定功率供给
    """
    
    def __init__(self, battery_model, cooling_system, P_RE,
                initial_temp=30, T_opt=25, P_target=142.3*1000, N=24, dt=3600, P_comp_limits=(0, 4000)):
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

        # 状态变量：电池温度, SOC
        self.opt_states = self.opti.variable(N + 1, 3)  # (N+1) x 3
        self.temp = self.opt_states[:, 0]  # 电池温度
        self.P_response = self.opt_states[:, 1]  # 电池pack的功率
        self.SOC = self.opt_states[:, 2]   # 电池pack的SOC

        # 控制变量：压缩机功率
        self.opt_controls = self.opti.variable(N, 2)  # N x 1
        self.comp_power = self.opt_controls[:, 0]  # 压缩机功率
        self.I_pack = self.opt_controls[:, 1]  # 电池pack的电流
        
        # 参数：初始状态、压缩机功率
        self.initial_state = self.opti.parameter(1, 3) 
        self.initial_comp_power = self.opti.parameter(1, 1)
        self.opt_temp_ref = self.opti.parameter(N + 1, 1)  # 参考温度
        self.P_RE = P_RE  # 可再生能源发电功率

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
            P_cool = self.comp_power[i] + 200

            # 计算功率缺口和所需电流
            P_gap = self.P_RE[i*self.dt] + P_cool
            I_pack_need = self.bm.Current_Pack2Cell(P_gap)

            # 计算下一步电池温度和SOC
            temp_next, P_response, SOC_next, _, _ = self.bm.battery_model(
                Q_cool=Q_cool, 
                I_pack=self.I_pack[i], 
                T_bat=self.temp[i], 
                SOC=self.SOC[i]
            )

            # 状态转移约束
            self.opti.subject_to(self.opt_states[i + 1, 0] == temp_next)  # 温度
            self.opti.subject_to(self.opt_states[i + 1, 1] == P_response)  # 电池pack的功率
            self.opti.subject_to(self.opt_states[i + 1, 2] == SOC_next)   # SOC

            # 动态约束
            if i == 0:
                self.opti.subject_to(self.opti.bounded(self.bm.I_min_limit+1, self.I_pack, self.bm.I_max_limit-1))

            # 权重系数
            omega_temp = 1000     # 温度偏差权重
            omega_dcomp = 1e-3    # 压缩机功率变化率权重
            omega_comp = 1e-5    # 压缩机功率权重
            omega_I_pack = 0  # 电池pack的电流权重
            # 温度优化目标
            temp_obj = omega_temp * (self.temp[i+1] - self.T_opt)**2

            # 给超过comp_power限制的comp_power非常大的约束
            omega_comp_limit = 1e12
            bound_comp_obj1 = omega_comp_limit * ca.fmin(0, self.comp_power[i])**2
            bound_comp_obj2 = omega_comp_limit * ca.fmax(4500, self.comp_power[i])**2


            # 压缩机功率变化率目标
            if i > 0:
                delta_comp_power_obj = omega_dcomp * (self.comp_power[i] - self.comp_power[i-1])**2
            else:
                delta_comp_power_obj = omega_dcomp * (self.comp_power[i] - self.initial_comp_power)**2

            # 压缩机功率目标
            comp_power_obj = omega_comp * self.comp_power[i]**2

            # 电池pack的电流目标
            I_pack_obj = omega_I_pack * (self.I_pack[i] - I_pack_need)**2

            # 累加目标函数
            obj += temp_obj + delta_comp_power_obj + comp_power_obj + I_pack_obj

        # 设置目标函数
        self.opti.minimize(obj)

        # === 3. 约束条件 ===
        # 压缩机功率限制
        self.opti.subject_to(self.opti.bounded(self.P_min, self.comp_power, self.P_max))
        
        # SOC范围限制
        self.opti.subject_to(self.opti.bounded(self.bm.SOC_min, self.SOC, 0.99))
        
        # 温度限制
        #self.opti.subject_to(self.opti.bounded(23, self.temp, 27))

        # === 4. 设定求解器 ===
        opts_setting = {
            'ipopt': {
                'max_iter': 500,
                'print_level': 0,
                'acceptable_tol': 1e-4,             # 宽松但合理的容差
                'acceptable_obj_change_tol': 1e-4,
                'tol': 1e-6,                        # 严格解的容差
                'bound_push': 1e-4,                 # 防止变量紧贴约束
                'bound_frac': 1e-3                 # 防止变量初始就位于约束边界
            },
            'print_time': False
        }
        self.opti.solver('ipopt', opts_setting)

    def solve(self, i, current_temp, SOC, comp_power):
        """
        求解MPC优化问题，获取最优控制输入序列。
        :param i: 当前时间步
        :param current_temp: 当前电池温度
        :param SOC: 当前SOC
        :param comp_power: 当前压缩机功率
        :return: 最优控制输入序列
        """
        # 构建参考温度轨迹
        opt_temp_ref = np.full((self.N + 1, 1), self.T_opt)

        # 设置参数
        self.opti.set_value(self.opt_temp_ref, opt_temp_ref)
        self.opti.set_value(self.initial_state, [float(current_temp), self.P_RE[i*self.dt],float(SOC)])
        self.opti.set_value(self.initial_comp_power, float(comp_power))

        # 设置初始猜测
        self.opti.set_initial(self.opt_states[:, 0], np.full((self.N + 1, 1), current_temp))  # 温度
        self.opti.set_initial(self.opt_states[:, 1], np.full((self.N + 1, 1), self.P_RE[i*self.dt]))   # 电池pack的功率
        self.opti.set_initial(self.opt_states[:, 2], np.full((self.N + 1, 1), float(SOC)))   # SOC
        self.opti.set_initial(self.opt_controls[:, 0], np.full((self.N, 1), 2000))     # 压缩机功率
        self.opti.set_initial(self.opt_controls[:, 1], np.full((self.N, 1), 0))     # 电池pack的电流

        sol = self.opti.solve()
        return {
            'control_sequence': sol.value(self.opt_controls),
            'state_trajectory': sol.value(self.opt_states)
        }
    
    def multi_solve(self, i, current_temp, SOC, comp_power):
        """
        多重求解
        """
        max_retries = 4
        self.opti.set_value(self.initial_state, [float(current_temp), self.P_RE[i*self.dt],float(SOC)])
        self.opti.set_value(self.initial_comp_power, float(comp_power))
        self.opti.set_initial(self.opt_states[:, 1], np.full((self.N + 1, 1), self.P_RE[i*self.dt]))   # 电池pack的功率
        self.opti.set_initial(self.opt_states[:, 2], np.full((self.N + 1, 1), float(SOC)))   # SOC
        temp_guess = np.linspace(current_temp, 25, self.N + 1).reshape(-1, 1)
        # dt=n，隔n个点取一个点
        P_need = self.P_RE[i*self.dt : i*self.dt + self.N*self.dt]  # 长度为 N
        P_need = P_need[::self.dt]
        current_guess = P_need / 80 / 3.7
        if current_temp > 25:
            if comp_power < 500:
                comp_power_towards = 750
            else:
                comp_power_towards = 2000
        else:
            if comp_power < 500:
                comp_power_towards = 750
            elif comp_power > 4000:
                comp_power_towards = 0
            else:
                comp_power_towards = 500
        comp_power_guess1 = np.linspace(comp_power, comp_power_towards, self.N).reshape(-1, 1)  # 逐渐减小
        # comp_power的变化趋势与P_RE[i*self.dt]～P_RE[i*self.dt+self.N*self.dt] 变化趋势一致
        # 转成 numpy 数组并归一化（或保留原始比例）并设置为正数
        P_RE_need = np.array(P_need).reshape(-1, 1)
        P_RE_need = np.abs(P_RE_need)
        # 按比例缩放到压缩机功率范围（比如 1000W ~ 4000W）
        min_RE, max_RE = np.min(P_RE_need), np.max(P_RE_need)
        if max_RE != min_RE:
            norm = (P_RE_need - min_RE) / (max_RE - min_RE)  # 归一化到 [0,1]
        else:
            norm = np.zeros_like(P_RE_need)  # 或者全 0.5
        comp_power_guess2 = norm * (4000)  # 缩放到 [0, 4000]
        for attempt in range(max_retries):
            if attempt == 0:
                comp_power_guess = comp_power_guess1
            elif attempt == 1:
                comp_power_guess = np.clip(comp_power_guess, 0, 4000)
                print(f'{x}, length: {len(x)}')
                temp_guess = x[:,0]
                current_guess = u[:,1]
            elif attempt == 2: # 随机生成
                comp_power_guess = comp_power_guess2
            elif attempt == 3:
                comp_power_guess = np.clip(comp_power_guess, 0, 4000)
                temp_guess = x[:,0]
                current_guess = u[:,1]
            try:
                self.opti.set_initial(self.opt_states[:, 0], temp_guess)  # 温度
                self.opti.set_initial(self.opt_controls[:, 0], comp_power_guess)     # 压缩机功率
                self.opti.set_initial(self.opt_controls[:, 1], current_guess)     # 电池pack的电流
                start_time = time.time()
                sol = self.opti.solve()
                end_time = time.time()
                print(f"Attempt {attempt + 1} time: {(end_time - start_time)*1000} ms")
                return {
                    'control_sequence': sol.value(self.opt_controls),
                    'state_trajectory': sol.value(self.opt_states)
                }
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                x = self.opti.debug.value(self.opt_states)
                u = self.opti.debug.value(self.opt_controls)
                if attempt == 0:
                    print(f'需求功率：{P_need}')
                
        return None


# === 使用示例 ===
if __name__ == "__main__":
    dt = 1
    bm = Battery(dt)
    cs = CoolingSystem(dt, T_amb=26)

    mpc = MPCController(bm, cs, N=24, dt=3600)

    T_current = 25.0  # 当前温度

    # 求解最优控制
    solution = mpc.solve(0, T_current, 0.5, 2000)
    
    print("最优控制序列:", solution['control_sequence'])
    print("状态轨迹:", solution['state_trajectory'])