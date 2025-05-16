from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem
from Controller.MPC_for_ES import MPCController
import numpy as np
from EnergyStorageSystem.TargetPower import POWER as P_RE
from SOH.inference import SOH_predictor
import numpy as np

def estimate_battery_capacity(noisy_power, P_target):
    """
    估算平滑目标功率所需的电池容量（单位：Ah）

    :param noisy_power: 每秒发电功率序列 (W)，numpy array，长度为总秒数
    :param P_target: 目标稳定输出功率 (W)
    :param V_bat: 电池工作电压 (V)
    :return: 所需电池容量 (Ah)，及电量变化轨迹 (Wh)
    """

    # 每秒功率差
    P_diff = noisy_power - P_target  # W

    # 电量变化轨迹（单位：Wh），Δt = 1s = 1/3600 h
    E_wh = np.cumsum(P_diff) / 3600.0

    # 所需电池容量是最大电量波动范围
    E_required = (np.max(E_wh) - np.min(E_wh))/1000  # 单位 kWh
    return E_required

if __name__ == "__main__":
    SOH_Predictor = SOH_predictor(dt=20, charge_time=100, cycle_num=100)
    data = [0.2,3.7,25]
    predictions = SOH_Predictor.inference(*data)
    print(predictions)
    raise Exception("stop")
    dt = 5
    bm = Battery(dt)
    cs = SimpleCoolingSystem(dt, T_amb=26)
    bm.update_module_parameters(I_cell=0.1, T_bat=35)
    print(f'R_cell = {bm.R_cell}')
    print(f'I_limit = {bm.I_max_limit}, {bm.I_min_limit}, {bm.max_I_charge}, {bm.max_I_discharge}')
    print(f'OCV = {bm.OCV}')
    mpc = MPCController(bm, cs, N=60, dt=dt)

    T_current = 35.0  # 当前温度

    # 求解最优控制
    mpc.opti.set_value(mpc.initial_state, np.array([T_current, 142.3*1000, 0.5]))
    solution = mpc.solve(T_current)
    
    print("最优控制序列:", solution['control_sequence'])
    print("状态轨迹:", solution['state_trajectory'])

    """# 滚动优化
    current_temp = 30
    for i in range(100):
        mpc_controller.solve(current_temp)
        current_temp = mpc_controller.battery.T_bat
        print(f"第{i+1}次迭代，电池温度为{current_temp}℃")"""
