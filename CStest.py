from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem
from Controller.MPC_for_ES import MPCController
import numpy as np
from EnergyStorageSystem.TargetPower import POWER as P_RE
from SOH.inference import SOH_predictor
import numpy as np


if __name__ == "__main__":
    dt = 1
    cs = SimpleCoolingSystem(dt, T_amb=35)
    Q_cool_list = []
    print(cs.lambda1, cs.lambda2, cs.lambda3, cs.lambda4, cs.lambda5, cs.lambda6)
    for i in [400, 450, 500, 1000, 2000, 3000, 4000 ,5000, 6000, 7000]:
        Q_cool = cs.battery_cooling(25, i)
        Q_cool_list.append(Q_cool)
    print(Q_cool_list)
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
