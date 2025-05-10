from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem
from Controller.MPC_for_ES import MPCController

if __name__ == "__main__":
    dt = 10
    bm = Battery(dt)
    cs = SimpleCoolingSystem(dt, T_amb=26)
    bm.update_module_parameters(I_cell=0.1, T_bat=35)
    print(f'R_cell = {bm.R_cell}')
    print(f'I_limit = {bm.I_max_limit}, {bm.I_min_limit}, {bm.max_I_charge}, {bm.max_I_discharge}')
    print(f'OCV = {bm.OCV}')
    mpc = MPCController(bm, cs, N=100, dt=dt)

    T_current = 35.0  # 当前温度

    # 求解最优控制
    solution = mpc.solve(T_current)
    
    print("最优控制序列:", solution['control_sequence'])
    print("状态轨迹:", solution['state_trajectory'])

    """# 滚动优化
    current_temp = 30
    for i in range(100):
        mpc_controller.solve(current_temp)
        current_temp = mpc_controller.battery.T_bat
        print(f"第{i+1}次迭代，电池温度为{current_temp}℃")"""
