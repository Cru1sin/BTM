import numpy as np
import os
import time
from datetime import datetime
from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem as CoolingSystem
from Controller.RB_controller import RBController

def load_data():
    data = np.genfromtxt('results/MPC_data.csv', delimiter=',')
    target_power = data[1:, 4]
    return target_power

def create_log_directory():
    """创建日志目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"results/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    return log_dir

def save_results(log_dir, data):
    """保存仿真结果到CSV文件"""
    # 保存数据
    print(f"data: {data[:5]}")
    header = "Time(s),Comp_Power(W),Temperature(℃),SOH_Loss(%)"
    np.savetxt(f"{log_dir}/data.csv", data, delimiter=',', header=header, comments='')

def main():
    # 创建日志目录
    log_dir = create_log_directory()
    
    # 仿真参数设置
    global dt, total_time, T_amb, initial_temp, initial_soc, P_target, rb_controller
    dt = 1.0                    # 时间步长(s)
    T_amb = 35.0               # 环境温度(℃)
    initial_temp = 25.0        # 初始电池温度(℃)
    initial_soc = 0.2          # 初始SOC
    delta_P = 300              # 压缩机功率变化量

    
    # 初始化模型和控制器
    bm = Battery(dt, T_amb=T_amb)
    cs = CoolingSystem(dt, T_amb=T_amb)
    target_power = load_data()
    rb_controller = RBController(bm, cs, target_power, dt=dt, delta_P=delta_P)
    
    # 初始化状态变量
    current_temp = initial_temp
    current_SOC = initial_soc
    comp_power = 0.0
    
    # 初始化数据记录数组
    n_steps = len(target_power)
    time_points = np.arange(0, n_steps, dt)
    
    data = []
    
    # 主控制循环
    for i in range(n_steps):
        # 获取控制输入
        comp_power, I_pack, current_temp, current_SOC = rb_controller.control(i, current_temp, current_SOC, comp_power)
        
        # 每30秒评估一次SOH损失
        if i % 30 == 0:
            soh_loss = float(bm.get_SOH_loss(I_pack, current_temp))
        else:
            soh_loss = 0

        # 添加温度噪声
        temp_noise = np.random.normal(0, 0.01)
        current_temp = current_temp + temp_noise  
        # 打印进度
        if i % 1 == 0:
            print(f"Step {i}/{n_steps}, 温度: {current_temp:.2f}℃, target_power: {target_power[i]:.2f}W, "
                  f"压缩机功率: {comp_power:.2f}W, SOC: {current_SOC:.3f}")
        
        data.append([time_points[i], comp_power, current_temp, soh_loss])
    
    # 保存结果
    save_results(log_dir, data)
    print(f"结果已保存到: {log_dir}")

if __name__ == "__main__":
    main()