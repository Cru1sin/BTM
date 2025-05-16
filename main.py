import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import argparse
from datetime import datetime
from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem
from Controller.MPC_for_ES import MPCController
from EnergyStorageSystem.TargetPower import TARGET
from utils import plot_results
from SOH.inference import SOH_predictor

def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='电池热管理系统MPC控制仿真')
    
    # 仿真参数
    parser.add_argument('--dt', type=int, default=1, help='时间步长（秒）')
    parser.add_argument('--N', type=int, default=100, help='MPC预测时域')
    parser.add_argument('--n_control', type=int, default=10, help='每次应用的控制步数')
    parser.add_argument('--total_steps', type=int, default=3600, help='总仿真步数（秒）')
    parser.add_argument('--max_retries', type=int, default=5, help='最大重试次数')
    
    # 系统初始状态
    parser.add_argument('--init_temp', type=float, default=25.0, help='电池初始温度（℃）')
    parser.add_argument('--init_soc', type=float, default=0.2, help='电池初始SOC')
    parser.add_argument('--T_amb', type=float, default=35.0, help='环境温度（℃）')
    
    
    # 系统参数
    parser.add_argument('--target_power', type=float, default=142.3e3, help='目标功率（W）')
    parser.add_argument('--BTM_power_base', type=float, default=200.0, help='BTMS基础功率（W）')
    
    # 温度扰动参数
    parser.add_argument('--temp_noise_std', type=float, default=0.01, help='温度高斯扰动标准差（℃）')
    
    # 日志和结果保存
    parser.add_argument('--log_interval', type=int, default=1, help='日志记录间隔（秒）')
    parser.add_argument('--save_interval', type=int, default=3600, help='结果保存时间（秒）')
    parser.add_argument('--SOH_interval', type=int, default=30, help='SOH损耗记录间隔（秒）')
    
    return parser.parse_args()

def setup_logging():
    """设置日志系统"""
    # 创建主日志文件夹
    base_log_dir = "logs"
    if not os.path.exists(base_log_dir):
        os.makedirs(base_log_dir)
    
    # 创建带时间戳的唯一子文件夹
    run_dir = os.path.join(base_log_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    
    # 设置日志记录
    log_file = os.path.join(run_dir, "mpc_control.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return run_dir

def initialize_system(args):
    """
    初始化系统模型和控制器
    参数:
        args: 命令行参数对象
    返回:
        bm_for_control: 控制器使用的电池模型
        bm_for_simulation: 仿真使用的电池模型
        cs_for_control: 控制器使用的冷却系统模型
        cs_for_simulation: 仿真使用的冷却系统模型
        mpc: MPC控制器实例
    """
    # 初始化电池模型
    bm_for_control = Battery(args.dt, args.T_amb)
    bm_for_simulation = Battery(args.dt, args.T_amb)
    
    # 初始化冷却系统模型
    cs_for_control = SimpleCoolingSystem(args.dt, T_amb=args.T_amb)
    cs_for_simulation = SimpleCoolingSystem(args.dt, T_amb=args.T_amb)

    # 更新电池模型参数
    for i in [bm_for_control, bm_for_simulation]:
        i.update_parameters(I_cell=1e-8, T_bat=args.init_temp, SOC=args.init_soc)
    
    # 初始化MPC控制器
    mpc = MPCController(bm_for_control, cs_for_control, TARGET, N=args.N, dt=args.dt)
    
    # 记录初始参数
    log_system_parameters(args)
    return bm_for_simulation, cs_for_simulation, mpc

def log_system_parameters(args):
    """记录系统关键参数"""
    logging.info("系统初始化参数:")
    logging.info(f"仿真参数: dt={args.dt}秒, N={args.N}, n_control={args.n_control}, total_steps={args.total_steps}秒")
    logging.info(f"初始状态: 温度={args.init_temp}℃, SOC={args.init_soc}, 环境温度={args.T_amb}℃")

def update_state(i, args, cs, bm, current_temp, comp_power, SOC):
    """
    更新系统状态
    
    参数:
        i: 当前时间步
        args: 命令行参数对象
        cs: 仿真冷却系统模型
        bm: 仿真电池模型
        current_temp: 当前温度
        comp_power: 压缩机功率
        SOC: 当前SOC
    """
    # 计算冷却量
    Q_cool = cs.battery_cooling(current_temp, comp_power)
    P_cool = comp_power + args.BTM_power_base
    
    # 计算功率缺口和所需电流
    P_gap = TARGET[int(i*args.dt)] + P_cool
    I_pack = np.clip(bm.Current_Pack2Cell(P_gap), bm.I_min_limit, bm.I_max_limit)
    
    # 更新电池状态
    temp_next, P_response, SOC_next, Q_gen, Q_ambient = bm.battery_model(
        Q_cool=Q_cool, 
        I_pack=I_pack,
        T_bat=current_temp,
        SOC=SOC
    )
    
    # 添加温度高斯扰动
    temp_noise = np.random.normal(0, args.temp_noise_std)
    temp_next = temp_next + temp_noise  

    # 计算SOH损耗
    if i % args.SOH_interval == 0:
        SOH_loss = bm.get_SOH_loss(I_pack, current_temp)
    else:
        SOH_loss = 0

    # 记录状态信息
    if i % args.log_interval == 0:
        log_state_info(i, args, current_temp, comp_power, Q_cool, Q_gen, Q_ambient,
                      I_pack, P_response, SOC_next, bm, TARGET[int(i*args.dt)], SOH_loss, temp_noise)
    
    return temp_next, SOC_next, SOH_loss, I_pack

def log_state_info(i, args, current_temp, comp_power, Q_cool, Q_gen, Q_ambient,
                  I_pack, P_response, SOC_next, bm, TARGET, SOH_loss, temp_noise):
    """记录系统状态信息"""
    logging.info(f"\n第{(i+1)*args.dt}秒系统状态:")
    logging.info(f"温度: {float(current_temp):.2f}℃, 温度扰动: {float(temp_noise):.3f}℃")
    logging.info(f"压缩机功率: {float(comp_power):.2f}W")
    logging.info(f"冷却量: {float(Q_cool):.2f}W, 产热量: {float(Q_gen):.2f}W, 环境热交换: {float(Q_ambient):.2f}W")
    logging.info(f"电池电流: {float(I_pack):.2f}A, 电池功率: {float(P_response):.2f}W, 电池目标功率: {float(TARGET):.2f}W")
    logging.info(f"电池Cell电流限制: 放电{float(bm.max_I_discharge):.2f}A, 充电{float(bm.max_I_charge):.2f}A")
    logging.info(f"电池Pack电流限制: 放电{float(bm.I_max_limit):.2f}A, 充电{float(bm.I_min_limit):.2f}A")
    logging.info(f"SOC: {SOC_next*100:.2f}%")
    logging.info(f"电池电压: {float(bm.OCV - I_pack/bm.N_parallel*bm.R_cell):.2f}V")
    logging.info(f"SOH损耗: {float(SOH_loss)}")

def run_mpc_simulation(args, mpc, bm_for_simulation, cs_for_simulation, log_dir):
    """运行MPC仿真，solve失败时使用上一次状态重试"""
    # 初始化状态
    i = 0
    current_temp = args.init_temp
    comp_power = 0
    current_SOC = args.init_soc
    logging.info("开始MPC控制仿真")

    # 存储结果
    control_sequence = []
    state_trajectory = []
    time_points = []
    
    # 记录上一次成功求解的状态
    last_successful_state = {
        'temp': current_temp,
        'soc': current_SOC,
        'comp_power': comp_power,
        'step': i
    }
    
    # 滚动优化
    while i < args.total_steps:
        remaining_steps = args.total_steps - i
        if remaining_steps < mpc.N:
            break
        
        solution = mpc.multi_solve(i, float(current_temp), float(current_SOC), float(comp_power))

        if solution is None:
            logging.warning(f"使用上一次成功求解的计算结果")
        else:
            comp_power = solution['control_sequence'][j][0]
        # 应用控制并更新状态
        for j in range(args.n_control):
            if i + j >= args.total_steps:
                break
                
            current_temp, current_SOC, SOH_total_loss, I_pack = update_state(
                i+j, args, cs_for_simulation, bm_for_simulation,
                current_temp, comp_power, current_SOC
            )
            
            time_points.append(i+j+1)
            control_sequence.append([float(comp_power)])
            state_trajectory.append([float(current_temp), float(current_SOC), float(SOH_total_loss), float(I_pack)])
        
        i += args.n_control
    
    return np.array(control_sequence), np.array(state_trajectory), np.array(time_points)

def save_results(args, time_points, state_trajectory, log_dir):
    """保存仿真结果"""
    base_results_dir = "results"
    if not os.path.exists(base_results_dir):
        os.makedirs(base_results_dir)
    
    results_dir = os.path.join(base_results_dir, os.path.basename(log_dir))
    os.makedirs(results_dir, exist_ok=True)
    
    # 提取指定时间间隔的数据
    mask = time_points <= args.save_interval
    time_data = time_points[mask]
    temp_data = state_trajectory[mask, 0]
    soh_data = state_trajectory[mask, 3]
    
    # 保存温度数据
    temp_file = os.path.join(results_dir, "temperature.csv")
    np.savetxt(temp_file, 
              np.column_stack((time_data, temp_data)),
              delimiter=',',
              header='Time(s),Temperature(℃)',
              comments='')
    logging.info(f"温度数据已保存到: {temp_file}")
    
    # 保存SOH损耗数据
    soh_file = os.path.join(results_dir, "soh_loss.csv")
    np.savetxt(soh_file,
              np.column_stack((time_data, soh_data)),
              delimiter=',',
              header='Time(s),SOH_Loss',
              comments='')
    logging.info(f"SOH损耗数据已保存到: {soh_file}")

def main():
    """主函数"""
    # 1. 解析命令行参数
    args = get_args()
    
    # 2. 设置日志系统
    log_dir = setup_logging()
    logging.info("开始MPC控制仿真")
    
    # 3. 初始化系统
    bm_for_simulation, cs_for_simulation, mpc = initialize_system(args)
    
    # 4. 运行MPC仿真
    control_sequence, state_trajectory, time_points = run_mpc_simulation(
        args, mpc, bm_for_simulation, cs_for_simulation, log_dir
    )
    
    # 5. 检查数据一致性
    if len(control_sequence) != len(state_trajectory) or len(control_sequence) != len(time_points):
        logging.error(f"数据长度不一致: control={len(control_sequence)}, state={len(state_trajectory)}, time={len(time_points)}")
        return
    logging.info(f"仿真完成，数据长度: {len(control_sequence)}")
    
    # 6. 保存结果
    save_results(args, time_points, state_trajectory, log_dir)
    
    # 7. 绘制结果图表
    plot_results(time_points, control_sequence, state_trajectory, log_dir)
    
    logging.info("MPC控制仿真完成")

if __name__ == "__main__":
    main()