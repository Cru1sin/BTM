import numpy as np
import os
import logging
from datetime import datetime
from .visualization import plot_power_comparison
from EnergyStorageSystem.TargetPower import P_RE_SECOND_C as P_RE

def calculate_statistics(time_points, state_trajectory, control_sequence, log_dir):
    """
    计算并输出系统运行统计信息
    """
    # 提取数据
    temperatures = state_trajectory[:, 0]  # 温度
    system_powers = state_trajectory[:, 1] * 142.3 * 1000  # 系统功率（转换为实际功率）
    soc_values = state_trajectory[:, 2]  # SOC
    comp_powers = control_sequence[:, 0]  # 压缩机功率
    battery_currents = control_sequence[:, 1]  # 电池电流

    # 计算温度统计
    temp_stats = {
        'mean': np.mean(temperatures),
        'max': np.max(temperatures),
        'min': np.min(temperatures),
        'std': np.std(temperatures),
        'time_above_26': np.sum(temperatures > 26) / len(temperatures) * 100,  # 超过26℃的时间百分比
        'time_below_24': np.sum(temperatures < 24) / len(temperatures) * 100   # 低于24℃的时间百分比
    }

    # 计算系统功率统计
    power_stats = {
        'mean': np.mean(system_powers),
        'max': np.max(system_powers),
        'min': np.min(system_powers),
        'std': np.std(system_powers),
        'power_fluctuation': np.std(system_powers) / np.mean(system_powers) * 100  # 功率波动率
    }

    # 计算SOC统计
    soc_stats = {
        'mean': np.mean(soc_values) * 100,  # 转换为百分比
        'max': np.max(soc_values) * 100,
        'min': np.min(soc_values) * 100,
        'std': np.std(soc_values) * 100,
        'time_below_20': np.sum(soc_values < 0.2) / len(soc_values) * 100,  # 低于20%的时间百分比
        'time_above_80': np.sum(soc_values > 0.8) / len(soc_values) * 100   # 高于80%的时间百分比
    }

    # 计算削峰填谷效果
    # 1. 计算原始可再生能源功率
    original_powers = P_RE[:len(time_points)]
    # 2. 计算功率平滑度改善
    original_fluctuation = np.std(original_powers) / np.mean(original_powers) * 100
    improved_fluctuation = power_stats['power_fluctuation']
    smoothing_improvement = (original_fluctuation - improved_fluctuation) / original_fluctuation * 100

    # 3. 计算峰谷差改善
    original_peak_valley = np.max(original_powers) - np.min(original_powers)
    improved_peak_valley = power_stats['max'] - power_stats['min']
    peak_valley_improvement = (original_peak_valley - improved_peak_valley) / original_peak_valley * 100

    # 输出统计信息到日志
    logging.info("\n=== 系统运行统计信息 ===")
    
    logging.info("\n温度统计:")
    logging.info(f"平均温度: {temp_stats['mean']:.2f}℃")
    logging.info(f"最高温度: {temp_stats['max']:.2f}℃")
    logging.info(f"最低温度: {temp_stats['min']:.2f}℃")
    logging.info(f"温度标准差: {temp_stats['std']:.2f}℃")
    logging.info(f"超过26℃的时间比例: {temp_stats['time_above_26']:.2f}%")
    logging.info(f"低于24℃的时间比例: {temp_stats['time_below_24']:.2f}%")

    logging.info("\n系统功率统计:")
    logging.info(f"平均功率: {power_stats['mean']/1000:.2f}kW")
    logging.info(f"最大功率: {power_stats['max']/1000:.2f}kW")
    logging.info(f"最小功率: {power_stats['min']/1000:.2f}kW")
    logging.info(f"功率标准差: {power_stats['std']/1000:.2f}kW")
    logging.info(f"功率波动率: {power_stats['power_fluctuation']:.2f}%")

    logging.info("\nSOC统计:")
    logging.info(f"平均SOC: {soc_stats['mean']:.2f}%")
    logging.info(f"最大SOC: {soc_stats['max']:.2f}%")
    logging.info(f"最小SOC: {soc_stats['min']:.2f}%")
    logging.info(f"SOC标准差: {soc_stats['std']:.2f}%")
    logging.info(f"低于20%的时间比例: {soc_stats['time_below_20']:.2f}%")
    logging.info(f"高于80%的时间比例: {soc_stats['time_above_80']:.2f}%")

    logging.info("\n削峰填谷效果评估:")
    logging.info(f"原始功率波动率: {original_fluctuation:.2f}%")
    logging.info(f"改善后功率波动率: {improved_fluctuation:.2f}%")
    logging.info(f"功率平滑度改善: {smoothing_improvement:.2f}%")
    logging.info(f"峰谷差改善: {peak_valley_improvement:.2f}%")

    # 保存统计信息到文件
    stats_file = os.path.join(log_dir, "statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("=== 系统运行统计信息 ===\n")
        
        f.write("\n温度统计:\n")
        f.write(f"平均温度: {temp_stats['mean']:.2f}℃\n")
        f.write(f"最高温度: {temp_stats['max']:.2f}℃\n")
        f.write(f"最低温度: {temp_stats['min']:.2f}℃\n")
        f.write(f"温度标准差: {temp_stats['std']:.2f}℃\n")
        f.write(f"超过26℃的时间比例: {temp_stats['time_above_26']:.2f}%\n")
        f.write(f"低于24℃的时间比例: {temp_stats['time_below_24']:.2f}%\n")

        f.write("\n系统功率统计:\n")
        f.write(f"平均功率: {power_stats['mean']/1000:.2f}kW\n")
        f.write(f"最大功率: {power_stats['max']/1000:.2f}kW\n")
        f.write(f"最小功率: {power_stats['min']/1000:.2f}kW\n")
        f.write(f"功率标准差: {power_stats['std']/1000:.2f}kW\n")
        f.write(f"功率波动率: {power_stats['power_fluctuation']:.2f}%\n")

        f.write("\nSOC统计:\n")
        f.write(f"平均SOC: {soc_stats['mean']:.2f}%\n")
        f.write(f"最大SOC: {soc_stats['max']:.2f}%\n")
        f.write(f"最小SOC: {soc_stats['min']:.2f}%\n")
        f.write(f"SOC标准差: {soc_stats['std']:.2f}%\n")
        f.write(f"低于20%的时间比例: {soc_stats['time_below_20']:.2f}%\n")
        f.write(f"高于80%的时间比例: {soc_stats['time_above_80']:.2f}%\n")

        f.write("\n削峰填谷效果评估:\n")
        f.write(f"原始功率波动率: {original_fluctuation:.2f}%\n")
        f.write(f"改善后功率波动率: {improved_fluctuation:.2f}%\n")
        f.write(f"功率平滑度改善: {smoothing_improvement:.2f}%\n")
        f.write(f"峰谷差改善: {peak_valley_improvement:.2f}%\n")

    logging.info(f"\n统计信息已保存至: {stats_file}")

    # 绘制功率对比图
    plot_power_comparison(time_points, original_powers, system_powers, log_dir)

    return {
        'temperature': temp_stats,
        'power': power_stats,
        'soc': soc_stats,
        'peak_valley': {
            'original_fluctuation': original_fluctuation,
            'improved_fluctuation': improved_fluctuation,
            'smoothing_improvement': smoothing_improvement,
            'peak_valley_improvement': peak_valley_improvement
        }
    } 