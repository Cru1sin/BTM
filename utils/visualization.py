import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import logging

def plot_results(time_points, control_sequence, state_trajectory, log_dir):
    """
    绘制MPC控制结果的主要图表，只显示压缩机功率和温度变化
    """
    # 创建两个子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # 1. 压缩机功率
    ax1.plot(time_points, control_sequence[:, 0], 'b-', label='压缩机功率')
    ax1.set_xlabel('时间 (s)')
    ax1.set_ylabel('功率 (W)')
    ax1.set_title('压缩机功率变化')
    ax1.grid(True)
    ax1.legend()
    
    # 2. 电池温度
    ax2.plot(time_points, state_trajectory[:, 0], 'r-', label='电池温度')
    ax2.set_xlabel('时间 (s)')
    ax2.set_ylabel('温度 (℃)')
    ax2.set_title('电池温度变化')
    ax2.grid(True)
    ax2.legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图片
    plot_file = os.path.join(log_dir, f"mpc_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    logging.info(f"结果图表已保存至: {plot_file}")
    
    plt.show()
    plt.close()

# 删除不需要的函数
# def plot_power_comparison(time_points, original_powers, system_powers, log_dir):
#     ... 