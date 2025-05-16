import numpy as np
from Battery.BatteryPack import BatteryPack as Battery
from CoolingSystem.CS_for_ES import SimpleCoolingSystem as CoolingSystem

class RBController:
    """
    基于规则的电池温度控制器
    
    控制逻辑：
    - 当温度 > 26℃ 时，开启压缩机（功率设为2000W）
    - 当温度 < 24℃ 时，关闭压缩机（功率设为0W）
    - 在24-26℃之间时，保持当前压缩机状态
    """
    
    def __init__(self, battery_model, cooling_system, P_RE, dt=1, delta_P=300):
        """
        初始化基于规则的控制器
        
        参数:
            battery_model: 电池热模型实例
            cooling_system: 冷却系统模型实例
            P_RE: 可再生能源发电功率序列
            dt: 时间步长(s)
        """
        self.bm = battery_model
        self.cs = cooling_system
        self.P_RE = P_RE
        self.dt = dt
        
        # 温度阈值
        self.T_high = 26.0  # 高温阈值
        self.T_low = 24.0   # 低温阈值
        
        # 压缩机功率设置
        self.P_comp_on = 3000.0  # 开启时的功率
        self.P_comp_off = 0.0    # 关闭时的功率
        self.delta_P = delta_P  # 压缩机功率变化量
        
        # 记录当前压缩机状态
        self.current_comp_power = 0.0

    def control(self, i, current_temp, SOC, comp_power):
        """
        根据当前温度决定压缩机功率
        
        参数:
            i: 当前时间步
            current_temp: 当前电池温度
            SOC: 当前SOC
            comp_power: 当前压缩机功率
            
        返回:
            dict: 包含控制序列和状态轨迹的字典
        """
        # 根据温度决定压缩机功率
        if current_temp > self.T_high:
            new_comp_power = min(self.P_comp_on, self.current_comp_power + self.delta_P)
        elif current_temp < self.T_low:
            new_comp_power = max(self.P_comp_off, self.current_comp_power - self.delta_P)
        else:
            new_comp_power = comp_power  # 保持当前状态
            
        # 更新当前压缩机功率
        self.current_comp_power = new_comp_power
        
        # 计算冷却量和冷却功率
        Q_cool = float(self.cs.battery_cooling(current_temp, new_comp_power))
        P_cool = new_comp_power + 200  # 压缩机功率加上基础功率
        
        # 计算功率缺口和所需电流
        P_gap = self.P_RE[i] + P_cool
        I_pack = self.bm.Current_Pack2Cell(P_gap)
        I_pack = min(I_pack, self.bm.I_max_limit)
        I_pack = max(I_pack, self.bm.I_min_limit)
        
        # 计算下一步状态
        temp_next, _, SOC_next, _, _ = self.bm.battery_model(
            Q_cool=Q_cool,
            I_pack=I_pack,
            T_bat=current_temp,
            SOC=SOC
        )
        
        return new_comp_power, I_pack, temp_next, SOC_next

# === 使用示例 ===
if __name__ == "__main__":
    # 初始化模型
    dt = 1
    bm = Battery(dt)
    cs = CoolingSystem(dt, T_amb=26)
    
    # 创建控制器
    rb_controller = RBController(bm, cs, P_RE=np.zeros(1000))
    
    # 测试不同温度下的控制效果
    test_temps = [23.5, 24.5, 26.5]
    for temp in test_temps:
        solution = rb_controller.control(0, temp, 0.5, 0)
        print(f"\n当前温度: {temp}℃")
        print(f"压缩机功率: {solution['control_sequence'][0,0]}W")
        print(f"下一时刻温度: {solution['state_trajectory'][1,0]}℃")