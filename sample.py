from scipy.interpolate import RegularGridInterpolator
import numpy as np
from parameter import m_clnt_vector, T_air_vector, lamda1_table, lamda2_table, lamda3_table, lamda4_table, lamda5_table, lamda6_table
from math import exp
from CoolingSystem import CoolingSystem
class SimpleCoolingSystem(CoolingSystem):
    """
    一个简单的液体冷却系统，假设传热方式为对流换热
    """
    def __init__(self, dt, T_amb):
        """
        初始化冷却系统
        :param dt: 采样时间
        :param T_amb: 环境温度
        """
        super().__init__(dt, T_amb)

        # 保存插值函数的字典
        self.interp_funcs = {}

        # 构建插值函数
        self.interp_funcs["lambda1"] = RegularGridInterpolator(
            (m_clnt_vector, T_air_vector), lamda1_table.T, method='linear'
        )
        self.interp_funcs["lambda2"] = RegularGridInterpolator(
            (m_clnt_vector, T_air_vector), lamda2_table.T, method='linear'
        )
        self.interp_funcs["lambda3"] = RegularGridInterpolator(
            (m_clnt_vector, T_air_vector), lamda3_table.T, method='linear'
        )
        self.interp_funcs["lambda4"] = RegularGridInterpolator(
            (m_clnt_vector, T_air_vector), lamda4_table.T, method='linear'
        )
        self.interp_funcs["lambda5"] = RegularGridInterpolator(
            (m_clnt_vector, T_air_vector), lamda5_table.T, method='linear'
        )
        self.interp_funcs["lambda6"] = RegularGridInterpolator(
            (m_clnt_vector, T_air_vector), lamda6_table.T, method='linear'
        )

        # 默认的冷却剂质量流量和空气温度
        self.massflow_clnt = 0.18 
        self.massflow_air = 1.2

        self._initial_lambda()

    def _initial_lambda(self):
        # 获取插值后的 lambda 值
        self.lambda1 = self.get_lambda("lambda1").item()
        self.lambda2 = self.get_lambda("lambda2").item()
        self.lambda3 = self.get_lambda("lambda3").item()
        self.lambda4 = self.get_lambda("lambda4").item()
        self.lambda5 = self.get_lambda("lambda5").item()
        self.lambda6 = self.get_lambda("lambda6").item()

    def get_lambda(self, lambda_name):
        """
        获取插值后的 lambda 值
        :param lambda_name: lambda 名称，如 "lambda1"
        :return: 插值后的 lambda 值
        """
        # 构造输入点
        point = np.array([self.massflow_clnt, self.T_amb])
        return self.interp_funcs[lambda_name](point)

    def battery_cooling(self, T_bat, P_comp, v_veh):
        """
        计算冷却量 Q_cool (W)
        Q_cool = lambda1 * P_comp + lambda2 * P_comp**2 + lambda3 * T_clnt_out +
                 lambda4 * T_amb * m_air + lambda5 * T_clnt_out * m_clnt + lambda6
        """
        

        T_clnt_out = (self.T_clnt_in - T_bat) * exp(-(self.h_bat * self.A_bat) / (self.massflow_clnt * self.capacity_clnt)) + T_bat # 冷却剂出口温度 (℃)
        massflow_air = 0.07065 + 0.00606 * v_veh  # 空气质量流量 (kg/s)
        # 计算 Q_cooling
        Q_cooling = (
            self.lambda1 * P_comp +
            self.lambda2 * P_comp**2 +
            self.lambda3 * T_clnt_out +
            self.lambda4 * self.T_amb * massflow_air +
            self.lambda5 * T_clnt_out * self.massflow_clnt +
            self.lambda6
        )*self.dt

        self.T_clnt_in = T_clnt_out - Q_cooling / (self.massflow_clnt * self.capacity_clnt)  # 更新冷却剂入口温度
        return Q_cooling



# **测试模型**
if __name__ == "__main__":
    dt = 0.1  # 采样时间
    T_amb = 26  # 环境温度
    cooling_system = SimpleCoolingSystem(dt, T_amb)

    T_bat = 40  # 当前电池温度 (℃)
    P_comp = 800  # 压缩机功率 (W)
    v_veh = 10  # 车辆速度 (m/s)

    Q_cool = cooling_system.battery_cooling(T_bat, P_comp, v_veh)

    print(f"冷却量 Q_cool: {Q_cool:.2f} W")