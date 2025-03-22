import math
import numpy as np

class Vehicle:
    delta = 1.13  # Correction coefficient for rotating mass
    m_veh = 2200
    f = 0.019  # Rolling resistance coefficient
    C_d = 0.3  # Aerodynamic drag coefficient
    A_wind = 3.2  # Vehicle frontal area (m^2)
    eta_p = 0.9  # Powertrain efficiency
    g = 9.8
    rho_air = 1.225 # 空气密度 (Kg/m^3)

    def __init__(self, dt):
        self.velocity = 0
        self.dt = dt # 采样时间间隔
        self.i = 0
        self.P_trac = 0

    def predict(self):
        """
        预测未来N个时间步长的速度和加速度
        ----------------------------------------
        假设
        1. 速度和加速度的变化服从匀加速运动，
            * 若速度 < 10 m/s
                加速度 = 3m/s^2
            * 若速度达到10m/s
                速度采取正弦波动
        2. 速度的变化是预测是完全准确的
        3. 加速度的变化是通过速度的变化计算的
        ----------------------------------------
        :param N: 预测的时间步长数
        :param dt: 时间步长 (s)
        :return: v_pred: 预测的未来N个时间步长的速度 (m/s)，速度为时刻内的速度，不是采样时间内的平均速度
                    a_pred: 预测的未来N个时间步长的加速度 (m/s^2)
        """
        v = self.velocity
        if v <= 10:
            a = 2
        elif v >= 12:
            a = -2
        else:
            a = 2 * math.cos(0.1 * self.i)
            self.i += 1
        v = v + a * self.dt
        self.velocity = v
        return v, a
    
    def random_generation(self, v_min=0, v_max=30, a_min=-3, a_max=3):
        current_velocity = self.velocity
        
        # 计算速度在区间中的位置比例
        v_range = v_max - v_min
        position_ratio = (current_velocity - v_min) / v_range if v_range != 0 else 0.5
        
        # 动态调整加速度方向概率（速度越高，正加速度概率越低）
        prob_positive = 1 - position_ratio
        
        # 根据概率生成加速度方向
        if np.random.rand() < prob_positive:
            new_acceleration = np.random.uniform(0, a_max)  # 正向加速
        else:
            new_acceleration = np.random.uniform(a_min, 0)  # 反向减速
        
        # 计算新速度
        new_velocity = current_velocity + new_acceleration * self.dt
        
        # 边界保护机制
        if new_velocity > v_max:
            # 当接近上限时，生成反向加速度避免突变
            overshoot = new_velocity - v_max
            new_velocity = v_max
            new_acceleration = (new_velocity - current_velocity) / self.dt
        elif new_velocity < v_min:
            # 当接近下限时，生成正向加速度避免突变
            undershoot = v_min - new_velocity
            new_velocity = v_min
            new_acceleration = (new_velocity - current_velocity) / self.dt

        return new_velocity, new_acceleration
    
    def traction(self):
        """
        计算牵引功率
        ----------------------------------------
        假设
        1. P_trac仅为速度和加速度的函数
        2. 速度减小，刹车时，P_trac为0
        3. 速度增加，加速时，P_trac为正
        """
        # v_pred, a_pred = self.predict()
        v_pred, a_pred = self.random_generation()

        F_r = Vehicle.f * Vehicle.m_veh * Vehicle.g
        constant_air_res = 0.5 * Vehicle.C_d * Vehicle.A_wind * Vehicle.rho_air
        F_a = constant_air_res * v_pred **2

        P_trac = (v_pred * (Vehicle.delta * Vehicle.m_veh * a_pred + F_r + F_a)) / Vehicle.eta_p if a_pred>= 0 else 0 

        self.velocity = v_pred
        self.P_trac = P_trac
        return P_trac, v_pred



def test(model):
    v_pred, a_pred = model.predict()
    P_trac = model.traction()
    print(f"速度为: {v_pred}, 加速度为: {a_pred}, 牵引功率为: {P_trac}")
    #----------------------------------------------
    print(f"更新后车速为 {model.velocity}")

if __name__ == '__main__':
    vehicle = Vehicle(dt = 0.1)
    vehicle.velocity = 0

    test(vehicle)
    test(vehicle)
    test(vehicle)
    test(vehicle)
    test(vehicle)
    # 由于sin(i)的关系每个时域内牵引功率是不同的
