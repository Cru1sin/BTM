import numpy as np

class PID:
    def __init__(self, Vehicle_model, Battery_model, Cooling_model, T_opt, dt, Kp, Ki, Kd):
        # 车辆、电池、冷却系统模型
        self.vehicle = Vehicle_model
        self.battery = Battery_model
        self.cooling = Cooling_model

        # PID参数（手动调整）
        self.Kp = Kp  # 比例增益
        self.Ki = Ki  # 积分增益
        self.Kd = Kd  # 微分增益

        # 目标温度
        self.T_opt = T_opt  

        # 误差项初始化
        self.integral_error = 0  
        self.prev_error = 0  

        # 时间步长
        self.dt = dt  

        # 风扇功率计算参数
        self.k_f = 1e-3  # 经验公式系数（需要根据实验调整）

    def control(self, T_bat):
        """PID 控制逻辑"""
        error = T_bat - self.T_opt  
        self.integral_error += error * self.dt  
        derivative_error = (error - self.prev_error) / self.dt  
        self.prev_error = error  

        # 计算控制量（转速）
        control_signal = (self.Kp * error + 
                          self.Ki * self.integral_error + 
                          self.Kd * derivative_error)

        # 转速限制（假设 0 ≤ N ≤ 5000 RPM） 
        N_pump = np.clip(control_signal, 0, 5000)  

        return N_pump

    def update(self):
        """更新控制过程"""
        # 计算牵引功率
        P_trac = self.vehicle.traction()

        # 读取当前电池温度
        T_bat = self.battery.T_bat 

        # 计算PID控制转速
        N_pump = self.control(T_bat)

        # 更新质量流速
        self.cooling.massflow_clnt(N_pump)

        # 计算功率
        P_pump = self.cooling.pump_power()

        # 计算冷却量
        if N_pump:
            Q_cool = self.cooling.Q_cooling(T_bat)
        else:
            Q_cool = 0

        # 计算电池温度
        P_total = P_trac + P_pump
        self.battery.battery_thermal_model(Q_cool, P_total)
        print(f"牵引功率: {P_trac}, 泵转速: {N_pump}, 泵功率: {P_pump}, 冷却量: {Q_cool}")

        return T_bat, P_total

# 示例：初始化
# vehicle = VehicleModel()
# battery = BatteryModel()
# cooling = CoolingModel()
# pid_controller = PID(vehicle, battery, cooling, T_opt=35, dt=1)
# for t in range(100):
#     T_bat, P_total = pid_controller.update()