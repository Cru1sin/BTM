from pyomo.environ import *
from battery_model import Battery_Model as Battery
from vehicle_dynamics_model import Vehicle
from Simple_Cooling_Model import simple_cooling_model as Cooling 
from pyomo.environ import inequality

# 创建模型
model = ConcreteModel()

# 电池温度和冷却系统功率的动态模型
T_amb = 25
T_set = 35  # 目标温度
T_min, T_max = 20, 40  # 温度范围
alpha_1, alpha_2, alpha_3 = 1.0, 0.1, 0.01   # 目标函数权重
T_battery_current = T_amb  # 当前电池温度（初值，可动态更新）
P_min, P_max = 0, 10000
h_rfg_min, h_rfg_max = 0, 20000

# 控制变量
v_min, v_max = 1000, 8000  # 转速范围
model.v_comp = Var(bounds=(v_min, v_max))
model.v_pump = Var(bounds=(v_min, v_max))
model.v_fan = Var(bounds=(v_min, v_max))  # 如果风扇是可调的
model.T_battery_next = Var(bounds=(T_min, T_max))
model.P_total = Var(bounds=(P_min, P_max))
model.h_rfg = Var(bounds=(h_rfg_min, h_rfg_max))

model.v_comp.set_value(2000)  # 设置压缩机速度初值
model.v_pump.set_value(2000)  # 设置泵速度初值
model.v_fan.set_value(2000)

vehicle = Vehicle(N=1, dt=0.1)
cooling = Cooling(T_amb=T_amb,dt=0.1,N=1)
battery = Battery(T_amb=T_amb,dt=0.1,N=1)

# 目标函数
def objective_rule(model):
    return alpha_1 * (model.T_battery_next - T_set)**2 + alpha_2 * model.P_total **2 + alpha_3 * model.h_rfg ** 2

model.obj = Objective(rule=objective_rule, sense=minimize)

def power_constraint(model):
    cooling.massflow_rfg(model.v_comp)
    cooling.massflow_clnt(model.v_pump)
    return model.P_total == cooling.compressor_power() + cooling.pump_power() + vehicle.traction()
model.P_constraint = Constraint(rule=power_constraint)

# 定义电池温度更新约束
def battery_temp_update_constraint(model):
    cooling.massflow_rfg(model.v_comp)
    cooling.massflow_clnt(model.v_pump)
    Q_cooling = cooling.Q_cooling(battery.T_bat)
    P_total = cooling.compressor_power() + cooling.pump_power() + vehicle.traction()
    return model.T_battery_next == battery.battery_thermal_model(Q_cool=Q_cooling, Power=P_total)
model.temp_update_constraint = Constraint(rule=battery_temp_update_constraint)

def h_rfg_update_constraint(model):
    cooling.massflow_rfg(model.v_comp)
    cooling.massflow_clnt(model.v_pump)
    return model.h_rfg == cooling.dynamic_h_rfg()
model.h_rfg_constraint = Constraint(rule=h_rfg_update_constraint)



model.temp_constraint = Constraint(expr=inequality(T_min, model.T_battery_next, T_max))

# 求解
solver = SolverFactory('ipopt', executable='/Users/cruisin/anaconda3/envs/python/bin/ipopt')
solver.options['max_iter'] = 10000  # 增加最大迭代次数
#solver.options['output_file'] = 'solver_log.txt'  # 保存详细日志
#solver.options['infeasibility_handling'] = 'yes'  # 启用不可行性分析

results = solver.solve(model)

# 输出结果
print(f"Optimal Compressor Speed: {model.v_comp.value}")
print(f"Optimal Pump Speed: {model.v_pump.value}")
print(f"Optimal Fan Speed: {model.v_fan.value}")