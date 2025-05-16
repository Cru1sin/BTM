import torch
from Model.Simple import Solution_u

model_path = '/Users/cruisin/Documents/BTM/SOH/best-model/model.pth'
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

# 初始化模型
model = Solution_u().to(device)

# 加载 state_dict（从 solution_u 字段中提取）
ckpt = torch.load(model_path, map_location=device)
model.load_state_dict(ckpt['solution_u'])

# 设置为评估模式
model.eval()

print("✅ 模型加载成功，已进入推理模式！")

# 测试数据
test_data = torch.randn(1, 6).to(device)
test_data = test_data.unsqueeze(0)
# 进行推理
with torch.no_grad():
    output = model(test_data)
    print(output)