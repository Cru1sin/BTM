import torch
import os
import numpy as np
from SOH.Model.Simple import Solution_u
#from utils.math_utils import min, max

class SOH_predictor:
    def __init__(self, dt, charge_time, cycle_num):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda')
        self.model_path = 'SOH/best-model/model.pth'
        # 初始化模型
        self.model = Solution_u().to(self.device)
        # 加载 state_dict（从 solution_u 字段中提取）
        ckpt = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['solution_u'])

        self.relative_time = dt
        self.charge_time = charge_time
        self.cycle_num = cycle_num
        self.current_direction = None
        

    def normalize_data(self, data):
        """数据归一化"""
        min_val = np.min(data)
        max_val = np.max(data)
        normalized_data = (data - min_val) / ((max_val - min_val) + 1e-6)
        return normalized_data
    
    def inference(self, I_cell, U_cell, T_bat):
        data = np.array([self.relative_time, self.charge_time, self.cycle_num, float(I_cell), float(U_cell), float(T_bat)])
        print(f"data: {data}")
        normalized_data = self.normalize_data(data)
        tensor_data = torch.tensor(normalized_data, dtype=torch.float32).to(self.device)
        # 为了符合模型的输入要求，需要添加batch和sequence维度
        # 将单个特征向量reshape为(1, 1, 6)
        single_feature = tensor_data.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 6)
        
        # 进行推理
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(single_feature)
        
        # 将结果转换为numpy数组
        predictions = predictions.cpu().numpy()
        self.charge_time += self.relative_time
        return predictions

if __name__ == "__main__":
    soh_predictor = SOH_predictor(dt=20, relative_time=0, charge_time=0, cycle_num=300)
    data1 = [3,3.7,25]
    data2 = [3,3.7,35]
    predictions1 = soh_predictor.inference(data1)
    predictions2 = soh_predictor.inference(data2)
    print(predictions1)
    print(predictions2)
