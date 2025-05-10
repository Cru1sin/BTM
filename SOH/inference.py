import torch
import numpy as np
import os
from Model.Simple import PINN
import argparse
from dataloader.XJTU_dataloader import XJTUdata
from utils.util import eval_metrix
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/user/nss/BTM/CNN-PINN4QSOH/results of reviewer/XJTU/model_4.21/model.pth', help='path to trained model')
    parser.add_argument('--save_folder', type=str, default='./results', help='save folder')
    parser.add_argument('--log_dir', type=str, default='./logs', help='log dir')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max,z-score')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup learning rate')
    parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=0.0001, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=0.001, help='learning rate for F')
    parser.add_argument('--alpha', type=float, default=0.7, 
                       help='weight for PDE loss (loss = l_data + alpha * l_PDE + beta * l_physics)')
    parser.add_argument('--beta', type=float, default=0.2, 
                       help='weight for physics loss (loss = l_data + alpha * l_PDE + beta * l_physics)')
    parser.add_argument('--wandb', type=bool, default=True, 
                       help='use wandb to log')
    parser.add_argument('--wandb_project_name', type=str, default='CNN-PINN4QSOH', 
                       help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default='test_batch_1', 
                       help='wandb name')
    parser.add_argument('--minmax_range', type=tuple, default=(0, 1))
    parser.add_argument('--random_seed', type=int, default=2025)
    parser.add_argument('--sample_size', type=int, default=100)

    return parser.parse_args()

def normalize_data(data, method='z-score'):
    """数据归一化"""
    if method == 'z-score':
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std == 0] = 1  # 避免除以0
        normalized_data = (data - mean) / std
        return normalized_data, mean, std
    elif method == 'min-max':
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # 避免除以0
        normalized_data = (data - min_val) / range_val
        return normalized_data, min_val, range_val
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def mock_data(batch_size=128, seq_length=126):
    """生成模拟数据用于推理"""
    features = []
    
    for i in range(batch_size):
        cycle_i = []
        # 生成时间序列数据
        charge_time = np.linspace(0, 10000, seq_length)  # 模拟充电时间
        current = np.sin(2 * np.pi * charge_time/10000) + np.random.normal(0, 0.1, seq_length)  # 模拟电流
        voltage = np.cos(2 * np.pi * charge_time/10000) * 4 + np.random.normal(0, 0.1, seq_length)  # 模拟电压
        temperature = np.sin(4 * np.pi * charge_time/10000) + np.random.normal(0, 0.1, seq_length) + 25 # 模拟温度
        
        # 组合每个时间步的数据
        for j in range(seq_length):
            # 计算历史统计特征
            current_stats = calculate_stats(current[:j+1])
            voltage_stats = calculate_stats(voltage[:j+1])
            
            # 计算相对充电时间
            relative_time = charge_time[0] if j == 0 else charge_time[j] - charge_time[j-1]
            
            # 组合特征
            data_j = np.concatenate([
                np.array([
                    relative_time,              # 相对充电时间
                    charge_time[j],             # 累计充电时间
                    i,                          # 循环次数
                    current[j],                 # 电流
                    voltage[j],                 # 电压
                    float(temperature[j])       # 温度
                ]),
                current_stats,                  # 电流统计特征
                voltage_stats                   # 电压统计特征
            ])
            cycle_i.append(data_j)
        
        features.append(np.array(cycle_i))
    
    return np.array(features, dtype=np.float32)

def calculate_stats(data):
    """计算统计特征"""
    if len(data) == 0:
        return np.zeros(7)  # 返回7个零作为默认值
    return np.array([
        np.mean(data),
        np.std(data),
        np.max(data),
        np.min(data),
        np.ptp(data),  # peak-to-peak
        np.percentile(data, 25),
        np.percentile(data, 75)
    ])

def load_data(args):
    """
    加载数据
    :param args: 参数
    :return: 包含训练、验证和测试数据的字典
    """
    root = '/home/user/nss/BTM/CNN-PINN4QSOH/data/XJTU'  # mat文件所在目录
    data_loader = XJTUdata(root=root, args=args)
    
    # 读取所有batch的数据
    data = data_loader.read_one_batch(batch='batch-1')
    
    return {
        'train': data['train'],
        'valid': data['valid'],
        'test': data['test']
    }

def inference():
    args = get_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 确保保存目录存在
    os.makedirs(args.save_folder, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 加载模型
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path)
        model = PINN(args).to(device)  # 使用args初始化模型
        model.solution_u.load_state_dict(checkpoint['solution_u'])
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"Model not found at {args.model_path}")
        return
    
    # 加载测试数据
    test_data = load_data(args)['test']
    
    # 获取一条测试数据
    for batch_idx, (x1, _, y1, _) in enumerate(test_data):
        if batch_idx == 0:  # 只取第一条数据
            test_features = x1.to(device)
            test_label = y1.to(device)
            break
    
    # 从测试数据中取出一个20维特征向量
    # 选择第一个样本的第一个时间步的特征
    single_feature = test_features[0, 0, :]  # shape: (20,)
    
    # 为了符合模型的输入要求，需要添加batch和sequence维度
    # 将单个特征向量reshape为(1, 1, 20)
    single_feature = single_feature.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 20)
    
    # 进行推理
    model.eval()
    with torch.no_grad():
        predictions = model.predict(single_feature)
    
    # 将结果转换为numpy数组
    predictions = predictions.cpu().numpy()
    test_label = test_label.cpu().numpy()
    
    # 打印预测结果
    print(f"Original test data shape: {test_features.shape}")
    print(f"Single feature shape: {single_feature.shape}")
    print(f"Prediction shape: {predictions.shape}")
    print(f"Single feature values: {single_feature.squeeze().cpu().numpy()}")
    print(f"Predicted value: {predictions}")
    
    # 计算评估指标
    #[MAE, MAPE, MSE, RMSE] = eval_metrix(predictions, test_label[0:1])  # 只取对应的标签
    #info = '[Test] MSE: {:.8f}, MAE: {:.6f}, MAPE: {:.6f}, RMSE: {:.6f}'.format(MSE, MAE, MAPE, RMSE)
    #print(info)

if __name__ == "__main__":
    inference()
