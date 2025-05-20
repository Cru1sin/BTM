from dataloader.XJTU_loader import XJTUDdataset
from Model.Model import PINN
import argparse
import os
import time
from datetime import datetime
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_data(args):
    """
    加载数据
    :param args: 参数
    :return: 包含训练、验证和测试数据的字典
    """
    data_loader = XJTUDdataset(args=args)
    
    # 读取所有batch的数据
    #data = data_loader.read_one_batch('Batch-4')
    #data = data_loader.read_all()
    #data = data_loader.get_charge_data(test_battery_id=1)
    data = data_loader.load_data_from_pkl()
    return data

def main():
    args = get_args()
    # 获取当前系统时间，格式化为字符串
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 创建保存结果的目录
    #save_folder = args.save_folder
    # args.save_folder改为当前时间
    args.save_folder = os.path.join(args.save_folder, current_time)
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
        
    # 创建一个实验配置文件
    with open(os.path.join(args.save_folder, 'experiment_info.txt'), 'w') as f:
        f.write(f'Experiment started at: {current_time}\n')
    
    # 设置日志文件
    log_dir = os.path.join(args.save_folder, 'logging.txt')
    with open(log_dir,'w') as f:
        f.write(f'Logging started at: {current_time}\n')
    setattr(args, "log_dir", log_dir)

    # 加载所有数据
    dataloader = load_data(args)
    # 初始化模型并训练
    if args.wandb:
        wandb.login()
        wandb.init(project=args.wandb_project_name, name=args.wandb_name)
    
    pinn = PINN(args)
    wandb.watch(pinn, log='all')
    pinn.Train(
        trainloader=dataloader['train'],
        validloader=dataloader['valid'],
        testloader=dataloader['test']
    )
    

def get_args():
    parser = argparse.ArgumentParser('XJTU数据集的超参数')
    
    # 数据相关参数
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--minmax_range', type=tuple, default=(0, 1))
    parser.add_argument('--random_seed', type=int, default=2025)
    parser.add_argument('--sample_size', type=int, default=100)
    #parser.add_argument('--data_root', type=str, default='/home/user/nss/BTM/CNN-PINN4QSOH/data')
    # data
    parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU', 'MIT', 'CALCE'])
    parser.add_argument('--input_type', type=str, default='charge',
                        choices=['charge', 'partial_charge', 'handcraft_features'])
    parser.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
    parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5])

    # 训练相关参数
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.001, help='warmup learning rate')
    parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=0.00001, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=0.001, help='learning rate for F')

    # 模型相关参数
    parser.add_argument('--F_layers_num', type=int, default=3, help='number of layers in F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='hidden dimension of F')

    # 损失函数相关参数
    parser.add_argument('--alpha', type=float, default=0.7, 
                       help='weight for PDE loss (loss = l_data + alpha * l_PDE + beta * l_physics)')
    parser.add_argument('--beta', type=float, default=0.2, 
                       help='weight for physics loss (loss = l_data + alpha * l_PDE + beta * l_physics)')

    # 保存相关参数
    parser.add_argument('--save_folder', type=str, default='results/Simple-model', 
                       help='folder to save results')
    parser.add_argument('--log_dir', type=str, default='logging.txt', 
                       help='log file path')
    
    parser.add_argument('--wandb', type=bool, default=True, 
                       help='use wandb to log')
    parser.add_argument('--wandb_project_name', type=str, default='SOH-prediction', 
                       help='wandb project name')
    parser.add_argument('--wandb_name', type=str, default='Simple-model-3', 
                       help='wandb name')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

