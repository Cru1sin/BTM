from scipy.io import loadmat
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
from utils.Scaler import Scaler
import pickle

class XJTUDdataset():
    def __init__(self,args):
        super(XJTUDdataset).__init__()
        self.root = '/home/user/nss/BTM/CNN-PINN4QSOH/data/XJTU'
        self.max_capacity = 2.0
        self.normalized_type = args.normalized_type
        self.minmax_range = args.minmax_range
        self.seed = args.random_seed
        self.batch = args.batch
        self.batch_size = args.batch_size

        self.save_path = '/home/user/nss/BTM/CNN-PINN4QSOH/data/XJTU/charge.pkl'

    def _parser_mat_data(self,battery_i_mat):
        '''
        :param battery_i_mat: shape:(1,len)
        :return: np.array
        '''
        data = []
        label = []
        cycle_i = []
        for i in range(1, battery_i_mat.shape[1]):
            cycle_i_data = battery_i_mat[0,i]
            charge_time = cycle_i_data['relative_time_min'][0] # (128,) 单位：min
            relative_time = np.array(([charge_time[0]] + [(charge_time[i] - charge_time[i-1])*60 for i in range(1, len(charge_time))])) # 计算相邻时间点的差值，并转化单位为秒
            current = cycle_i_data['current_A'][0] # (1,127)
            voltage = cycle_i_data['voltage_V'][0] # (1,127)
            temperature = cycle_i_data['temperature_C'][0] # (1,127)
            for j in range(len(charge_time)):
                data_j = np.array([relative_time[j], charge_time[j]*60,i, current[j],voltage[j],temperature[j]])
                # 相对充电时间、累计充电时间、循环次数、电流、电压、温度 dim = 6
                # 归一化
                scaler = Scaler(data_j)
                if self.normalized_type == 'standard':
                    data_norm = scaler.standerd()
                else:
                    data_norm = scaler.minmax(feature_range=self.minmax_range)
                cycle_i.append(data_norm)

            capacity_loss = cycle_i_data['capacity'][0] - battery_i_mat[0,i-1]['capacity'][0]
            label.append(capacity_loss)

            data.append(cycle_i)
            cycle_i = []
        data = np.array(data,dtype=np.float32)
        label = np.array(label,dtype=np.float32)
        print(data.shape,label.shape) # （cycle_num, 128, 6）,（cycle_num,）
        soh_loss = label / self.max_capacity

        data_1 = data[0:-1]
        data_2 = data[1:]
        soh_loss_1 = soh_loss[0:-1]
        soh_loss_2 = soh_loss[1:]

        return (data_1,soh_loss_1),(data_2, soh_loss_2)

    def _encapsulation(self,train_x1,train_y1,train_x2, train_y2, test_x1, test_x2, test_y1, test_y2):
        '''
        Encapsulate the numpy.array into DataLoader
        :param train_x: numpy.array
        :param train_y: numpy.array
        :param test_x: numpy.array
        :param test_y: numpy.array
        :return:
        '''
        train_x1 = torch.from_numpy(train_x1)
        train_x2 = torch.from_numpy(train_x2)
        train_y1 = torch.from_numpy(train_y1)
        train_y2 = torch.from_numpy(train_y2)
        test_x1 = torch.from_numpy(test_x1)
        test_x2 = torch.from_numpy(test_x2)
        test_y1 = torch.from_numpy(test_y1)
        test_y2 = torch.from_numpy(test_y2)

        train_x1, valid_x1, train_y1, valid_y1 = train_test_split(train_x1, train_y1, test_size=0.2, random_state=self.seed)
        train_x2, valid_x2, train_y2, valid_y2 = train_test_split(train_x2, train_y2, test_size=0.2, random_state=self.seed)

        train_loader = DataLoader(TensorDataset(train_x1, train_x2, train_y1, train_y2),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_x1, valid_x2, valid_y1, valid_y2),
                                  batch_size=self.batch_size,
                                  shuffle=True)
        test_loader = DataLoader(TensorDataset(test_x1, test_x2, test_y1, test_y2),
                                 batch_size=self.batch_size,
                                 shuffle=False)
        return train_loader, valid_loader, test_loader

    def _get_raw_data(self,path,test_battery_id):
        mat = loadmat(path)
        battery = mat['battery']
        battery_ids = list(range(1, battery.shape[1] + 1))
        if test_battery_id not in battery_ids:
            raise IndexError(f'"test_battery" must be in the {battery_ids}, but got {test_battery_id}. ')

        test_battery = battery[0, test_battery_id - 1][0]
        print(f'test battery id: {test_battery_id}, test data shape: ', end='')
        (test_x1, test_y1), (test_x2, test_y2) = self._parser_mat_data(test_battery)
        train_x1, train_y1 = [], []
        train_x2, train_y2 = [], []
        for id in battery_ids:
            if id == test_battery_id:
                continue
            print(f'train battery id: {id}, ', end='')
            train_battery = battery[0, id - 1][0]
            (x1, y1), (x2,y2) = self._parser_mat_data(train_battery)
            train_x1.append(x1)
            train_x2.append(x2)
            train_y1.append(y1)
            train_y2.append(y2)
        train_x1 = np.concatenate(train_x1, axis=0)
        train_x2 = np.concatenate(train_x2, axis=0)
        train_y1 = np.concatenate(train_y1, axis=0)
        train_y2 = np.concatenate(train_y2, axis=0)
        print('train data shape: ', train_x1.shape, train_y1.shape, train_x2.shape, train_y2.shape)
        print(f'test data shape: {test_x1.shape}, {test_y1.shape}, {test_x2.shape}, {test_y2.shape}')
        return train_x1, train_y1, train_x2, train_y2, test_x1, test_x2, test_y1, test_y2

    def get_charge_data(self,test_battery_id=1):
        print('----------- load charge data -------------')
        file_name = f'batch-{self.batch}.mat'
        self.charge_path = os.path.join(self.root, 'charge', file_name)
        train_x1, train_y1, train_x2, train_y2, test_x1, test_x2, test_y1, test_y2 = self._get_raw_data(path=self.charge_path,test_battery_id=test_battery_id)
        train_loader, valid_loader, test_loader = self._encapsulation(train_x1, train_y1, train_x2, train_y2, test_x1, test_x2, test_y1, test_y2)
        data_dict = {'train':train_loader,
                     'test':test_loader,
                     'valid':valid_loader}
        print('-------------  finished !  ---------------')
        return data_dict
    
    def get_all_data(self):
        train_x1_all, train_y1_all, train_x2_all, train_y2_all, test_x1_all, test_x2_all, test_y1_all, test_y2_all = [], [], [], [], [], [], [], []
        for batch_id in range(1, 7):
            print(f'----------- load batch-{batch_id} data -------------')
            file_name = f'batch-{batch_id}.mat'
            self.charge_path = os.path.join(self.root, 'charge', file_name)
            train_x1, train_y1, train_x2, train_y2, test_x1, test_x2, test_y1, test_y2 = self._get_raw_data(path=self.charge_path,test_battery_id=1)
            train_x1_all.append(train_x1)
            train_y1_all.append(train_y1)
            train_x2_all.append(train_x2)
            train_y2_all.append(train_y2)
            test_x1_all.append(test_x1)
            test_x2_all.append(test_x2)
            test_y1_all.append(test_y1)
            test_y2_all.append(test_y2)
        train_x1_all = np.concatenate(train_x1_all, axis=0)
        train_y1_all = np.concatenate(train_y1_all, axis=0)
        train_x2_all = np.concatenate(train_x2_all, axis=0)
        train_y2_all = np.concatenate(train_y2_all, axis=0)
        test_x1_all = np.concatenate(test_x1_all, axis=0)
        test_x2_all = np.concatenate(test_x2_all, axis=0)
        test_y1_all = np.concatenate(test_y1_all, axis=0)
        test_y2_all = np.concatenate(test_y2_all, axis=0)
        train_loader, valid_loader, test_loader = self._encapsulation(train_x1_all, train_y1_all, train_x2_all, train_y2_all, test_x1_all, test_x2_all, test_y1_all, test_y2_all)
        data_dict = {'train':train_loader,
                     'test':test_loader,
                     'valid':valid_loader}
        self.save_data(data_dict)
        print('-------------  finished !  ---------------')
        return data_dict

    def save_data(self,data_dict):
        with open(self.save_path, 'wb') as f:
            pickle.dump(data_dict, f)
        print(f'save data to {self.save_path}')

    def load_data_from_pkl(self):
        with open(self.save_path, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

if __name__ == '__main__':
    import argparse
    def get_args():

        parser = argparse.ArgumentParser(description='dataloader test')
        parser.add_argument('--random_seed',type=int,default=2023)
        # data
        parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU', 'MIT', 'CALCE'])
        parser.add_argument('--input_type', type=str, default='charge',
                            choices=['charge', 'partial_charge', 'handcraft_features'])
        parser.add_argument('--normalized_type', type=str, default='minmax', choices=['minmax', 'standard'])
        parser.add_argument('--minmax_range', type=tuple, default=(0, 1), choices=[(0, 1), (1, 1)])
        parser.add_argument('--batch_size', type=int, default=128)
        # the parameters for XJTU data
        parser.add_argument('--batch', type=int, default=1, choices=[1, 2, 3, 4, 5, 6])

        args = parser.parse_args()
        return args

    args = get_args()
    data = XJTUDdataset(args)
    charge_data = data.get_charge_data()
