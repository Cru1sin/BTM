import numpy as np
from scipy.io import loadmat
import os
from sklearn import metrics

def sum_of_cycle_soh(cycle_num, c_rate, temp, Ah):
    soh_loss_total = []
    for i in range(cycle_num.shape[1]):
        #print(cycle_num[:,i].shape, c_rate[:,i].shape, temp[:,i].shape, Ah[:,i].shape)
        if i == 100:
            j = 100
            print(cycle_num[j,i], c_rate[j,i], temp[j,i], Ah[j,i])
        soh_loss = aging_model(cycle_num[:,i], c_rate[:,i], temp[:,i], Ah[:,i])
        soh_loss_total.append(soh_loss)
    print(soh_loss_total.shape)
    soh_loss_diff = np.diff(soh_loss_total)
    soh_loss_total = np.sum(soh_loss_diff)
    return soh_loss_total

def aging_model(cycle_num, c_rate, temp, Ah):
    capacity = 2

    B = -10.8061524 * c_rate**3 + 274.4685 * c_rate**2 - 2127.7295 * c_rate + 8141.8878
    Ah_tp = cycle_num * 1 *capacity + Ah
    temp += 273.15
    #print(cycle_num.shape, c_rate.shape, temp.shape, Ah.shape)
    Q_loss = B * np.exp((-34700+370.3*c_rate)/(8.314*temp))*Ah_tp**0.72
    return Q_loss

def parser_mat_data(battery_i_mat):
    '''
    :param battery_i_mat: shape:(1,len)
    :return: np.array
    '''
    data = []
    label = []
    cycle_i = []
    for i in range(1, battery_i_mat.shape[1]):
        cycle_i_data = battery_i_mat[0,i]
        charge_time = cycle_i_data['relative_time_min'][0]*60 # (128,) 单位：s
        relative_time = np.array(([charge_time[0]] + [(charge_time[i] - charge_time[i-1]) for i in range(1, len(charge_time))])) # 计算相邻时间点的差值，并转化单位为秒
        current = cycle_i_data['current_A'][0]/2 # 电流倍率 (1,127)
        voltage = cycle_i_data['voltage_V'][0] # (1,127)
        # 计算Ah吞吐量
        Ah = np.sum(current * relative_time) / 3600
        temperature = cycle_i_data['temperature_C'][0] # (1,127)
        for j in range(len(charge_time)):
            data_j = np.array([relative_time[j], charge_time[j]*60,i, Ah, current[j],voltage[j],temperature[j]])
            # 相对充电时间、累计充电时间、循环次数、Ah吞吐量、电流倍率、电压、温度 dim = 7
            cycle_i.append(data_j)

        soh_loss = (battery_i_mat[0,0]['capacity'][0] - battery_i_mat[0,i-1]['capacity'][0]) / 2
        label.append(soh_loss)
        data.append(cycle_i)
        cycle_i = []
    #print(data,label) # （cycle_num, 128, 6）,（cycle_num,）
    soh = label

    return (data,soh)

def load_test_data():
    root = '/Users/cruisin/Documents/BTM/SOH/data/charge'
    test_battery_id = 1
    test_all_x = []
    test_all_y = []
    for file in os.listdir(root):
        if file.endswith('batch-1.mat'):
            data = loadmat(os.path.join(root, file))
            battery = data['battery']
            test_battery = battery[0, test_battery_id - 1][0]
            print(f'test battery id: {test_battery_id}, test data shape: ', end='')
            test_x, test_y = parser_mat_data(test_battery)
            test_all_x.append(test_x)
            test_all_y.append(test_y)
    test_all_x = np.concatenate(test_all_x, axis=0)
    test_all_y = np.concatenate(test_all_y, axis=0)
    return test_all_x, test_all_y

def eval_metrix(true_label,pred_label):
    MAE = metrics.mean_absolute_error(true_label,pred_label)
    MAPE = metrics.mean_absolute_percentage_error(true_label,pred_label)
    MSE = metrics.mean_squared_error(true_label,pred_label)
    RMSE = np.sqrt(metrics.mean_squared_error(true_label,pred_label))

    return [MAE,MAPE,MSE,RMSE]

def main():
    test_all_x, test_all_y = load_test_data()
    print(test_all_x.shape, test_all_y.shape)
    c_rate = test_all_x[:,:,3]
    temp = test_all_x[:,:,6]
    Ah = test_all_x[:,:,4]
    cycle_num = test_all_x[:,:,2]
    print(test_all_y.shape)
    print(c_rate.shape)
    print(temp.shape)
    print(Ah.shape)
    print(cycle_num.shape)
    pred_label = sum_of_cycle_soh(cycle_num, c_rate, temp, Ah)
    print(pred_label.shape)
    #print(eval_metrix(test_all_y, pred_label))
if __name__ == '__main__':
    main()