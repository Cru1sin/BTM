import torch
import os
import numpy as np
from SOH.inference import SOH_predictor

if __name__ == "__main__":
    soh_predictor = SOH_predictor(dt=30, charge_time=0, cycle_num=300)
    data1 = [1,3.7,25]
    data2 = [6,4,45]
    soh_loss1 = 0
    soh_loss2 = 0
    for i in range(1):
        predictions1 = soh_predictor.inference(*data1)
        predictions2 = soh_predictor.inference(*data2)
        soh_loss1 += predictions1
        soh_loss2 += predictions2
    print(f"soh_loss1: {soh_loss1}")
    print(f"soh_loss2: {soh_loss2}")
