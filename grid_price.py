def get_electricity_price(self, time_index):
    """根据不同时间段返回买电和卖电价格"""
    # 计算当前时间对应的小时
    hour = (time_index * self.dt) / 3600  # 秒数转换为小时
    hour = int(hour % 24)  # 确保 hour 在 0-23 之间

    if 9 <= hour < 11 or 15 <= hour < 17:  # 尖峰时段
        p_buy = 1.2  # 买电价格 (元/kWh)
    elif 8 <= hour < 9 or 17 <= hour < 23:  # 高峰时段
        p_buy = 1.0
    elif 13 <= hour < 15 or 23 <= hour < 24:  # 平段时段
        p_buy = 0.8
    else:  # 低谷时段
        p_buy = 0.5
    
    p_sell = 0.8 * p_buy  # 假设卖电价是买电价的80%

    return p_buy, p_sell