import casadi as ca

def exp(x):
    return ca.exp(x)

def sqrt(x):
    return ca.sqrt(x)

def log(x):
    return ca.log(x)

def power(x,y):
    return ca.power(x, y)

def min(*args):
    """CasADi符号化的最小值函数，支持多参数"""
    return ca.fmin(*args)  # 解包参数传递给 ca.fmin

def max(*args):
    """CasADi符号化的最大值函数，支持多参数"""
    return ca.fmax(*args)  # 解包参数传递给 ca.fmax

def if_else(cond, expr_true, expr_false):
    """符号化条件判断函数"""
    return ca.if_else(cond, expr_true, expr_false)

def logical_eq(x, y):
    """符号化等于判断"""
    return ca.logic_and(x >= y, x <= y)  # CasADi的等式判断需特殊处理

def logical_le(x, y):
    """符号化小于等于判断"""
    return x <= y