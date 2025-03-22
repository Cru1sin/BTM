import casadi
# --------------- python --------------
opti = casadi.Opti()    # 实例化一个 opti

x = opti.variable()     # 声明变量
y = opti.variable()

opti.minimize(  (y-x**2)**2   )     # 优化目标
opti.subject_to( x**2+y**2==1 )     # 约束1
opti.subject_to(       x+y>=1 )     # 约束2

opti.solver('ipopt')    # 设置求解器

sol = opti.solve()      # 求解

print(sol.value(x))
print(sol.value(y))

"""
Opti stack 的主要特征有：

允许更自然的语法来描述约束，不用像第4章一样用上下界
Indexing/bookkeeping of decision variables is hidden.
数值类型与原编程语言具有更紧密的映射，没有 DM 符号

Variables: 声明任意数量的决策变量（要求解的变量）

x = opti.variable(): 标量、 
 的矩阵
x = opti.variable(5): 具有5个元素的列向量、 
 矩阵
x = opti.variable(5,3): 
 矩阵
x = opti.variable(5,5,'symmetric'): 
 对称矩阵
求解器遵循声明变量的顺序。请注意，变量实际上是普通的 MX 符号。可以对它们执行任何 CasADi MX 操作。

Parameters: 声明任意数量的参数。必须在求解之前将它们固定为特定的数值，并且可以随时重新赋值。
"""

# --------------- python -------------
p = opti.parameter()
opti.set_value(p, 3)    # 设置 p 的值为 3

# ----------- python ----------
opti.subject_to([x*y>=1,x==3])  # 同时设置一个等式和不等式约束

# ------------ python -------------
A = opti.variable(5,5)
opti.subject_to( casadi.vec(A)<=3 )

"""
每一个 subject_to(约束) 命令，会增量式地添加一个约束到约束集中，
调用 subject_to() 函数，不带任何约束，则会清空约束集中的所有约束，从头再来。
"""

"""
Solver: 优化器的声明是必要的。在声明优化器时，第2个可选参数为 CasADi 的配置，
第3个可选参数为 solver 优化器的配置，均为字典类型(for python) / 结构体类型 (for matlab)。
"""
# ------------- python ----------
p_opts = {"expand":True}    # Casadi 配置
s_opts = {"max_iter": 100}  # solver 配置
opti.solver("ipopt",p_opts,s_opts)

# ------------ python ------------
opti.set_initial(x, 2)
opti.set_initial(10*x[0], 2)