# 計算函數積分，以黎曼和逼近
import numpy as np 
def f(x):
    return x**2 * np.sqrt(x**3 + 1)

# 設定積分上下界線
a = 0
b = 3
n_table = [5, 30, 100, 1000, 10000]
ans_array = []
for n in n_table:
    dx = (b - a) / n
    x = np.linspace(a, b, n+1)
    riemann_sum = np.sum(f(x[:-1]) * dx)
    ans_array.append(riemann_sum)

print(",  ".join(f'{ans:.6f}' for ans in ans_array)) # 儲存到小數點後6位
