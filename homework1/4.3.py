# problem 4.3 用plot函數畫出三條曲線在同一張圖上，要不同線標、格式、顏色
import matplotlib.pyplot as plt
import numpy as np
import math
# 劃出泰勒級數，取f(X) = COS(X)
x_axis = np.linspace(-2*math.pi, 2*math.pi, 200)
def taylor_series(x, n):
    sum = 0
    for i in range(n):
        coef = (-1)**i / math.factorial(2*i)
        sum += coef * x**(2*i)
    return sum

# 設定三條線分別劃出泰勒級數的前1項、前3項、前5項
y1 = taylor_series(x_axis, 1)
y2 = taylor_series(x_axis, 3)
y3 = taylor_series(x_axis, 5)
# 劃出cos(x)圖形、y1、y2、y3
plt.figure(figsize = (10,6))
plt.plot(x_axis, np.cos(x_axis), label = 'cos(x)', color = 'black', linestyle = '-',linewidth = 1)
plt.plot(x_axis, y1, label = 'Tatlor_series-n=1', color = 'green', linestyle = 'dotted',linewidth = 2)
plt.plot(x_axis, y2, label = 'Tatlor_series-n=3', color = 'red', linestyle = '--',linewidth = 2)
plt.plot(x_axis, y3, label = 'Tatlor_series-n=5', color = 'blue', linestyle = 'dashdot',linewidth = 2)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.title('Cosine(x) using Taylor Series to approximate')
plt.legend()
plt.show()


