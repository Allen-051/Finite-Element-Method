# problem 4.1 寫出一個break程式並說明其流程及結果
# 猜水果名稱
def guess_name(names):
    correct_name = '芒果' # 正確答案是'芒果'
    try:                  # 用try來檢驗是否猜對
        names = input('請猜Allen最愛吃的水果：')
        if names == correct_name:
            print('恭喜你答對了!\n')
            return True
        else:
            raise ValueError('歐歐...答錯了。小提示：是夏天才有的水果。再猜一次吧!')
        
    except ValueError as ve:
            print(ve)
            return False
# 用while迴圈讓使用者重複猜
while True:               
     if guess_name('芒果') == True:
          break

#  problem 4.2 用if取出1到100中不是2、3、5、7倍數的數字
result = []
for i in range(1, 101):
    if i % 2 != 0 and i % 3 != 0 and i % 5 != 0 and i % 7 != 0:
        result.append(i)

print(', '.join(map(str, result)))

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

# 設定兩條線分別劃出泰勒級數的前3項和前6項
y1 = taylor_series(x_axis, 3)
y2 = taylor_series(x_axis, 6)
# 劃出cos(x)圖形、y1、y2
plt.figure(figsize = (10,6))
plt.plot(x_axis, np.cos(x_axis), label = 'cos(x)', color = 'black', linestyle = '-')
plt.plot(x_axis, y1, label = 'Tatlor_series-n=3', color = 'red', linestyle = '--')
plt.plot(x_axis, y2, label = 'Tatlor_series-n=6', color = 'blue', linestyle = 'dashdot')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-5, 5)
plt.ylim(-3, 3)
plt.title('Cosine(x) using Taylor Series to approximate')
plt.legend()
plt.show()

