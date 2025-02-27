# problem 4.1 寫出一個break程式並說明其流程及結果


#  problem 4.2 用if取出1到100中不是2、3、5、7倍數的數字
result = []
for i in range(1, 101):
    if i % 2 != 0 and i % 3 != 0 and i % 5 != 0 and i % 7 != 0:
        result.append(i)

print(', '.join(map(str, result)))

# problem 4.3 用plot函數畫出三條曲線在同一張圖上，要不同線標、格式、顏色