# problem 4.1 寫出一個break程式並說明其流程及結果
def guess_name(names):
    correct_name = '芒果'
    try:
        names = input('請猜Allen最愛吃的水果：')
        if names == correct_name:
            print('恭喜你答對了!\n')
            return True
        else:
            raise ValueError('歐歐...答錯了。小提示：是夏天才有的水果。再猜一次吧!')
        
    except ValueError as ve:
            print(ve)
            return False
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