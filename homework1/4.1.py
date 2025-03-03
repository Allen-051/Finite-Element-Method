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