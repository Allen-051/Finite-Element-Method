# problem 4.1 寫出一個break程式並說明其流程及結果

def find_first_even(numbers):
    """
    找到列表中第一個偶數並返回它。如果沒有找到偶數，返回 None。
    """
    for number in numbers:
        if number % 2 == 0:
            return number
        # 當找到第一個偶數時，使用 break 結束循環
        break
    return None

# 測試範例
numbers = [1, 3, 5, 7, 11, 19]
result = find_first_even(numbers)
print(f'The first even number is {result}.')