# 輸入一個csv檔案位置
# 讀取csv檔案
import pandas as pd
def find_umax(csv_file):
    # 讀取csv檔案
    df = pd.read_csv(csv_file, header=None)
    # 取joint_number（假設在第1欄，從第2列開始）
    joint_numbers = df.iloc[1:, 0].reset_index(drop=True)
    # 取u1（第2行，去掉第1欄）
    u1 = df.iloc[1, 1:].astype(float).reset_index(drop=True)
    # 取u2（第3行，去掉第1欄）
    u2 = df.iloc[2, 1:].astype(float).reset_index(drop=True)
    # 計算平方和
    u_sum = (u1**2 + u2**2)**0.5
    # 找最大值及其索引
    umax = u_sum.max()
    umax_index = u_sum.idxmax()
    umax_joint = joint_numbers.iloc[umax_index]
    return umax, umax_joint

# 讀取一個txt檔案位置
def read_txt_file(file_path, umax_joint):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read().strip()
    except UnicodeDecodeError:
        with open(file_path, 'r', encoding='cp950') as file:
            content = file.read().strip()
    # 只在TABLE:  JOINT COORDINATES之後搜尋
    lines = content.split('\n')
    start_idx = None
    for idx, line in enumerate(lines):
        if 'TABLE:' in line and 'JOINT COORDINATES' in line:
            start_idx = idx + 1
            break
    if start_idx is not None:
        search_lines = lines[start_idx:]
    else:
        search_lines = lines
    keyword = f"Joint={umax_joint}"
    found = False
    for line in search_lines:
        if keyword in line:
            print(f"Found Joint={umax_joint} in the file.")
            print('\n'+line+'\n')
            found = True
            break
    if not found:
        print(f"\nJoint={umax_joint} not found in the file after 'TABLE:  JOINT COORDINATES'.")
    return content
def main():
    # 輸入csv檔案位置
    csv_file = input("請輸入csv檔案位置: ")
    # 呼叫find_umax函數
    umax, umax_joint = find_umax(csv_file)
    print(f"最大值: {umax}, Joint: {umax_joint}")
    
    # 輸入txt檔案位置
    txt_file = input("請輸入txt檔案位置: ")
    # 呼叫read_txt_file函數
    read_txt_file(txt_file, umax_joint)
if __name__ == "__main__":
    main()
