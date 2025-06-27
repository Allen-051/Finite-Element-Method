import pandas as pd

def find_umax(csv_file):
    # 讀取 csv 檔案
    df = pd.read_csv(csv_file, header=None)

    # 讀取 u1 與 u2（假設在第 3、4 列，索引為 2、3）
    u1 = df.iloc[:, 2].values.astype(float)  # 排除第 0 欄（例如 "u1"）
    u2 = df.iloc[:, 3].values.astype(float)

    # 計算總位移 u
    u = (u1**2 + u2**2)**0.5

    # 找出最大位移的 index
    max_idx = u.argmax()
    umax = u[max_idx]
    u1_max = u1[max_idx]
    u2_max = u2[max_idx]

    # 找到對應的 Joint index（來自第 1 列）
    joint_index = df.iloc[max_idx , 0]  # +1 因為跳過第 0 欄

    return joint_index, umax, u1_max, u2_max
    

# 請使用者再輸入一個csv檔案位置後，從這個csv檔案裡面，找尋Joint_index，最後該列所在的資訊
def find_buckling_point(csv_file, Joint_index):
    # 讀取csv檔案
    df = pd.read_csv(csv_file, header=None)
    # 在這個檔案的第一行(直)找尋Joint_index
    if Joint_index in df.iloc[:, 0].values:
        # 找到Joint_index所在的列
        index = df.iloc[:, 0].values.tolist().index(Joint_index)  

        # 回傳該列第四行(直)的資訊
        buckling_info = df.iloc[index, 3]  # 假設第四行是z座標資訊
        z_coord = buckling_info - 0.2
        return z_coord
    else:
        return None

def main():
    # 輸入csv檔案位置
    buckling_file = input("請輸入buckling.csv檔案位置: ")
    # 如果csv檔案不存在，則提示錯誤
    try:
        with open(buckling_file, 'r') as f:
            pass
    except FileNotFoundError:
        print("檔案不存在，請確認檔案路徑是否正確。")
        return
    # 呼叫find_umax函數，取得Joint_index和umax
    Joint_index, umax, u1_max, u2_max = find_umax(buckling_file)
    print(f"Joint Index: {Joint_index}, Umax: {umax:.4f}, u1: {u1_max:.4f}, u2: {u2_max:.4f}")
    # 呼叫find_buckling_point函數，取得buckling_info
    coord_path = input("請輸joint.csv檔案位置: ")
    z_coord = find_buckling_point(coord_path, Joint_index)
    # 如果找不到該Joint_index，則提示錯誤
    if z_coord is None:
        print(f"找不到Joint No. {Joint_index} 的座標資訊。")
    else:
        print(f"Joint No.{Joint_index} 的z座標: {z_coord}m")

if __name__ == "__main__":
    main()
