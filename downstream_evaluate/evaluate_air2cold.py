import numpy as np
import pandas as pd

def eva_air2cold(data_path, node_path, label_path):
    # 读取原始矩阵 (20+18)*2
    data = np.load(data_path)
    # print("data shape:", data.shape)
    
    # 读取节点名称
    node_name = pd.read_csv(node_path, index_col=0)["0"].tolist()
    # print("node num:", len(node_name))

    # 读取标注label
    label = pd.read_csv(label_path, index_col=0)
    # print("node num:", len(node_name))
    
    # 提取需要测评的矩阵-即左半边矩阵
    data_extract = data[:, :38]
    # print(data_extract.shape)

    # 转化为DataFrame
    if data_extract.shape[0]==38:
        df_data_extract = pd.DataFrame(data_extract, index=node_name, columns=node_name)
    else:
        df_data_extract = pd.DataFrame(data_extract, index=node_name+node_name, columns=node_name)
    # print(df_data_extract.head())

    # 提取出异常测评DataFrame- 即index中出现'冷通道温度'的行，与columns中不出现'冷通道温度'的列
    wrong_num = 0
    right_num = 0

    for i in range(len(df_data_extract)):
        row = df_data_extract.iloc[i]
        name = row.name   # 实际就是该行的index
        
        if "送风温度" in name:  # 仅考虑上游对下游是否判错
            air2cold_label = label.loc[name]   # 提取该空调的 label
            
            selected_label = air2cold_label[air2cold_label==1].index.tolist()  # 选出该空调《应该》影响的冷通道清单。
            
            label_counts = row[selected_label]   # 选出对应的冷通道的值
            
            right_num += len(label_counts[label_counts==1])  # 1: 判断正确的数目
            wrong_num += len(label_counts[label_counts==0])  # 0: 未召回的数目

    print("wrong_num:", wrong_num, "right_num:", right_num)

    # 找出值为1的位置
    # pos = df_data_extract.where(df_data_extract == 1).stack().reset_index()
    # print(pos)

    
    return wrong_num, right_num

# 下游对上游判伪，  42  0  330
# 上游对下游判伪，  136  0  372

if __name__ == "__main__":
    all_wrong_nums = 0
    all_right_nums = 0

    node_path = "/home/lipeiwen.lpw/TECDI/data_real/realdata_preprocess/transfered/downstream/node_name.csv" # "node_name.csv"
    label_path = "/home/lipeiwen.lpw/TECDI/downstream_evaluate/air2cold_label.csv" # "air2cold_label.csv"
    
    for i in range(1,11):
    # for i in [6]:
        # data_path = f"./downstream_less2/GC_est{i}.npy"
        # data_path = f"./downstream_less/dataset{i}/train/DAG.npy"
        # data_path = f"./dynotears_downstream_less/DAG{i}.npy"
        # data_path = f"./PCMCI_downstream_less/result_data{i}.npy"
        data_path = f'/home/lipeiwen.lpw/TECDI/baseline/DYNOTEARS/downstream/DAG{i}.npy'
        # data_path = f'/home/lipeiwen.lpw/TECDI/exp_real/downstream_old/dataset{i}/train/DAG.npy'

        wrong_num, right_num = eva_air2cold(data_path, node_path, label_path)
        
        all_wrong_nums += wrong_num
        all_right_nums += right_num

    print('all_wrong_nums:', all_wrong_nums, "all_right_num:", all_right_nums)
    


