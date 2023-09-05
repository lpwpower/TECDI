import numpy as np
import pandas as pd


def eva_cold2air(data_path, node_path):
    # 读取原始矩阵 (20+18)*2
    data = np.load(data_path)
    edge_num = np.count_nonzero(data==1)
    print("data shape:", data.shape)
    
    # 读取节点名称
    node_name = pd.read_csv(node_path, index_col=0)["0"].tolist()
    print("node num:", len(node_name))

    # 提取需要测评的矩阵-即左半边矩阵
    data_extract = data[:, :38]
    print("half data shape:", data_extract.shape)

    # 转化为DataFrame
    if data_extract.shape[0]==38:
        df_data_extract = pd.DataFrame(data_extract, index=node_name, columns=node_name)
    else:
        df_data_extract = pd.DataFrame(data_extract, index=node_name+node_name, columns=node_name)
    # print(df_data_extract.head())

    # 提取出异常测评DataFrame-即字段名中出现'冷通道温度'的行，与字段名中不出现'冷通道温度'的列
    df_rows_extract = df_data_extract.loc[df_data_extract.index.str.contains("冷通道温度")]    
    df_rows_cols_extract = df_rows_extract.loc[:, ~df_rows_extract.columns.str.contains("冷通道温度")]
    print("extract data shape:", df_rows_cols_extract.shape)
    # df_rows_cols_extract.head()

    # 计算不符合实际的数目和 
    # 直接对dataframe求和即为（冷通道温度，送风温度）值为1的总数。
    wrong_num = int(df_rows_cols_extract.sum().sum())
    print("wrong num:", wrong_num)

    # 找出值为1的位置
    pos = df_rows_cols_extract.where(df_rows_cols_extract == 1).stack().reset_index()
    print(pos)
    
    return wrong_num, edge_num


if __name__ == "__main__":
    num_dataset = 10
    all_wrong = np.zeros(num_dataset)
    all_num_edge = np.zeros(num_dataset)
    for i in range(num_dataset):
        data_path = f"/home/lipeiwen.lpw/TECDI/exp_real/downstream/dataset{i+1}/train/DAG.npy"
        node_path = "/home/lipeiwen.lpw/TECDI/data_real/realdata_preprocess/transfered/downstream_less/node_name.csv"
        wrong_num, edge_num = eva_cold2air(data_path, node_path)
        all_wrong[i] = wrong_num
        all_num_edge[i] = edge_num
        # print(wrong_num)
    print('wrong mean:',np.mean(all_wrong),'\n','wrong stdev:',np.std(all_wrong))
    print('edge mean:',np.mean(all_num_edge),'\n','edge stdev:',np.std(all_num_edge))