import numpy as np
import statistics
from evaluate_air2cold import eva_air2cold
from evaluate_cold2air import eva_cold2air

num_dataset = 10
c2a_all_wrong = np.zeros(num_dataset)
c2a_all_edge = np.zeros(num_dataset)
a2c_all_wrong = np.zeros(num_dataset)
a2c_all_right = np.zeros(num_dataset)
node_path = "/home/lipeiwen.lpw/TECDI/data_real/realdata_preprocess/transfered/downstream/node_name.csv" # "node_name.csv"
label_path = "/home/lipeiwen.lpw/TECDI/downstream_evaluate/air2cold_label.csv" # "air2cold_label.csv"
    
for i in range(num_dataset):
    # data_path = f"/home/lipeiwen.lpw/TECDI/baseline/DYNOTEARS/downstream/DAG{i+1}.npy"
    # data_path = f"/home/lipeiwen.lpw/TECDI/baseline/PCMCI/baseline_results/downstream/PCMCI_est{i+1}.npy"
    # data_path = f"/home/lipeiwen.lpw/TECDI/exp_real/downstream/dataset{i+1}/train/DAG.npy"
    # data_path = f"/home/lipeiwen.lpw/TECDI/baseline/NeuralGC/results/downstream/NeuralGC_baseline/GC_est{i+1}.npy"
    data_path = f"/home/lipeiwen.lpw/TECDI/baseline/NeuralGC/results/downstream/dcdi_results_transfer/GC_est{i+1}.npy"
    c2a_wrong, c2a_edge = eva_cold2air(data_path, node_path)
    c2a_all_wrong[i] = c2a_wrong
    c2a_all_edge[i] = c2a_edge
    a2c_wrong, a2c_right = eva_air2cold(data_path, node_path, label_path)
    # print('********',a2c_wrong, a2c_right)
    a2c_all_wrong[i] = a2c_wrong
    a2c_all_right[i] = a2c_right
print('c2a_all_wrong:',round(np.mean(c2a_all_wrong),2),'\pm',round(statistics.stdev(c2a_all_wrong),2))
print('c2a_all_edge:',round(np.mean(c2a_all_edge),2),'\pm',round(statistics.stdev(c2a_all_edge),2))
print('a2c_all_wrong:',round(np.mean(a2c_all_wrong),2),'\pm',round(statistics.stdev(a2c_all_wrong),2))
print('a2c_all_right:',round(np.mean(a2c_all_right),2),'\pm',round(statistics.stdev(a2c_all_right),2))
