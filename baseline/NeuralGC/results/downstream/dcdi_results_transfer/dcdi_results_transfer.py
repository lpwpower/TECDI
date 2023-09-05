import numpy as np
num_nodes = 38
for i_dataset in range(1,11):
    estdag = np.load(f'/home/lipeiwen.lpw/TECDI/exp_real/downstream/dataset{i_dataset}/train/DAG.npy')
    GC_est = estdag[:num_nodes,:num_nodes] + estdag[num_nodes:,:num_nodes]
    GC_est = np.where(GC_est==0,GC_est,1)
    np.save(f'/home/lipeiwen.lpw/TECDI/baseline/NeuralGC/results/downstream/dcdi_results_transfer/GC_est{i_dataset}.npy',GC_est)