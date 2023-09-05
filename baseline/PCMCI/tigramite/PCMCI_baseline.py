# %%
# Imports
import numpy as np
from matplotlib import pyplot as plt    

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr

# %%
graph = 'downstream' # 5_p1_smp3000, 10_p1_smp5500, downstream
if graph == '5_p1_smp3000':
    num_nodes = 5
    data = 'simu'
elif graph == '10_p1_smp5500':
    num_nodes = 10
    data = 'simu'
elif graph == 'downstream':
    num_nodes = 38
    data = 'real'

for i in range(1,11):
    file_name = 'data'+str(i)+'.npy'
    path = f'/home/lipeiwen.lpw/TECDI/data_{data}/{graph}/'+file_name
    data1 = np.load(path)
    data1 = data1[:,0:num_nodes]
    # 检查每一列是否都一样
    col_equal = [np.allclose(data1[:, i], data1[0, i]) for i in range(data1.shape[1])]

    # 删除重复列
    data1 = np.delete(data1, np.where(col_equal)[0], axis=1)
    col_num = data1.shape[1]
    # mean = np.mean(data1)
    # std = np.std(data1)
    # data = (data1 - mean) / std
    dataframe = pp.DataFrame(data1)
    parcorr = ParCorr()
    pcmci_parcorr = PCMCI(
    dataframe=dataframe, 
    cond_ind_test=parcorr,
    verbosity=1)
    results = pcmci_parcorr.run_pcmci(tau_max=2, pc_alpha=0.2)
    pcmci_parcorr.print_significant_links(
    p_matrix = results['p_matrix'], 
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)
    link_matrix = pcmci_parcorr.return_significant_parents(pq_matrix=results['p_matrix'],
                        val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
    result_mat = np.zeros([col_num*2,col_num*2])
    result_mat[0:col_num,0:col_num] = link_matrix[:,:,0]+0
    result_mat[col_num:col_num*2,0:col_num] = link_matrix[:,:,1]+0
    np.save(f'/home/lipeiwen.lpw/TECDI/baseline/PCMCI/baseline_results/{graph}/PCMCI_est{i}',result_mat)



