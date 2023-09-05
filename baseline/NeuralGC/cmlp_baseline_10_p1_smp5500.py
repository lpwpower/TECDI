import os
import pandas as pd
import torch
import numpy as np
from models.cmlp import cMLP, cMLPSparse, train_model_ista, train_unregularized

device = torch.device('cuda:0')
time_points = 500
num_nodes = 10
load_path = '/home/lipeiwen.lpw/TECDI/data_simu/10_p1_smp5500/'
save_path = '/home/lipeiwen.lpw/TECDI/baseline/NeuralGC/results/10_p1_smp5500/NeuralGC_baseline/'
os.makedirs(save_path, exist_ok=True)

for i_dataset in range(1,11):
    X_np = np.load(load_path + f'data{i_dataset}.npy')[:,:num_nodes]
    df_g = pd.DataFrame(X_np)
    df_g = df_g.apply(lambda x : (x-np.min(x))/(np.max(x)-np.min(x)))
    X_np = df_g.to_numpy()
    X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

    cmlp = cMLP(X.shape[-1], lag=1, hidden=[100]).cuda(device=device)

    train_loss_list = train_model_ista(
        cmlp, X, lam=0.01, lam_ridge=0.01, lr=1e-2, penalty='H', max_iter=50000,
        check_every=100)
    
    GC_est = cmlp.GC().cpu().data.numpy()
    np.save(save_path+f'GC_est{i_dataset}.npy',GC_est)



