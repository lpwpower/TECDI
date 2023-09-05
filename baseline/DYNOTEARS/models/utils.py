import math
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Dict, List, Tuple, Union
from causalnex.structure.transformers import DynamicDataTransformer
# from sklearn import preprocessing 

def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
	# 	print "---  new folder...  ---"
	# 	print "---  OK  ---"
	# else:
	# 	print "---  There is this folder!  ---"

# -------------------------------------------------------------------------------------------
def get_info(data_num, graph):
    room = 1 if data_num>=7 and data_num<=9 else 2
    # 设备序号
    sensor = []
    crah = []
    if room == 2:
        for i in range(18,36): sensor.append('sensor_{}_'.format(i))
        for i in range(20,40): crah.append('crah_{}_'.format(i))
    else:
        for i in range(0,18): sensor.append('sensor_{}_'.format(i))
        for i in range(40,60): crah.append('crah_{}_'.format(i))
    # 上下游关系
    if graph == 'upstream_crah':
        upstream = ['水路进水温度', '水阀开度', '回风温度']
        downstream = ['送风温度']
    if graph == 'downstream_sensor':
        upstream = ['送风温度','风机转速']
        downstream = ['冷通道温度']
    return room, sensor, crah, upstream, downstream



# --------------------------------------------删除列--------------------------------------------
def delete_redundant_col(data,allcrah,allsensor,up_point,down_point):
    all_point_name = ['水路进水温度', '水阀开度', '回风温度', '送风温度', '风机转速', '冷通道温度']
    delete_point_name = list(set(all_point_name) - set(up_point) - set(down_point))
    for point in delete_point_name:
        if point != '冷通道温度':
            for i in allcrah:
                data = data.drop(columns=i+point)
        else:
            for i in allsensor:
                data = data.drop(columns=i+point)
    return data



# ----------------------------------------用先验知识去除边-------------------------------------------
def preknowledge(graph,room,lag,sensor,crah,del_diff_crahroom=True, del_betw_fan_speed=True, del_down2up=True, up_point=[], down_point=[]):
    '''
    加先验知识
    tabu_edges: list of edges(lag, from, to) not to be included in the graph. `lag == 0` implies that the edge is
            forbidden in the INTRA graph (W), while lag > 0 implies an INTER weight equal zero.
    tabu_parent_nodes: list of nodes banned from being a parent of any other nodes.
    tabu_child_nodes: list of nodes banned from being a child of any other nodes.
    '''
    del_edge = []

    # 两个空调包间的intra+inter关系去掉 down:(10*2)*(10*2)*(1+10)*2 up:(10*5)*(10*5)*(1+10)*2
    if del_diff_crahroom:

        if graph == 'downstream_sensor': crah_point = ['送风温度', '风机转速']# 空调信息
        if graph == 'upstream_crah': crah_point = ['水路进水温度', '水阀开度', '回风温度', '送风温度']
        room1crah = [] #10
        room2crah = [] #10
        if room == 1:
            for i in range(40,50): room1crah.append('crah_{}_'.format(i))
            for i in range(50,60): room2crah.append('crah_{}_'.format(i))
        if room == 2:
            for i in range(30,40): room1crah.append('crah_{}_'.format(i))
            for i in range(20,30): room2crah.append('crah_{}_'.format(i))

        for crah1 in room1crah:
            for crah2 in room2crah:
                [del_edge.append((l, crah1+i, crah2+j)) for l in range(lag+1) for i in crah_point for j in crah_point]
                [del_edge.append((l, crah2+i, crah1+j)) for l in range(lag+1) for i in crah_point for j in crah_point]

    # 风机转速同级之间的intra+inter关系去掉 down:20*20*(1+10) up:20*20*(1+10)
    if graph == 'downstream_sensor' and del_betw_fan_speed:
        [del_edge.append((l, i+'风机转速', j+'风机转速')) for l in range(lag+1) for i in crah for j in crah]

    # 下游到上游的关系intra+inter关系去掉 down:18*(20*2)*(1+10) up:(20*2)*(20*3)*(1+10)
    if del_down2up:
        down_device = sensor if graph == 'downstream_sensor' else crah
        for up in up_point:
            for down in down_point:
                [del_edge.append((l, i+down, j+up)) for l in range(lag+1) for i in down_device for j in crah]
                
    return del_edge



def cal_acc(y, y_hat, method):
    """
    Args:
        y
        y_hat
        method: 'MSE', 'RMSE', 'MAE', 'MAPE'
    """
    if method == 'MSE':  acc = metrics.mean_squared_error(y, y_hat)
    if method == 'RMSE': acc = metrics.mean_squared_error(y, y_hat)**0.5
    if method == 'MAE':  acc = metrics.mean_absolute_error(y, y_hat)
    if method == 'MAPE': acc = metrics.mean_absolute_percentage_error(y, y_hat)
    return acc

def cal_mtx_acc(X,Xpred,method):
    """
    Args:
        X
        Xpred
        method: 'MSE', 'RMSE', 'MAE', 'MAPE'
    """
    assert X.shape==Xpred.shape
    acc = 0
    d_vars = X.shape[1]
    for i in range(d_vars):
        acc_i = cal_acc(X[:,i],Xpred[:,i],method)
        acc = acc + acc_i
    acc = acc / d_vars
    return acc

# -------------------------------------------预测-----------------------------------------------
# prediction = XW + YA, Y = Xlags
def pred_by_point(
        time_series: Union[pd.DataFrame, List[pd.DataFrame]],
        p_orders: int,
        W,
        A,
        del_W=False):
    """
    Method: predict point by point based on true test data
    Args:
        p_orders (int): number of past indexes we to use
        W: estimated intra-slice adjacency matrices
        A: estimated inter-slice adjacency matrices
        del_W: if del_W == True, only use A to predict
    """
    if del_W == True: W = np.zeros(W.shape)
    time_series = [time_series] if not isinstance(time_series, list) else time_series
    X, Xlags = DynamicDataTransformer(p=p_orders).fit_transform(time_series, return_df=False)
    n, d_vars = X.shape # d_vars (int): number of variables in the model
    pred = X.dot(W) + Xlags.dot(A)
    # loss = (
    #         1
    #         / n
    #         * np.square(
    #             np.linalg.norm(
    #                 X.dot(np.eye(d_vars, d_vars)) - pred, "fro"
    #             )
    #         )
    #     )
    return pred #, loss

# ---------------------------------find root nodes at present-------------------------------------
def find_present_roots(w,remains): # w[source,target] can be source, can not be target
    roots = []
    w = w[remains,:]
    for i in remains:
        if np.all(w[:,i]==0):
            roots.append(i)
    remains = list(set(remains)-set(roots))
    return roots, remains

def pred_one_time( # X=testdata[10:,:], Xlags=testdata[10:,:]的lag版本
        Xlags_i,
        W,
        A,
        d_vars):
    remain_nodes = list(range(d_vars))
    p_once = np.zeros((1,d_vars))
    while remain_nodes != []:
        p_once = p_once.dot(W) + Xlags_i.dot(A)
        roots, remain_nodes = find_present_roots(W, remain_nodes) # delete nodes have been calculated
        # print(roots)
    return p_once

def pred_by_len(
        time_series: Union[pd.DataFrame, List[pd.DataFrame]],
        p_orders: int,
        W,
        A,
        pred_len=10):
    time_series = [time_series] if not isinstance(time_series, list) else time_series
    X, Xlags = DynamicDataTransformer(p=p_orders).fit_transform(time_series, return_df=False)
    n, d_vars = X.shape # X:n*d, Xlags:n*pd, n=M(T+1-p), X=testdata[10:,:], Xlags=testdata[10:,:]的lag版本
    pred_times = math.ceil(n/pred_len)
    pred = np.empty((0,d_vars), float)
    for i in range(pred_times):
        Xlags_i = Xlags[i*pred_len:i*pred_len+1,:] # use pred_len get Xlag_i for predicting i:i+10
        for j in range(pred_len):
            p_once = pred_one_time(Xlags_i,W,A,d_vars)
            pred = np.append(pred, p_once, axis=0)
            Xlags_i = np.concatenate((p_once,Xlags_i[:,:(Xlags.shape[1]-d_vars)]),axis=1) #np.append(Xlags_i, np.array([p_once]), axis=0) # 横向拼接
            if pred.shape[0] == n:
                return pred
    return 

