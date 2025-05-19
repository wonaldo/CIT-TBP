from cdt.metrics import SHD
from sklearn.metrics import f1_score
import numpy as np




def adj_matrix(nnodes, dic):
    adj = np.zeros((nnodes, nnodes), dtype=int)
    for src, targets in dic.items():
        for target in targets:
            adj[src, target] = 1
    # There are algorithms that do not recognize autocorrelation, and in order to avoid autocorrelation affecting metric evaluation, the diagonals are all set to 1.
    np.fill_diagonal(adj, 1)
    return adj

def SHD_distance(D1,D2):
    shd=SHD(D1,D2,double_for_anticausal=False)
    return shd


# fdr
def count_accuracy(true_adj, pred_adj):
    ## fdr: (reverse + false positive) / prediction positive
    cond=np.flatnonzero(true_adj)
    pred=np.flatnonzero(pred_adj) 
 
    cond_reversed=np.flatnonzero(true_adj.T)
    cond_skeleton=np.concatenate([cond,cond_reversed])

    # reverse
    extra = np.setdiff1d(pred,cond,assume_unique=True)
    reverse=np.intersect1d(extra,cond_reversed,assume_unique=True)  
    
    # false positive
    false_pos=np.setdiff1d(pred,cond_skeleton,assume_unique=True) 

    # total num
    pred_size=len(pred)

    # fdr
    ## How many reversals and errors are there where predictions have an edge.
    fdr=float(len(reverse)+len(false_pos))/max(pred_size,1) 
    return fdr

## f1-score
def compute_f1(true_adj,pred_adj):
    pred_adj=np.array(pred_adj)
    true_adj=np.array(true_adj)
    y_pred=pred_adj.flatten()
    y_true=true_adj.flatten()
    f1=f1_score(y_true, y_pred)
    return f1