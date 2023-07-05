import warnings
import faiss
import numpy as np

try:
    from .rank_cylib.roc_cy import evaluate_roc_cy

    IS_CYTHON_AVAI = True
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn(
        'Cython roc evaluation (very fast so highly recommended) is '
        'unavailable, now use python evaluation.'
    )

def evaluate_roc_py(distmat, q_pids, g_pids, q_camids, g_camids):
    num_q, num_g = distmat.shape

    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    pos = []
    neg = []
    for q_idx in range(num_q):
        
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]    
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)        
        keep = np.invert(remove)
        raw_cmc = matches[q_idx][keep]
        sort_idx = order[keep]
        q_dist = distmat[q_idx]
        ind_pos = np.where(raw_cmc == 1)[0]
        pos.extend(q_dist[sort_idx[ind_pos]])
        ind_neg = np.where(raw_cmc == 0)[0]
        neg.extend(q_dist[sort_idx[ind_neg]])

    scores = np.hstack((pos, neg))

    labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))
    return scores, labels

def evaluate_roc(
        distmat,
        q_pids,
        g_pids,
        q_camids,
        g_camids,
        use_cython=True
):
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_roc_cy(distmat, q_pids, g_pids, q_camids, g_camids)
    else:
        return evaluate_roc_py(distmat, q_pids, g_pids, q_camids, g_camids)
