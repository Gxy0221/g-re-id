import torch
import numpy as np
import torch.nn.functional as F

def aqe(query_feat: torch.tensor, gallery_feat: torch.tensor,
        qe_times: int = 1, qe_k: int = 10, alpha: float = 3.0):
    num_query = query_feat.shape[0]
    all_feat = torch.cat((query_feat, gallery_feat), dim=0)
    norm_feat = F.normalize(all_feat, p=2, dim=1)

    all_feat = all_feat.numpy()
    for i in range(qe_times):
        all_feat_list = []
        sims = torch.mm(norm_feat, norm_feat.t())
        sims = sims.data.cpu().numpy()
        for sim in sims:
            init_rank = np.argpartition(-sim, range(1, qe_k + 1))
            weights = sim[init_rank[:qe_k]].reshape((-1, 1))
            weights = np.power(weights, alpha)
            all_feat_list.append(np.mean(all_feat[init_rank[:qe_k], :] * weights, axis=0))
        all_feat = np.stack(all_feat_list, axis=0)
        norm_feat = F.normalize(torch.from_numpy(all_feat), p=2, dim=1)

    query_feat = torch.from_numpy(all_feat[:num_query])
    gallery_feat = torch.from_numpy(all_feat[num_query:])
    return query_feat, gallery_feat
