import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from scipy.stats import norm
from sklearn import metrics
from .file_io import PathManager

class Visualizer:
    def __init__(self, dataset):
        self.dataset = dataset
    def get_model_output(self, all_ap, dist, q_pids, g_pids, q_camids, g_camids):
        self.all_ap = all_ap
        self.dist = dist
        self.sim = 1 - dist
        self.q_pids = q_pids
        self.g_pids = g_pids
        self.q_camids = q_camids
        self.g_camids = g_camids

        self.indices = np.argsort(dist, axis=1)
        self.matches = (g_pids[self.indices] == q_pids[:, np.newaxis]).astype(np.int32)

        self.num_query = len(q_pids)
    def get_matched_result(self, q_index):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        order = self.indices[q_index]
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)
        cmc = self.matches[q_index][keep]
        sort_idx = order[keep]
        return cmc, sort_idx
    def save_rank_result(self, query_indices, output, max_rank=5, vis_label=False, label_sort='ascending',
                         actmap=False):
        if vis_label: 
            fig, axes = plt.subplots(2, max_rank + 1, figsize=(3 * max_rank, 12))
        else:
            fig, axes = plt.subplots(1, max_rank + 1, figsize=(3 * max_rank, 6))
        for cnt, q_idx in enumerate(tqdm.tqdm(query_indices)):
            all_imgs = []
            cmc, sort_idx = self.get_matched_result(q_idx)
            query_info = self.dataset[q_idx]
            query_img = query_info['images']
            cam_id = query_info['camids']
            query_name = query_info['img_paths'].split('/')[-1]
            all_imgs.append(query_img)
            query_img = np.rollaxis(np.asarray(query_img.numpy(), dtype=np.uint8), 0, 3)
            plt.clf()
            ax = fig.add_subplot(1, max_rank + 1, 1)
            ax.imshow(query_img)
            ax.set_title('{:.4f}/cam{}'.format(self.all_ap[q_idx], cam_id))   
            for i in range(max_rank):
                if vis_label:
                    ax = fig.add_subplot(2, max_rank + 1, i + 2)
                else:
                    ax = fig.add_subplot(1, max_rank + 1, i + 2)
                g_idx = self.num_query + sort_idx[i]
                gallery_info = self.dataset[g_idx]
                gallery_img = gallery_info['images']
                cam_id = gallery_info['camids']
                all_imgs.append(gallery_img)
                gallery_img = np.rollaxis(np.asarray(gallery_img, dtype=np.uint8), 0, 3)
                if cmc[i] == 1:
                    label = 'true'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1, edgecolor=(0, 1, 0),
                                               fill=False, linewidth=5))
                else:
                    label = 'false'
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1,
                                               edgecolor=(1, 0, 0), fill=False, linewidth=5))
                ax.imshow(gallery_img)
                ax.set_title(f'{self.sim[q_idx, sort_idx[i]]:.3f}/{label}/cam{cam_id}')  
                ax.axis("off")
           
            if vis_label:
                label_indice = np.where(cmc == 1)[0]
                if label_sort == "ascending": label_indice = label_indice[::-1]
                label_indice = label_indice[:max_rank]
                for i in range(max_rank):
                    if i >= len(label_indice): break
                    j = label_indice[i]
                    g_idx = self.num_query + sort_idx[j]
                    gallery_info = self.dataset[g_idx]
                    gallery_img = gallery_info['images']
                    cam_id = gallery_info['camids']
                    gallery_img = np.rollaxis(np.asarray(gallery_img, dtype=np.uint8), 0, 3)
                    ax = fig.add_subplot(2, max_rank + 1, max_rank + 3 + i)
                    ax.add_patch(plt.Rectangle(xy=(0, 0), width=gallery_img.shape[1] - 1,
                                               height=gallery_img.shape[0] - 1,
                                               edgecolor=(0, 0, 1),
                                               fill=False, linewidth=5))
                    ax.imshow(gallery_img)
                    ax.set_title(f'{self.sim[q_idx, sort_idx[j]]:.3f}/cam{cam_id}')
                    ax.axis("off")

            plt.tight_layout()
            filepath = os.path.join(output, "{}.jpg".format(cnt))
            fig.savefig(filepath)

    def vis_rank_list(self, output, vis_label, num_vis=300, rank_sort="ascending", label_sort="ascending", max_rank=5,
                      actmap=False):
        assert rank_sort in ['ascending', 'descending'], "{} not match [ascending, descending]".format(rank_sort)

        query_indices = np.argsort(self.all_ap)
        if rank_sort == 'descending': query_indices = query_indices[::-1]

        query_indices = query_indices[:int(num_vis)]
        self.save_rank_result(query_indices, output, max_rank, vis_label, label_sort, actmap)

    def vis_roc_curve(self, output):
        PathManager.mkdirs(output)
        pos, neg = [], []
        for i, q in enumerate(self.q_pids):
            cmc, sort_idx = self.get_matched_result(i)  
            ind_pos = np.where(cmc == 1)[0]
            q_dist = self.dist[i]
            pos.extend(q_dist[sort_idx[ind_pos]])

            ind_neg = np.where(cmc == 0)[0]
            neg.extend(q_dist[sort_idx[ind_neg]])

        scores = np.hstack((pos, neg))
        labels = np.hstack((np.zeros(len(pos)), np.ones(len(neg))))

        fpr, tpr, thresholds = metrics.roc_curve(labels, scores)

        self.plot_roc_curve(fpr, tpr)
        filepath = os.path.join(output, "roc.jpg")
        plt.savefig(filepath)
        return fpr, tpr, pos, neg

    @staticmethod
    def plot_roc_curve(fpr, tpr, name='model', fig=None):
        if fig is None:
            fig = plt.figure()
            plt.semilogx(np.arange(0, 1, 0.01), np.arange(0, 1, 0.01), 'r', linestyle='--', label='Random guess')
        plt.semilogx(fpr, tpr, color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
                     label='ROC curve with {}'.format(name))
        plt.title('Receiver Operating Characteristic')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='best')
        return fig

    @staticmethod
    def plot_distribution(pos, neg, name='model', fig=None):
        if fig is None:
            fig = plt.figure()
        pos_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(pos, bins=80, alpha=0.7, density=True,
                              color=pos_color,
                              label='positive with {}'.format(name))
        mu = np.mean(pos)
        sigma = np.std(pos)
        y = norm.pdf(bins, mu, sigma) 
        plt.plot(bins, y, color=pos_color)  

        neg_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))
        n, bins, _ = plt.hist(neg, bins=80, alpha=0.5, density=True,
                              color=neg_color,
                              label='negative with {}'.format(name))
        mu = np.mean(neg)
        sigma = np.std(neg)
        y = norm.pdf(bins, mu, sigma) 
        plt.plot(bins, y, color=neg_color)  

        plt.xticks(np.arange(0, 1.5, 0.1))
        plt.title('positive and negative pairs distribution')
        plt.legend(loc='best')
        return fig

    @staticmethod
    def save_roc_info(output, fpr, tpr, pos, neg):
        results = {
            "fpr": np.asarray(fpr),
            "tpr": np.asarray(tpr),
            "pos": np.asarray(pos),
            "neg": np.asarray(neg),
        }
        with open(os.path.join(output, "roc_info.pickle"), "wb") as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_roc_info(path):
        with open(path, 'rb') as handle: res = pickle.load(handle)
        return res