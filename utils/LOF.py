from scipy.spatial.distance import pdist, squareform
import numpy as np
import pickle

from datasets.cifar10 import Cifar10
from utils import metrics
from datasets.dataset_type import DatasetType
from configs.dataset_config import cfg as dataset_cfg
from utils.visualization import Visualization


class LOF(object):
    def __init__(self, cfg, train_cfg):

        self.cfg = cfg
        self.train_cfg = train_cfg
        self.visualization = Visualization(cfg)

    def get_data(self):
        dataset = Cifar10(dataset_cfg=dataset_cfg, dataset_type=DatasetType.Test)
        self.dataset_len = len(dataset)
        self.vectors = self.get_AE_features()
        self.labels = dataset.labels

    def get_AE_features(self):
        with open(self.train_cfg.vectors_path + f'test_set_vectors.pickle', 'rb') as f:
            features = pickle.load(f)
        return features

    def get_distances(self):
        if self.cfg.load_saved_dists:
            with open(self.cfg.saved_dists_path, 'rb') as f:
                square_dists = pickle.load(f)
        else:
            if self.cfg.calculate_dists_in_loop:
                square_dists = np.zeros((self.dataset_len, self.dataset_len), dtype=np.float32)
                for i, vec in enumerate(self.vectors):
                    if i % 100 == 0:
                        print(f'calculated dists for {i}/{self.dataset_len} vectors')
                    square_dists[i] = np.sort(np.linalg.norm(self.vectors - vec, 2, -1))#[1:]
                with open(self.cfg.saved_dists_path, 'wb') as f:
                    pickle.dump(square_dists, f)
            else:
                dists_vector = pdist(self.vectors, metric='euclidean')
                square_dists = squareform(dists_vector)

        self.square_dists = square_dists
        self.sorted_dists = np.sort(square_dists, 1)[:, 1:]

    def get_k_distances(self, k):
        k_dists = self.sorted_dists[:, k - 1]
        return k_dists

    def get_reachability_distances(self, k_distances, use_simple_reach_dist=False):
        if use_simple_reach_dist:
            reachability_distances = self.square_dists
        else:
            h, _ = self.square_dists.shape
            k_distances_matrix = np.tile(k_distances, h).reshape((h, h))
            reachability_distances = np.max([self.square_dists, k_distances_matrix], 0)
        return reachability_distances

    def get_lof(self, reachability_distances, k_distances, eps=1e-12):
        argsort = np.argsort(self.square_dists, 1)[:, 1:]
        N_k = []
        for i in range(len(k_distances)):
            n_k = np.where(self.sorted_dists[i] <= k_distances[i])[0]
            N_k.append(argsort[i][n_k])
        N_k = np.asarray(N_k)

        N_k_reach_dists = np.asarray([reachability_distances[j][idxs] for j, idxs in enumerate(N_k)])
        lrd = np.asarray([len(N_k[i]) / (np.sum(N_k_reach_dists[i] + eps)) for i in range(len(N_k_reach_dists))])
        lof = [np.sum(lrd[N_k[i]]) / len(N_k[i]) / lrd[i] for i in range(len(lrd))]
        assert not np.any(np.isnan(lof))

        return np.asarray(lof)

    def get_best_k_by_AP(self):
        best_ap_score = -1
        best_k_info = {}

        for k in self.cfg.k_list:
            k_distances = self.get_k_distances(k=k)
            reachability_distances = self.get_reachability_distances(k_distances)
            lof = self.get_lof(reachability_distances, k_distances)

            ids_to_sort = np.argsort(lof)[::-1]
            predictions = lof[ids_to_sort]
            labels = self.labels[ids_to_sort]

            tp = np.cumsum(labels)
            precision = tp / (np.arange(self.dataset_len) + 1)
            recall = tp / sum(labels == 1)

            ap_score = metrics.average_precision_score(precision, recall)
            print(f'k: {k}, calculated AP: {ap_score}')

            if ap_score > best_ap_score:
                best_ap_score = ap_score
                best_k_info['best_ap_score'] = ap_score
                best_k_info['best_k'] = k
                best_k_info['best_k_precision'] = precision
                best_k_info['best_k_recall'] = recall
                best_k_info['best_k_labels'] = labels
                best_k_info['best_k_scores'] = predictions
        return best_k_info

    def run(self):
        self.get_data()
        self.get_distances()

        best_k_info = self.get_best_k_by_AP()
        print(f'best k: {best_k_info["best_k"]}, AP: {best_k_info["best_ap_score"]}')

        p, r, t = metrics.precision_recall_curve(best_k_info["best_k_labels"],
                                                 best_k_info["best_k_scores"],
                                                 best_k_info['best_k_precision'],
                                                 best_k_info['best_k_recall'])
        f1_score = metrics.f1_score(p, r)
        best_f1_score_idx = np.argmax(f1_score)
        best_f1_score = f1_score[best_f1_score_idx]
        print(f'best F1-score: {best_f1_score}')

        best_thr = t[best_f1_score_idx - 1]
        fin_prediction = np.zeros(self.dataset_len)
        fin_prediction[self.get_k_distances(k=best_k_info["best_k"]) > best_thr] = 1

        conf_matrix_for_best_thr = metrics.confusion_matrix(self.labels, fin_prediction)
        print(f'Confusion matrix (tn, fp, fn, tp): {np.concatenate(conf_matrix_for_best_thr)}')

        self.visualization.plot_precision_recall_curve(p, r, best_k_info["best_k_labels"],
                                                       best_k_info["best_k_scores"], 'pr_curve')
        self.visualization.plot_conf_matrix(conf_matrix_for_best_thr, 'confusion_matrix')
