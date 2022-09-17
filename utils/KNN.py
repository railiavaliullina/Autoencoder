from scipy.spatial.distance import pdist, squareform
import numpy as np
import pickle
from sklearn import metrics as sklearn_metrics

from datasets.cifar10 import Cifar10
from utils import metrics
from datasets.dataset_type import DatasetType
from configs.dataset_config import cfg as dataset_cfg
from utils.visualization import Visualization


class KNN(object):
    def __init__(self, knn_cfg, train_cfg):

        self.cfg = knn_cfg
        self.train_cfg = train_cfg
        self.visualization = Visualization(knn_cfg)

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
                sorted_dists = pickle.load(f)
        else:
            if self.cfg.calculate_dists_in_loop:
                sorted_dists = np.zeros((self.dataset_len, self.dataset_len - 1), dtype=np.float32)
                for i, vec in enumerate(self.vectors):
                    if i % 100 == 0:
                        print(f'calculated dists for {i}/{self.dataset_len} vectors')
                    sorted_dists[i] = np.sort(np.linalg.norm(self.vectors - vec, 2, -1))[1:]
                with open(self.cfg.saved_dists_path, 'wb') as f:
                    pickle.dump(sorted_dists, f)
            else:
                dists_vector = pdist(self.vectors)
                square_dists = squareform(dists_vector)
                sorted_dists = np.sort(square_dists, 1)[:, 1:]
        return sorted_dists

    def get_anomaly_scores(self, sorted_dists, k=1):
        predictions = sorted_dists[:, k - 1]
        return predictions

    def get_best_k_by_AP(self, sorted_dists):
        best_ap_score = -1
        best_k_info = {}

        for k in self.cfg.k_list:
            anomaly_scores = self.get_anomaly_scores(sorted_dists, k=k)

            ids_to_sort = np.argsort(anomaly_scores)[::-1]
            predictions = anomaly_scores[ids_to_sort]
            labels = self.labels[ids_to_sort]

            tp = np.cumsum(labels)
            precision = tp / (np.arange(self.dataset_len) + 1)
            recall = tp / sum(labels == 1)

            ap_score = metrics.average_precision_score(precision, recall)
            print(f'k: {k}, calculated AP: {ap_score}')

            print(f'Implemented = {ap_score}')
            print(f'sklearn = {sklearn_metrics.average_precision_score(labels, predictions)}')

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
        sorted_dists = self.get_distances()

        best_k_info = self.get_best_k_by_AP(sorted_dists)
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
        fin_prediction[self.get_anomaly_scores(sorted_dists, k=best_k_info["best_k"]) > best_thr] = 1

        conf_matrix_for_best_thr = metrics.confusion_matrix(self.labels, fin_prediction)
        print(f'Confusion matrix (tn, fp, fn, tp): {np.concatenate(conf_matrix_for_best_thr)}')

        self.visualization.plot_precision_recall_curve(p, r, best_k_info["best_k_labels"],
                                                       best_k_info["best_k_scores"], 'pr_curve')
        self.visualization.plot_conf_matrix(conf_matrix_for_best_thr, 'confusion_matrix')
