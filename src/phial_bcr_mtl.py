"""
Multitask learning for classification, regression and survival prediction
"""

from __future__ import print_function

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as tf
from sklearn.metrics import matthews_corrcoef

from phial_bcr import *


def get_syn_data(num_samples=30, num_kmers=30, kmer_size=3, num_pos_kmers=3, aa_list=AA_LIST, positive=None):
    """ Helper function for generating the synthetic data

        Args:
            num_samples: number of samples in the batch
            num_kmers: maximum number of kmers in each sample
            kmer_size: the size of k-mer, i.e.: k
            num_pos_kmers: number of recurrent kmers in positive samples
            aa_list: the list of amino acids used for creating k-mers
            positive: the list of positive kmers. If None, will generate based on num_pos_kmers

        Returns:
            xs: encoded repertoires, [num_samples, num_kmers, kmer_size]
                where each amino acid is represented as itegers
            cs: indication of the missing data [num_samples, num_kmers]
            ys: labels for positive or negative samples [num_samples]
            positive: the list of positive kmers used in generating the data
    """
    aa_size = len(aa_list)

    xs = np.zeros((num_samples, num_kmers, kmer_size), dtype=int)  # Features
    cs = np.zeros((num_samples, num_kmers), dtype=int)  # Kmer count
    ys_class = np.zeros(num_samples, dtype=int)  # Labels class
    ys_regress = np.random.uniform(-1, 1, (num_samples, 51))  # Labels regress

    for i in range(num_samples):
        N = np.random.randint(num_kmers // 2) + num_kmers // 2 - 1
        for j in range(N):
            cs[i, j] = 1.0
            for k in range(kmer_size):
                xs[i, j, k] = np.random.randint(aa_size)

    if positive is None:
        positive = []
        for i in range(num_pos_kmers):
            kmer = ''
            for k in range(kmer_size):
                kmer += aa_list[np.random.randint(aa_size)]
            positive.append(kmer)
        print('Positive kmers:', positive)

    for i in range(round(num_samples / 2)):
        ys_class[i] = 1.0
        kmer = positive[np.random.randint(len(positive))]
        j = np.random.randint(0, num_kmers // 2, 1)[0]
        cs[i, j] = 1.0
        for k in range(len(kmer)):
            xs[i, j, k] = aa_list.find(kmer[k])

    ys = np.concatenate((ys_class.reshape(-1, 1), ys_regress), axis=1)

    return xs, cs, ys, positive


class PhialBCR_MTL(PhialBCR_batch):
    '''
    Multitask layers
    '''

    def __init__(self, model_name='PhialBCR_MTL', num_labels=None,
                 lambda_param=None, num_labels_class=None, num_labels_regress=None,
                 *args, **kwargs):
        super(PhialBCR_MTL, self).__init__(model_name=model_name, *args, **kwargs)
        if num_labels is None and num_labels_class is not None:
            num_labels = [num_labels_class]
            if num_labels_regress is not None:
                num_labels.append(-num_labels_regress)
        self.num_labels = num_labels
        if lambda_param is None and num_labels is not None:
            lambda_param = [1.0] * len(num_labels)
        self.lambda_param = lambda_param
        self.load_hidden_params = None

    def set_run_mode(self, run_mode=None, num_labels=2):
        self.run_mode = 'Multitask'
        print(self.model_name, 'runs in', self.run_mode, 'mode.')

    def net_init(self, seqs, counts):
        super(PhialBCR_MTL, self).net_init(seqs, counts)

    def net_update(self):
        self.out = nn.ModuleList()
        for label in self.num_labels:
            if label > 1:
                self.out.append(torch.nn.Linear(self.num_motifs, label).to(self.device))
            elif label == 1 or label == -1:
                self.out.append(torch.nn.Linear(self.num_motifs, 1).to(self.device))
            elif label < -1:
                self.out.append(torch.nn.Linear(self.num_motifs, abs(label)).to(self.device))
            else:
                raise ValueError('num_labels must > 1, == 1, == -1, or < -1')

    def objective(self, outputs, targets):

        if self.device == torch.device("cuda"):
            outputs = outputs.to(torch.device("cpu"))
            targets = targets.to(torch.device("cpu"))

        obj = 0.0
        idx_pred, idx_target = 0, 0

        for i, label_idx in enumerate(self.num_labels):

            # Compute the loss for the current task
            if label_idx > 1:  # classification
                pred = outputs[:, idx_pred:idx_pred + label_idx]
                target = torch.flatten(targets[:, idx_target:idx_target + 1])
                valid_idx = torch.logical_not(torch.isnan(target))

                sub_obj = tf.cross_entropy(pred[valid_idx], target.type(torch.long)[valid_idx])

                idx_pred += label_idx
                idx_target += 1

            elif label_idx == 1:  # regression
                pred = torch.flatten(outputs[:, idx_pred:idx_pred + label_idx])
                target = torch.flatten(targets[:, idx_target:idx_target + label_idx])
                valid_idx = torch.logical_not(torch.isnan(target))

                sub_obj = tf.smooth_l1_loss(pred[valid_idx], target[valid_idx].float())

                idx_pred += 1
                idx_target += label_idx

            elif label_idx < -1:  # regression
                pred = outputs[:, idx_pred:idx_pred + abs(label_idx)]
                target = targets[:, idx_target:idx_target + abs(label_idx)]
                valid_idx = torch.logical_not(torch.any(torch.isnan(target), dim=1))

                sub_obj = tf.smooth_l1_loss(pred[valid_idx,:], target[valid_idx,:].float())

                idx_pred += abs(label_idx)
                idx_target += abs(label_idx)

            elif label_idx == -1:  # survival
                pred = outputs[:, idx_pred]
                target = targets[:, idx_target]

                valid_idx = torch.logical_not(torch.isnan(target))
                pred = pred[valid_idx]
                target = target[valid_idx]

                time = torch.abs(target)
                censor = (target > 0).type(torch.float)
                sidx = time.argsort(descending=True)
                sub_obj = self.negative_log_partial_likelihood(pred[sidx], censor[sidx])

                idx_pred += 1
                idx_target += 1

            sub_obj[sub_obj != sub_obj] = 0  # NA to 0
            obj += self.lambda_param[i] * sub_obj

        return obj

    def evaluate(self, outputs, targets, mode=None):
        """ Evaluating the performance """

        if self.device == torch.device("cuda"):
            outputs = outputs.to(torch.device("cpu"))
            targets = targets.to(torch.device("cpu"))

        eval_out = []
        idx_pred, idx_target = 0, 0
        for i, label_idx in enumerate(self.num_labels):

            if label_idx > 1:
                pred = outputs[:, idx_pred:idx_pred + label_idx]
                pred = torch.flatten(torch.argmax(pred, dim=1).type(torch.float))
                target = torch.flatten(targets[:, idx_target:idx_target + 1])
                valid_idx = torch.logical_not(torch.isnan(target))

                eval_out.append(torch.tensor(matthews_corrcoef(target.type(torch.long)[valid_idx], pred[valid_idx].detach().numpy())))

                idx_pred += label_idx
                idx_target += 1

            elif label_idx == 1:
                pred = torch.flatten(outputs[:, idx_pred:idx_pred + label_idx])
                target = torch.flatten(targets[:, idx_target:idx_target + label_idx])
                valid_idx = torch.logical_not(torch.isnan(target))

                eval_out.append(self.pearson_cor(pred[valid_idx].double(), target[valid_idx].double())[0])

                idx_pred += 1
                idx_target += label_idx

            elif label_idx < -1:
                pred = outputs[:, idx_pred:idx_pred + abs(label_idx)]
                target = targets[:, idx_target:idx_target + abs(label_idx)]
                valid_idx = torch.logical_not(torch.any(torch.isnan(target), dim=1))

                res = self.pearson_cor(pred[valid_idx,:].double(), target[valid_idx,:].double())
                res[(res != res) | (res == float('inf')) | (res == float('-inf'))] = 0  # ignore failed cases
                eval_out.append(res.mean())

                idx_pred += abs(label_idx)
                idx_target += abs(label_idx)

            elif label_idx == -1:
                pred = outputs[:, idx_pred:idx_pred + 1]
                target = targets[:, idx_target:idx_target + 1]
                
                valid_idx = torch.logical_not(torch.isnan(target))
                pred = pred[valid_idx]
                target = target[valid_idx]

                time = torch.abs(target)
                censor = (target > 0).type(torch.float)
                if censor.sum() == 0:
                    eval_out.append(torch.tensor(0))
                else:
                    eval_out.append(self.get_concordance_index(pred, time, censor))
                idx_pred += 1
                idx_target += 1

        if mode is not None:
            return eval_out
        return torch.sum(torch.tensor(eval_out))

    def test_batch(self, data_loader, first_batch_only=False):
        self.eval()

        N, obj, eva = 0.0, 0.0, [0.0] * len(self.num_labels)
        out = {}
        for i in range(len(self.num_labels)):
            out[i] = []
        with torch.no_grad():
            for data in data_loader:
                idx_pred = 0
                seqs, counts, targets = data['seqs'], data['counts'], data['targets']
                n = seqs.shape[0]
                seqs = seqs.type(dtype=torch.float).to(self.device)
                counts = counts.type(dtype=torch.float).to(self.device)
                outputs = self.forward(seqs, counts)
                obj_val = self.objective(outputs, targets).item()
                eva_val = self.evaluate(outputs, targets, mode='eval')

                obj += float(obj_val) * n
                for i, val in enumerate(eva_val):
                    eva[i] += val * n

                for i, label_idx in enumerate(self.num_labels):
                    if label_idx > 1 or label_idx < -1:
                        pred = outputs[:, idx_pred:idx_pred + abs(label_idx)]
                        idx_pred += abs(label_idx)
                    elif label_idx == 1 or label_idx == -1:
                        pred = outputs[:, idx_pred:idx_pred + 1]
                        idx_pred += 1
                    out[i].append(pred)

                N += n
                if first_batch_only:
                    break
        obj /= N
        for i, val in enumerate(eva):
            val /= N
            eva[i] = round(val.item(), 3)
        for i in range(len(out)):
            val = out[i]
            out[i] = torch.cat(val).numpy()
        print('Test {}: Objective: {:.4f}, Evaluate: {}'.format(self.model_name, obj, eva))
        return obj, eva, out

    def predict_batch(self, data_loader, select_col=None, append_rank=False):
        self.eval()
        out = []
        with torch.no_grad():
            for data in data_loader:
                print('#', end='', flush=True)
                seqs, counts = data['seqs'], data['counts']
                seqs = seqs.type(dtype=torch.float).to(self.device)
                counts = counts.type(dtype=torch.float).to(self.device)
                outputs = self.forward(seqs, counts).numpy()

                # only output regression result. TODO: generalize
                outputs = outputs[:, self.num_labels[0]:]

                if append_rank:
                    order = outputs.argsort(axis=1)
                    ranks = order.argsort(axis=1)
                    if select_col is not None:
                        ranks = ranks[:, select_col]
                if select_col is not None:
                    outputs = outputs[:, select_col]
                if append_rank:
                    out.append(np.hstack([outputs, ranks]))
                else:
                    out.append(outputs)
            print('')
        return np.vstack(out)

    def forward(self, seqs, counts):
        x = self.hidden_layers(seqs, counts)
        x_out = []
        for out_layer in self.out:
            x_out.append(out_layer(x))
        x = torch.cat(x_out, 1)
        return x


def test(workpath):
    res = []

    n = 1000  # number of samples
    k = 9  # sequence length
    for m in [20]:  # number of seqs
        for i in [3]:  # number of positive seqs
            print(m, '<-->', i)
            save_path = os.path.join(workpath, 'test_m{}_i{}'.format(m, i))
            xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size=k, num_pos_kmers=i)

            print('Train data', xs.shape, cs.shape, ys[0].shape, ys[1].shape)
            XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size=k, num_pos_kmers=i, positive=ref)
            print('Test data', XS.shape, CS.shape, YS[0].shape, YS[1].shape)

            model = PhialBCR_MTL(num_labels_class=2, num_labels_regress=51, save_path=save_path)
            t = model.train_all(xs, np.tile(cs.reshape(n, m, 1), (1, 1, 2)), ys)
            r = model.test(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)), YS)
            model.predict(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)))
            res.append((n, m, i, model.model_name, t[0], r[0]))

    res = pd.DataFrame(res, columns=['#Samples', '#Snips', '#PosCases', 'Model', 'Train_obj', 'Test_obj'])
    res.to_csv(os.path.join(save_path, 'result_comparison.csv'), index=False)
    print('Results have been saved in', save_path)


if __name__ == '__main__':
    set_seed(0)
    print('Run module tests...')

    import optparse

    parser = optparse.OptionParser()
    parser.add_option("--workpath", type=str, default='../work')

    args, _ = parser.parse_args()
    workpath = os.path.abspath(args.workpath)
    if not os.path.exists(workpath):
        os.mkdir(workpath)

    from datetime import datetime

    START_TIME = datetime.now()
    test(workpath)
    FINISH_TIME = datetime.now()
    print('Start  at', START_TIME)
    print('Finish at', FINISH_TIME)
    print("Time Cost", FINISH_TIME - START_TIME)
