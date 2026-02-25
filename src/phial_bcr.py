"""
GV20 STEAD-AI library defines all the deep learning classes for modeling the B cell receptor (BCR) repertoires.

"""

from __future__ import print_function

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as tf
import torch.optim as optim
from sklearn.metrics import matthews_corrcoef


__all__ = [
    'AA_LIST',
    'get_syn_data',
    'index_to_binary',
    'weights_init',
    'set_seed',
    'RepertoireModel',
    'TwoLayerModel',
    'EncodeLayerModel',
    'IsotypeModel',
    'IsotypeModelFast',
    'PhialBCR',
    'PhialBCR_batch',
]


AA_LIST = 'ACDEFGHIKLMNPQRSTVWY'  # The list of amino acids for forming protein sequences


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
    ys = np.zeros(num_samples, dtype=int)  # Labels

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
        ys[i] = 1.0
        kmer = positive[np.random.randint(len(positive))]
        j = np.random.randint(0, num_kmers // 2, 1)[0]
        cs[i, j] = 1.0
        for k in range(len(kmer)):
            xs[i, j, k] = aa_list.find(kmer[k])

    return xs, cs, ys, positive


def index_to_binary(xs, aa_list=AA_LIST, flat=True):
    """ Transform a index matrix into a binary matrix

        For example, if aa_list='ABCD' and flat=False:
            [0 1]     [[1 0 0 0], [0 1 0 0]]
            [3 0] --> [[0 0 0 1], [1 0 0 0]]
            [2 1]     [[0 0 1 0], [0 1 0 0]]
    """
    codes = np.eye(len(aa_list))
    dims = list(xs.shape)
    x = xs.reshape(-1)
    y = codes[x]
    if flat:  # expand the last dimension
        dims[-1] = -1
    else:  # push to the extra dimension
        dims.append(len(aa_list))
    return y.reshape(tuple(dims))


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class RepertoireModel(nn.Module):
    """ Collection of the train and test tasks for different PhialBCR models.
        Run a linear layer with the maximum pool as a test

        train_all():    train a deep learning model
        train_batch():  train a deep learning model in batches
        test():         test the model performance
        predict():      predict a new data with unknown labels
    """

    def __init__(self, model_name='RepertoireModel', save_path='', device='cpu'):
        super(RepertoireModel, self).__init__()
        self.model_name = model_name
        self.save_path = save_path
        self.device = torch.device(device)
        self = self.to(self.device)

        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        if not os.path.isdir(self.save_path + '/' + self.model_name):
            os.mkdir(self.save_path + '/' + self.model_name)
        if not os.path.isdir(self.save_path + '/' + self.model_name + '/model'):
            os.mkdir(self.save_path + '/' + self.model_name + '/model')

    def net_init(self, seqs, counts):
        """ Network initalization """
        batch_size, max_seqs, num_features = seqs.shape
        self.fc = torch.nn.Linear(num_features, 2)
        self.apply(weights_init)

    def net_update(self):  # update the network structure based on running mode
        pass

    def forward(self, seqs, counts):
        """ Network forward structure """
        batch_size, max_seqs, num_features = seqs.shape

        x = seqs.view(batch_size * max_seqs, num_features)
        i = counts.view(batch_size * max_seqs, 1)
        x = self.fc(x)
        x = torch.relu(x)
        x = torch.where(i > 0, x, torch.tensor(-0.0))
        x = x.view(batch_size, max_seqs, 2)
        x, _ = torch.max(x, dim=1)
        return x

    def objective(self, outputs, targets):
        """ Objective for training """
        return tf.cross_entropy(outputs, targets.type(torch.long))

    def evaluate(self, outputs, targets):
        """ Evaluating the performance """
        preds = torch.argmax(outputs, dim=1).type(torch.float)
        return preds.eq(targets).type(dtype=torch.float).mean()

    def train_all(self, seqs, counts, targets, epochs=500, save_step=100, learning_rate=0.01):
        self.train()
        self.net_init(seqs, counts)
        self.net_update()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        seqs = torch.tensor(seqs, dtype=torch.float).to(self.device)
        counts = torch.tensor(counts, dtype=torch.float).to(self.device)
        targets = torch.tensor(targets, dtype=torch.float).to(self.device)

        saved_epoch = self.load_model(select_epoch=epochs)
        if saved_epoch is None:
            saved_epoch = 0
        else:
            print('\nLoad saved {} model after {} epoches'.format(self.model_name, saved_epoch))
        for epoch in range(saved_epoch, epochs + 1):
            optimizer.zero_grad()
            outputs = self.forward(seqs, counts)
            obj = self.objective(outputs, targets)
            obj.backward()
            eva = self.evaluate(outputs, targets)

            if epoch > saved_epoch:  # skip the optimization of a saved model
                optimizer.step()

            if epoch % save_step == 0 or epoch == epochs:
                print('Train {} at {}: Objective: {:.4f}, Evaluate: {:.5f}'.format(self.model_name, epoch, obj, eva))
                torch.save(self.state_dict(),
                           '{}/{}/model/Epoch_{}_pytorch'.format(self.save_path, self.model_name, epoch))
        return obj.item(), eva.item()

    def train_batch(self, data_loader, epochs=50, save_step=10, save_one=False, learning_rate=0.01, lr_decay=10, lr_factor=0.5, grad_fn=None):
        self.train()
        saved_epoch = None
        has_init = False

        all_epochs = []
        objective = []
        accuracy = []
        for epoch in range(1, epochs + 1):
            if saved_epoch is not None and epoch < saved_epoch:  # directly skip saved cases
                scheduler.step()  # notify the scheduler
                continue

            total_num, total_obj, total_eva = 0, 0, 0
            print('Epoch %s: ' % epoch, end='', flush=True)
            for data in data_loader:
                print('#', end='', flush=True)
                seqs, counts, targets = data['seqs'], data['counts'], data['targets']
                batch_size = seqs.shape[0]
                seqs = seqs.type(dtype=torch.float).to(self.device)
                counts = counts.type(dtype=torch.float).to(self.device)
                targets = targets.type(dtype=torch.float).to(self.device)
                if 'clinical_features' in data:
                    clin_feat = data['clinical_features']
                    clin_feat = clin_feat.type(dtype=torch.float).to(self.device)
                if 'exps_scores' in data:
                    exps_feat = data['exps_scores']
                    exps_feat = exps_feat.type(dtype=torch.float).to(self.device)

                if not has_init:
                    self.net_init(seqs, counts)
                    self.net_update()
                    if epoch == 1 and grad_fn is not None:
                        for name, param in self.named_parameters():
                            layer_name = name.split('.')[0]
                            grad_layers = grad_fn.split('-')
                            if 'out' not in name and layer_name not in grad_layers:
                                param.requires_grad = False
                    optimizer = optim.Adam(self.parameters(), lr=learning_rate)
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay, gamma=lr_factor)
                    saved_epoch = self.load_model(select_epoch=epochs)
                    has_init = True

                if saved_epoch is not None and epoch < saved_epoch:
                    break

                optimizer.zero_grad()

                if 'clinical_features' in data:
                    outputs = self.forward(seqs, counts, clin_feat, exps_feat)
                else:
                    outputs = self.forward(seqs, counts)
                obj = self.objective(outputs, targets)

                if self.device == torch.device("cuda"):
                    eva = self.evaluate(outputs.to(torch.device("cpu")), targets.to(torch.device("cpu")))
                else:
                    eva = self.evaluate(outputs, targets)

                total_obj += obj.item() * batch_size
                total_eva += eva.item() * batch_size
                total_num += batch_size

                if saved_epoch is None or epoch > saved_epoch:  # skip the optimization of a saved model
                    obj.backward()
                    optimizer.step()
                elif epoch == saved_epoch:  # process one batch for evaluation
                    break

            scheduler.step()  # epoch +1
            print(' LR=%.2E'%scheduler.get_last_lr()[0])

            if saved_epoch is not None and epoch < saved_epoch:  # skip again
                continue

            total_obj /= total_num
            total_eva /= total_num

            if epoch % save_step == 0 or epoch == epochs:
                print('Train {} at {}: Objective: {:.4f}, Evaluate: {:.5f}'.format(self.model_name, epoch, total_obj, total_eva))
                torch.save(self.state_dict(),
                           '{}/{}/model/Epoch_{}_pytorch'.format(self.save_path, self.model_name, epoch))
                if save_one:
                    old_save = '{}/{}/model/Epoch_{}_pytorch'.format(self.save_path, self.model_name, epoch - save_step)
                    if os.path.exists(old_save):
                        os.remove(old_save)

            all_epochs.append(epoch)
            objective.append(total_obj)
            accuracy.append(total_eva)

            if self.device == torch.device("cuda"):
                del seqs
                del counts
                del targets
                torch.cuda.empty_cache()

        save_fig = '{}/{}/train_performance.pdf'.format(self.save_path, self.model_name)
        if not os.path.exists(save_fig):
            fig, ax1 = plt.subplots()

            ax2 = ax1.twinx()
            ax1.plot(all_epochs, objective, 'g-')
            ax2.plot(all_epochs, accuracy, 'b-')

            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Total loss', color='g')
            ax2.set_ylabel('Batch performance', color='b')

            plt.savefig(save_fig)

        return obj.item(), eva.item()

    def test(self, seqs, counts, targets):
        self.eval()
        with torch.no_grad():
            seqs = torch.tensor(seqs, dtype=torch.float).to(self.device)
            counts = torch.tensor(counts, dtype=torch.float).to(self.device)
            targets = torch.tensor(targets, dtype=torch.float).to(self.device)
            outputs = self.forward(seqs, counts)
            obj = self.objective(outputs, targets)
            eva = self.evaluate(outputs, targets)

        print('{}, Objective: {:.4f}, Evaluate: {:.5f}'.format(self.model_name, obj, eva))
        return obj.item(), eva.item(), outputs.numpy()

    def test_batch(self, data_loader, first_batch_only=False):
        self.eval()
        N, obj, eva, out = 0.0, 0.0, 0.0, []
        with torch.no_grad():
            for data in data_loader:
                seqs, counts, targets = data['seqs'], data['counts'], data['targets']
                if 'clinical_features' in data:
                    clin_feat = data['clinical_features']
                    clin_feat = clin_feat.type(dtype=torch.float).to(self.device)
                if 'exps_scores' in data:
                    exps_feat = data['exps_scores']
                    exps_feat = exps_feat.type(dtype=torch.float).to(self.device)

                n = seqs.shape[0]
                seqs = seqs.type(dtype=torch.float).to(self.device)
                counts = counts.type(dtype=torch.float).to(self.device)
                targets = targets.type(dtype=torch.float).to(self.device)
                if 'clinical_features' in data:
                    outputs = self.forward(seqs, counts, clin_feat, exps_feat)
                else:
                    outputs = self.forward(seqs, counts)

                obj += self.objective(outputs, targets).item() * n
                out.append(outputs.to(torch.device("cpu")))

                if self.device == torch.device("cuda"):
                    eva += self.evaluate(outputs.to(torch.device("cpu")), targets.to(torch.device("cpu"))).item() * n
                else:
                    eva += self.evaluate(outputs, targets).item() * n
                N += n
                if first_batch_only:
                    break
        obj /= N
        eva /= N

        print('{}, Objective: {:.4f}, Evaluate: {:.5f}'.format(self.model_name, obj, eva))
        return obj, eva, torch.cat(out).numpy()

    def predict(self, seqs, counts):
        self.eval()
        with torch.no_grad():
            seqs = torch.tensor(seqs, dtype=torch.float).to(self.device)
            counts = torch.tensor(counts, dtype=torch.float).to(self.device)
            outputs = self.forward(seqs, counts)
        return outputs.numpy()

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

    def load_state_dict_smart(self, para, data_loader=None):
        if data_loader is not None:  # need to init the nodes
            for data in data_loader:
                seqs, counts = data['seqs'], data['counts']
                self.net_init(seqs, counts)
                self.net_update()
        return self.load_state_dict(para, strict=False)

    def load_model(self, model_name=None, select_epoch=None, data_loader=None, return_file=False):
        if model_name is None:
            model_name = self.model_name
        files = []
        for f in os.listdir('{}/{}/model/'.format(self.save_path, model_name)):
            if f.startswith('Epoch_'):
                ep = int(f.split('_')[1])
                files.append((ep, '{}/{}/model/{}'.format(self.save_path, model_name, f)))
        files.sort()
        if len(files) == 0:
            return None
        if select_epoch is None:
            self.load_state_dict_smart(torch.load(files[-1][1], map_location=self.device), data_loader=data_loader)
            if return_file:
                return files[-1][1]
            return files[-1][0]
        for ep, f in files:
            if ep == select_epoch:
                self.load_state_dict_smart(torch.load(f, map_location=self.device), data_loader=data_loader)
                if return_file:
                    return f
                return ep
        # return the largest epoch run
        self.load_state_dict(torch.load(f, map_location=self.device))
        if return_file:
            return f
        return ep


class TwoLayerModel(RepertoireModel):
    """ A simple two-layer full-connected neural network
        num_motifs is the key parameter for controlling the model complexity
    """

    def __init__(self, num_motifs=30, num_labels=2, model_name='TwoLayerModel', *args, **kwargs):
        super(TwoLayerModel, self).__init__(model_name=model_name, *args, **kwargs)
        self.num_motifs = num_motifs
        self.num_labels = num_labels

    def net_init(self, seqs, counts):
        batch_size, max_seqs, num_features = seqs.shape
        self.fc1 = torch.nn.Linear(num_features, self.num_motifs)
        self.fc2 = torch.nn.Linear(self.num_motifs, self.num_labels)
        self.apply(weights_init)

    def forward(self, seqs, counts):
        """ Network forward structure """
        batch_size, max_seqs, num_features = seqs.shape

        x = seqs.view(batch_size * max_seqs, num_features)
        i = counts.view(batch_size * max_seqs).unsqueeze(1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = torch.where(i > 0, x, torch.tensor(0.0))
        x = x.view(batch_size, max_seqs, self.num_motifs)
        x, _ = torch.max(x, dim=1)
        x = self.fc2(x)
        return x


class EncodeLayerModel(TwoLayerModel):
    """ Add the amino acid encoding layer
        num_motifs controls the model complexity
        encode_init is the inital encoding matrix
        num_labels controls the number of labels to output
    """

    def __init__(self, encode_init=None, model_name='EncodeLayerModel', *args, **kwargs):
        super(EncodeLayerModel, self).__init__(model_name=model_name, *args, **kwargs)

        if encode_init is None:
            self.encode_init = np.random.normal(0, 1, size=(len(AA_LIST), len(AA_LIST)))
        elif type(encode_init) is int:
            self.encode_init = np.random.normal(0, 1, size=(len(AA_LIST), encode_init))
        elif type(encode_init) is tuple:
            self.encode_init = np.random.normal(0, 1, size=encode_init)
        else:
            self.encode_init = encode_init

    def net_init(self, seqs, counts):
        batch_size, max_seqs, seq_len = seqs.shape
        amino_acid, encode_size = self.encode_init.shape

        self.em = torch.nn.Embedding(amino_acid, encode_size)
        self.fc1 = torch.nn.Linear(seq_len * encode_size, self.num_motifs)
        self.fc2 = torch.nn.Linear(self.num_motifs, self.num_labels)
        self.apply(weights_init)
        self.em.weight = torch.nn.Parameter(torch.tensor(self.encode_init))

    def forward(self, seqs, counts):
        batch_size, max_seqs, seq_len = seqs.shape
        amino_acid, encode_size = self.encode_init.shape
        if len(counts.shape) == 3:  # has isotypes
            counts = counts.mean(dim=2)

        x = seqs.type(dtype=torch.LongTensor)
        x = self.em(x).type(dtype=torch.float)
        x = x.view(batch_size * max_seqs, seq_len * encode_size)
        x = self.fc1(x)
        x = torch.relu(x)
        i = counts.view(batch_size * max_seqs).unsqueeze(1)
        x = torch.where(i > 0, x, torch.tensor(0.0))
        x = x.view(batch_size, max_seqs, self.num_motifs)
        x, _ = torch.max(x, dim=1)
        x = self.fc2(x)
        return x


class IsotypeModel(EncodeLayerModel):
    """ Model the differences in Ig isotypes
    """

    def __init__(self, model_name='IsotypeModel', dropout=0.5, *args, **kwargs):
        super(IsotypeModel, self).__init__(model_name=model_name, *args, **kwargs)
        self.train_dropout = dropout

    def net_init(self, seqs, counts):
        batch_size, max_seqs, seq_len = seqs.shape
        batch_size, max_seqs, gene_num = counts.shape
        amino_acid, encode_size = self.encode_init.shape
        self.em = torch.nn.Embedding(amino_acid, encode_size)
        self.fc1 = torch.nn.Linear(seq_len * encode_size, self.num_motifs)
        self.fc2 = torch.nn.Linear(gene_num, self.num_motifs)
        self.drop1 = torch.nn.Dropout(p=self.train_dropout)
        self.drop2 = torch.nn.Dropout(p=self.train_dropout)
        self.out = torch.nn.Linear(self.num_motifs, self.num_labels)
        self.apply(weights_init)
        self.em.weight = torch.nn.Parameter(torch.tensor(self.encode_init))

    def hidden_layers(self, seqs, counts):
        batch_size, max_seqs, seq_len = seqs.shape
        batch_size, max_seqs, gene_num = counts.shape
        amino_acid, encode_size = self.encode_init.shape

        # batch_size  -> B
        # max_seqs    -> M
        # seq_len     -> S
        # gene_num    -> C
        # num_motis   -> N
        # encode_size -> E

        # encoding_layer (B, M, S) -> (B*M, S*E)
        x = seqs.type(dtype=torch.LongTensor)
        x = self.em(x).type(dtype=torch.float)
        x = x.view(batch_size * max_seqs, seq_len * encode_size)

        # kmer_layer (B*M, S*E) -> (B*M, N)
        x = self.fc1(x)
        x = torch.relu(x)

        # gene_layer (B*M, N) -> (B*M, N, C)
        x = x.view(batch_size * max_seqs, self.num_motifs, 1)
        i = counts.view(batch_size * max_seqs, 1, gene_num)
        x = x.matmul(i)

        # max_pooling layer (B, M, N, C) -> (B, N, C)
        x = x.view(batch_size, max_seqs, self.num_motifs, gene_num)
        x, _ = torch.max(x, dim=1)

        # dropout_layer
        x = self.drop1(x)

        # motif_layer (B, N, C) -> (B, N)
        w = self.fc2.weight.view(1, self.num_motifs, gene_num)
        x = (x * w).sum(dim=2)
        b = self.fc2.bias.view(1, self.num_motifs)
        x = torch.relu(x + b)

        # dropout_layer
        x = self.drop2(x)
        return x

    def forward(self, seqs, counts):
        x = self.hidden_layers(seqs, counts)
        x = self.out(x)
        return x


class IsotypeModelFast(IsotypeModel):
    """ Sparse matrix implementation for faster computation
    """

    def hidden_layers(self, seqs, counts):
        batch_size, max_seqs, seq_len = seqs.shape
        batch_size, max_seqs, gene_num = counts.shape
        amino_acid, encode_size = self.encode_init.shape

        # batch_size  -> B
        # max_seqs    -> M
        # seq_len     -> S
        # gene_num    -> C
        # num_motis   -> N
        # encode_size -> E
        # valid_size  -> V

        # get the idx of valid seqs based on counts
        x = seqs.view(batch_size * max_seqs, seq_len)
        v = counts.sum(2).view(batch_size * max_seqs)
        x = x[v > 0, :]  # save valid seqs
        valid_size, seq_len = x.shape

        # encoding_layer (V, S) -> (V, S*E)
        x = self.em(x.type(dtype=torch.LongTensor)).type(dtype=torch.float)
        x = x.view(valid_size, seq_len * encode_size)

        # kmer_layer (V, S*E) -> (V, N)
        x = self.fc1(x)
        x = torch.relu(x)

        # gene_layer (V, N) -> (V, N, C)
        x = x.view(valid_size, self.num_motifs, 1)
        i = counts.view(batch_size * max_seqs, gene_num)
        i = i[v > 0, :].view(valid_size, 1, gene_num)
        x = x.matmul(i)

        # max_pooling layer (V, N, C) -> (B, N, C)
        x_ = torch.zeros(batch_size * max_seqs, self.num_motifs, gene_num, device=x.device)
        x_[v > 0, :, :] = x
        x = x_.view(batch_size, max_seqs, self.num_motifs, gene_num)
        x, _ = torch.max(x, dim=1)

        # dropout_layer
        x = self.drop1(x)

        # motif_layer (B, N, C) -> (B, N)
        w = self.fc2.weight.view(1, self.num_motifs, gene_num)
        x = (x * w).sum(dim=2)
        b = self.fc2.bias.view(1, self.num_motifs)
        x = torch.relu(x + b)

        # dropout_layer
        x = self.drop2(x)
        return x


class PhialBCR(IsotypeModelFast):
    """ The final deep learning model for B cell receptor repertoires

        PhialBCR runs in four modes depending on the output type (defined by the num_labels parameter):

        1. Classification (e.g. cancer types): num_labels > 1
        2. Linear regression (e.g. progression-free survival): num_labels == 1
        3. Cox-PH regression (e.g. hazard ratio): num_labels == -1
        4. Multiple linear regression (e.g. target scores): num_labels < -1

        By default, negative values are survival data with censorship
    """

    def __init__(self, model_name='PhialBCR', *args, **kwargs):
        super(PhialBCR, self).__init__(model_name=model_name, *args, **kwargs)
        self.set_run_mode(num_labels=self.num_labels)
        self.load_hidden_params = None

    def set_run_mode(self, run_mode=None, num_labels=2):
        options = ['Classification', 'Linear Regression',
                   'Cox-PH Regression', 'Multiple Linear Regression']
        if run_mode is not None:
            if run_mode not in options:
                raise ValueError('run_mode ' + run_mode + ' is not in ' + str(options))
            self.run_mode = run_mode
        else:
            # infer run_mode from num_labels
            if num_labels > 1:
                self.run_mode = options[0]
            elif num_labels == 1:
                self.run_mode = options[1]
            elif num_labels == -1:
                self.run_mode = options[2]
            elif num_labels < -1:
                self.run_mode = options[3]
            else:
                raise ValueError('num_labels must > 1, == 1, == -1, or < -1')
        print(self.model_name, 'runs in', self.run_mode, 'mode.')

    def net_init(self, seqs, counts):
        batch_size, max_seqs, seq_len = seqs.shape
        batch_size, max_seqs, gene_num = counts.shape
        amino_acid, encode_size = self.encode_init.shape

        self.em = torch.nn.Embedding(amino_acid, encode_size).to(self.device)
        self.fc1 = torch.nn.Linear(seq_len * encode_size, self.num_motifs).to(self.device)
        self.fc2 = torch.nn.Linear(gene_num, self.num_motifs).to(self.device)
        self.drop1 = torch.nn.Dropout(p=0.5).to(self.device)
        self.drop2 = torch.nn.Dropout(p=0.5).to(self.device)

    def net_update(self):
        if self.load_hidden_params is not None:
            if not os.path.exists(self.load_hidden_params):
                print('Failed to load', self.load_hidden_params)
            else:
                saved_model = torch.load(self.load_hidden_params, map_location=self.device)
                del_out = [key for key in saved_model if key.startswith('out')]
                for key in del_out:
                    del saved_model[key]
                self.load_state_dict(saved_model)
                print('\nLoad parameters from {} without {}'.format(self.load_hidden_params, del_out))
        if self.run_mode == 'Classification':
            self.out = torch.nn.Linear(self.num_motifs, self.num_labels).to(self.device)
        elif self.run_mode == 'Linear Regression':
            self.out = torch.nn.Linear(self.num_motifs, 1).to(self.device)
        elif self.run_mode == 'Multiple Linear Regression':
            self.out = torch.nn.Linear(self.num_motifs, abs(self.num_labels)).to(self.device)
        elif self.run_mode == 'Cox-PH Regression':
            self.out = torch.nn.Linear(self.num_motifs, 1).to(self.device)
        else:
            raise ValueError('unknown run_mode ' + self.run_mode)

        if self.load_hidden_params is None:
            if self.device == torch.device('cuda'):
                self.encode_init = torch.tensor(self.encode_init).to(self.device)
            self.em.weight = torch.nn.Parameter(torch.tensor(self.encode_init))
            if self.device == torch.device('cuda'):
                self.em.weight = self.em.weight.to(self.device)
        else:
            weights_init(self.out)

    def forward(self, seqs, counts):
        x = self.hidden_layers(seqs, counts)
        x = self.out(x)
        return x

    @staticmethod
    def negative_log_partial_likelihood(risk, censor):
        """ Return the negative log-partial likelihood of the prediction

            Modified from DeepSurv model's implementation:
            https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py

            *Note: Need to sort data points by time first!

        Parameters:
            risk: predicted risk factors
            censor: 0 means censorship and 1 means event

        Returns:
            neg_likelihood: -log of partial Cox-PH likelihood
        """
        hazard = torch.exp(risk)
        log_risk = torch.log(torch.cumsum(hazard, dim=0))
        uncensored_likelihood = risk - log_risk
        censored_likelihood = uncensored_likelihood * censor
        neg_likelihood = - censored_likelihood.sum() / censor.sum()
        return neg_likelihood

    def objective(self, outputs, targets):
        """ Objective for training """
        if self.run_mode == 'Classification':
            return tf.cross_entropy(outputs, targets.type(torch.long))
        elif self.run_mode == 'Linear Regression':
            return tf.smooth_l1_loss(outputs, targets)
        elif self.run_mode == 'Multiple Linear Regression':
            return tf.smooth_l1_loss(outputs, targets)
        elif self.run_mode == 'Cox-PH Regression':
            time = targets[:, 0]
            censor = targets[:, 1]
            sidx = time.argsort(descending=True)
            return self.negative_log_partial_likelihood(outputs[sidx], censor[sidx])
        else:
            raise ValueError('Unknown run_mode ' + self.run_mode)

    @staticmethod
    def kl_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # D_KL(P || Q)
        return tf.kl_div(q.log(), p, reduction='batchmean')

    @staticmethod
    def js_div(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        # JSD(P || Q)
        m = 0.5 * (p + q)
        return 0.5 * (PhialBCR.kl_div(p, m) + PhialBCR.kl_div(q, m))

    @staticmethod
    def pearson_cor(x, y, dim=0):
        var = torch.mean(x * y, dim=dim) - (torch.mean(x, dim=dim) * torch.mean(y, dim=dim))
        ref = torch.sqrt((torch.mean(x ** 2, dim=dim) - torch.mean(x, dim=dim) ** 2) *
                         (torch.mean(y ** 2, dim=dim) - torch.mean(y, dim=dim) ** 2))
        return var / ref

    @staticmethod
    def get_concordance_index(risk, time, censor, **kwargs):
        """ Calculate the C-index for evaluating survival models

        Parameters:
            risk: predicted risk factors
            time: survival time
            censor: 0 means censorship and 1 means event

        Returns:
            concordance_index: calcualted using lifelines.utils.concordance_index
        """
        from lifelines.utils import concordance_index
        if censor.sum() == 0:
            print('Warning: all data points are censored. Fail to estimate the C-index.')
            return torch.tensor(0)
        partial_hazard = torch.exp(risk)
        return concordance_index(time.detach().numpy(), -partial_hazard.detach().numpy(), censor.detach().numpy())

    def evaluate(self, outputs, targets):
        """ Evaluating the performance """
        if self.run_mode == 'Classification':
            preds = torch.argmax(outputs, dim=1).type(torch.float)
            return torch.tensor(matthews_corrcoef(targets, preds.detach().numpy()))
        elif self.run_mode == 'Linear Regression':
            return self.pearson_cor(outputs, targets)[0]
        elif self.run_mode == 'Multiple Linear Regression':
            res = self.pearson_cor(outputs, targets)
            res[(res != res) | (res == float('inf')) | (res == float('-inf'))] = 0  # ignore failed cases
            return res.mean()
        elif self.run_mode == 'Cox-PH Regression':
            time = targets[:, 0]
            censor = targets[:, 1]
            if censor.sum() == 0:
                print('Warning: all data points are censored. Fail to estimate the C-index.')
                return torch.tensor(0)
            return self.get_concordance_index(outputs, time, censor)
        else:
            raise ValueError('Unknown run_mode ' + self.run_mode)


class PhialBCR_batch(PhialBCR):
    """
        With Batch Normalization Layers
    """

    def __init__(self, model_name='PhialBCR_batch', *args, **kwargs):
        super(PhialBCR_batch, self).__init__(model_name=model_name, *args, **kwargs)

    def net_init(self, seqs, counts):
        super(PhialBCR_batch, self).net_init(seqs, counts)

        batch_size, max_seqs, seq_len = seqs.shape
        amino_acid, encode_size = self.encode_init.shape

        self.bn0 = torch.nn.BatchNorm1d(seq_len * encode_size).to(self.device)
        self.bn1 = torch.nn.BatchNorm1d(self.num_motifs).to(self.device)
        self.bn2 = torch.nn.BatchNorm1d(self.num_motifs).to(self.device)

    def hidden_layers(self, seqs, counts):
        batch_size, max_seqs, seq_len = seqs.shape
        batch_size, max_seqs, gene_num = counts.shape
        amino_acid, encode_size = self.encode_init.shape

        x = seqs.view(batch_size * max_seqs, seq_len)
        v = counts.sum(2).view(batch_size * max_seqs).to(self.device)
        x = x[v > 0, :]  # save valid seqs
        valid_size, seq_len = x.shape

        x = x.type(dtype=torch.LongTensor).to(self.device)
        x = self.em(x).type(dtype=torch.float)
        x = x.view(valid_size, seq_len * encode_size).to(self.device)
        x = self.bn0(x)

        x = self.fc1(x)
        x = torch.relu(x)
        x = self.bn1(x)

        x = x.view(valid_size, self.num_motifs, 1)
        i = counts.view(batch_size * max_seqs, gene_num)
        i = i[v > 0, :].view(valid_size, 1, gene_num)
        x = x.to(self.device)
        x = x.matmul(i)

        x_ = torch.zeros(batch_size * max_seqs, self.num_motifs, gene_num).to(self.device)
        x_[v > 0, :, :] = x
        x = x_.view(batch_size, max_seqs, self.num_motifs, gene_num)
        x, _ = torch.max(x, dim=1)

        x = self.drop1(x)

        w = self.fc2.weight.view(1, self.num_motifs, gene_num).to(self.device)
        x = (x * w).sum(dim=2)
        b = self.fc2.bias.view(1, self.num_motifs).to(self.device)
        x = torch.relu(x + b)
        x = self.bn2(x)

        x = self.drop2(x)
        return x


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
    res = []

    n = 1000  # number of samples
    k = 9  # sequence length
    for m in [20]:  # number of seqs
        for i in [3]:  # number of positive seqs
            print(m, '<-->', i)
            save_path = os.path.join(workpath, 'test_m{}_i{}'.format(m, i))
            xs, cs, ys, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size=k, num_pos_kmers=i)

            print('Train data', xs.shape, cs.shape, ys.shape)
            XS, CS, YS, ref = get_syn_data(num_samples=n, num_kmers=m, kmer_size=k, num_pos_kmers=i, positive=ref)
            print('Test data', XS.shape, CS.shape, YS.shape)

            m1 = RepertoireModel(save_path=save_path)
            t1 = m1.train_all(index_to_binary(xs), cs, ys)
            r1 = m1.test(index_to_binary(XS), CS, YS)
            m1.predict(index_to_binary(XS), CS)
            res.append((n, m, i, m1.model_name, t1[0], r1[0]))

            m2 = TwoLayerModel(save_path=save_path)
            t2 = m2.train_all(index_to_binary(xs), cs, ys)
            r2 = m2.test(index_to_binary(XS), CS, YS)
            m2.predict(index_to_binary(XS), CS)
            res.append((n, m, i, m2.model_name, t2[0], r2[0]))

            m3 = EncodeLayerModel(save_path=save_path)
            t3 = m3.train_all(xs, cs, ys)
            r3 = m3.test(XS, CS, YS)
            m3.predict(XS, CS)
            res.append((n, m, i, m3.model_name, t3[0], r3[0]))

            m4 = IsotypeModelFast(save_path=save_path)
            t4 = m4.train_all(xs, np.tile(cs.reshape(n, m, 1), (1, 1, 2)), ys)
            r4 = m4.test(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)), YS)
            m4.predict(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)))
            res.append((n, m, i, m4.model_name, t4[0], r4[0]))

            m5 = PhialBCR(save_path=save_path)
            t5 = m5.train_all(xs, np.tile(cs.reshape(n, m, 1), (1, 1, 2)), ys)
            r5 = m5.test(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)), YS)
            m5.predict(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)))
            res.append((n, m, i, m5.model_name, t5[0], r5[0]))

            m6 = PhialBCR_batch(save_path=save_path)
            t6 = m6.train_all(xs, np.tile(cs.reshape(n, m, 1), (1, 1, 2)), ys)
            r6 = m6.test(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)), YS)
            m6.predict(XS, np.tile(CS.reshape(n, m, 1), (1, 1, 2)))
            res.append((n, m, i, m6.model_name, t6[0], r6[0]))

    res = pd.DataFrame(res, columns=['#Samples', '#Snips', '#PosCases', 'Model', 'Train_obj', 'Test_obj'])
    res.to_csv(os.path.join(save_path, 'result_comparison.csv'), index=False)

    FINISH_TIME = datetime.now()
    print('Results have been saved in', save_path)
    print('Start  at', START_TIME)
    print('Finish at', FINISH_TIME)
    print("Time Cost", FINISH_TIME - START_TIME)
