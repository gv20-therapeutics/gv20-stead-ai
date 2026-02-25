"""
This tcga_bcr library defines the helper functions for processing TCGA datasets.
"""

from __future__ import print_function

import pandas as pd

pd.options.mode.chained_assignment = None  # suppress the slice data copy warning

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, WeightedRandomSampler
from torchvision import transforms

from phial_bcr import *

__all__ = [
    'evaluate_classifier',
    'colwise_pearson',
    'colwise_spearman',
    'evaluate_survival',
    'discrete_neg_log_likelihood',
    'neg_partial_log_likelihood',
    'c_idx',
    'evaluate_survival_meta',
    'CodonTable',
    'translate',
    'read_fa_seq',
    'read_clinical',
    'read_clinical_meta',
    'load_data',
    'AA_LIST',
    'AA_CHEM',
    'to_list',
    'pid2label',
    'pid2survival',
    'pid2survival_meta',
    'read_scores',
    'RepertoireDataset',
    'TCGA_TRUST4',
    'MapPID',
    'count_kmer',
    'summary_count',
    'Rep2Kmer',
    'encode_aa_index',
    'encode_aa_onehot',
    'encode_aa_atchley',
    'fill_zero_counts',
    'EncodeKmer',
    'make_tumor_weights',
    'get_tumor_type',
    'get_label_map',
    'get_test_environment',
]

###############################################################################
## Helper Functions for Basic Stats

from scipy.special import softmax
from sklearn.metrics import log_loss, matthews_corrcoef
from lifelines.utils import concordance_index
import scipy.stats


def evaluate_classifier(y_true, y_pred, class_labels=None):
    y_true = np.array(y_true.reshape(-1), dtype=int)
    if class_labels is not None:
        loss = log_loss(y_true, softmax(y_pred, axis=1), labels=np.arange(0, class_labels))
    else:
        loss = log_loss(y_true, softmax(y_pred, axis=1))
    y_p = np.argmax(y_pred, axis=1)
    mcc = matthews_corrcoef(y_true, y_p)
    return loss, mcc


def colwise_pearson(real, pred):
    pearsonr = np.vectorize(scipy.stats.pearsonr, signature='(n),(n)->(),()')
    pcc, pval = pearsonr(real.T, pred.T)
    return pcc, pval

def colwise_spearman(real, pred):
    spearmanr = np.vectorize(scipy.stats.spearmanr, signature='(n),(n)->(),()')
    pcc, pval = spearmanr(real.T, pred.T)
    return pcc, pval


def evaluate_survival(real, pred):
    real = real.reshape(-1)
    time = np.abs(real)
    censor = np.array(real > 0, dtype=float)
    sidx = np.argsort(-time)
    time = time[sidx]
    censor = censor[sidx]
    pred = pred[sidx]
    # C-index
    hazard = np.exp(pred)
    cidx = concordance_index(time, -hazard, censor)
    # Loss
    log_risk = np.log(np.cumsum(hazard))
    uncensored_likelihood = pred - log_risk
    censored_likelihood = uncensored_likelihood * censor
    loss = - censored_likelihood.sum() / censor.sum()
    return loss, cidx


def _convert_labels(time, event, breaks):
    """Convert event and time labels to label array.

    Each patient label array has dimensions number of intervals x 2:
        * First half is 1 if patient survived interval, 0 if not.
        * Second half is for non-censored times and is 1 for time interval
        in which event time falls and 0 for other intervals.
    """
    n_intervals = len(breaks) - 1
    timegap = breaks[1:] - breaks[:-1]
    breaks_midpoint = breaks[:-1] + 0.5 * timegap

    out = torch.zeros(len(time), n_intervals * 2)

    for i, (t, e) in enumerate(zip(time, event)):

        y = t / 365  # compute the time in years (float)
        int_y = int(y)  # compute the time in years (int)

        if e:  # if not censored

            # survived time intervals
            out[i, 0:int_y + 1] = 1.0

            # if time is greater than end of last time interval, no
            # interval is marked
            if t < breaks[-1]:
                out[i, n_intervals + int_y] = 1

        else:  # if censored

            # if lived more than half-way through interval, give credit for
            # surviving the interval
            out[i, 0:n_intervals] = 1.0 * (t >= breaks_midpoint)

        if int_y < n_intervals:
            out[i, int_y] = y - int_y

    return out


def discrete_neg_log_likelihood(risk, label, n_intervals):
    """

    Args:
        risk: Yearwise risk associated with each patient (N_intervals)
        label: Time and event label for each patient
        n_intervals: Number of intervals (Years)

    Returns:
        Negative log partial likelihood (Discrete time)

    """
    all_patients = 1. - label[:, 0:n_intervals] * risk
    noncensored = 1. + label[:, n_intervals:2 * n_intervals] * (risk - 1.)

    neg_log_like = -torch.log(torch.clamp(torch.cat((all_patients, noncensored), dim=1), 1e-07, None))

    return neg_log_like.mean().item()


def neg_partial_log_likelihood(targets, outputs):
    time = torch.abs(targets[:, 0])
    sidx = time.argsort(descending=True)
    pred = outputs[sidx]
    event = targets[:, 1][sidx]
    hazard = torch.exp(pred)
    log_risk = torch.log(torch.cumsum(hazard, dim=0))
    uncensored_likelihood = pred - log_risk
    censored_likelihood = uncensored_likelihood * event
    loss = -censored_likelihood.sum() / event.sum()

    return loss


def c_idx(targets, outputs):
    time = torch.abs(targets[:, 0])
    censor = (targets[:, 1]).type(torch.float)
    if censor.sum() == 0:
        print('Warning: all data points are censored. Fail to estimate the C-index.')
        return torch.tensor(0)
    partial_hazard = torch.exp(outputs)
    return concordance_index(time.detach().numpy(), -partial_hazard.detach().numpy(), censor.detach().numpy())


def evaluate_survival_meta(targets, outputs, interval_list):
    targets = torch.tensor(targets, dtype=torch.float)
    outputs = torch.tensor(outputs, dtype=torch.float)

    if interval_list is None:
        # Compute negative log partial log likelihood and c-index
        loss = neg_partial_log_likelihood(targets, outputs)
        c_index = torch.tensor(c_idx(targets, outputs))
    else:
        # Compute discrete time loss and c-index
        label_array = _convert_labels(torch.abs(targets[:, 0]), targets[:, 1], interval_list)
        loss = discrete_neg_log_likelihood(outputs, label_array, len(interval_list) - 1)

        probs_by_interval = outputs.permute(1, 0).detach().numpy()
        c_index = torch.tensor([concordance_index(event_times=torch.abs(targets[:, 0]),
                                                  predicted_scores=-interval_probs,
                                                  event_observed=targets[:, 1])
                                for interval_probs in probs_by_interval])
    return loss, c_index


CodonTable = {'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
              'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
              'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
              'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
              'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
              'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
              'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
              'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
              'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
              'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
              'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
              'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
              'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
              'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
              'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
              'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G'}


def translate(dna):
    aa = ''
    if type(dna) is not str:
        return aa
    for i in range(len(dna) // 3):
        code = dna[i * 3: i * 3 + 3]
        if 'N' in code or '-' in code:
            aa += '*'
        else:
            aa += CodonTable[code]
    return aa


def read_fa_seq(fa_file):
    """ Read fastq file and yield sequences"""
    with open(fa_file) as tmp:
        s = ''
        for line in tmp:
            if line.startswith('>'):
                yield s
            else:
                s += line.strip()
        yield s


def read_clinical(clinical_file='../data/clinical_data.csv'):
    """ Return the clinical information indexed by patient ids"""
    meta = pd.read_csv(clinical_file)
    print('Read', meta.shape[0], 'clinical data and columns are', list(meta))
    return meta.set_index('Uniq_Sample_id').T.to_dict()


def read_clinical_meta(clinical_file='../data/clinical_data.csv'):
    """ Return the encoded clinical information indexed by patient ids"""
    meta = pd.read_csv(clinical_file)

    # Categorical transformation of the clinical data
    meta.Patient_gender = pd.Categorical(meta.Patient_gender)
    meta['Patient_gender_encode'] = meta.Patient_gender.cat.codes.replace(-1, np.nan)
    meta.Disease_stage = pd.Categorical(meta.Disease_stage)
    meta['Disease_stage_encode'] = meta.Disease_stage.cat.codes.replace(-1, np.nan)
    meta.Disease_name = pd.Categorical(meta.Disease_name)
    meta['Disease_name_encode'] = meta.Disease_name.cat.codes.replace(-1, np.nan)

    meta['Patient_age_norm'] = meta['Patient_age'] / meta['Patient_age'].mean()
    print(list(meta))
    return meta.set_index('Uniq_Patient_id').T.to_dict()


def load_data(case, datapath='../data/', rawdata='../data/'):
    prefix = os.path.join(datapath, case)
    info = prefix + '_info.csv'
    meta = prefix + '_meta.csv'

    if os.path.exists(info) and os.path.exists(meta):
        return pd.read_csv(info, index_col=0), pd.read_csv(meta, index_col=0), prefix

    if case == 'MDS7b': # start the new data loader; older cases may not work
        data = pd.read_csv(os.path.join(rawdata, 'mds7-v20220716/MDS.20220716.bcr.csv.gz')).rename(columns={'Sample':'Uniq_Sample_id'})
        data['Uniq_Sample_id'] = data['Uniq_Sample_id'].ffill().astype('category')
        data_meta = pd.read_csv(os.path.join(rawdata, 'mds7-v20220716/file_meta.csv'))
        ratios = [0.8, 0.1, 0.1]
        min_seqs_for_sample = 100
        min_samples_for_label = 100
        seq_features = ['CDR1_aa','CDR2_aa','CDR3_aa','Cgene']
    else:
        raise ValueError('Unknown dataset name: ' + case)

    # Select the right subset to work on
    smp_count = data.Uniq_Sample_id.value_counts()
    samples = smp_count[smp_count >= min_seqs_for_sample].index

    data_meta = data_meta[data_meta.Uniq_Sample_id.isin(samples)].drop_duplicates(subset='Uniq_Sample_id', keep='first')
    data_meta = data_meta[~data_meta.Sample_type.str.contains('normal', case=False)]
    data_meta = data_meta[~data_meta.Sample_type.str.contains('cell line', case=False)]
    data_meta['Label'] = get_tumor_type(data_meta)

    lab_count = data_meta.Label.value_counts()
    labels = lab_count[lab_count >= min_samples_for_label].index
    data_meta = data_meta[data_meta.Label.isin(labels)]

    # Randomly split the data set by patient ids and save to files
    upids = data_meta.Uniq_Patient_id.unique()
    N = len(upids)
    print('We use', data_meta.shape[0], 'samples from', N, 'patients')
    random = np.random.RandomState(seed=0)
    random.shuffle(upids)

    last_ratio = 0
    data_meta['DL_set'] = None
    for s, r in enumerate(ratios):
        left_idx = int(N * last_ratio)
        last_ratio += r
        right_idx = int(N * last_ratio)
        if s == len(ratios) - 1:
            right_idx = N  # save all the remaining

        meta_idx = data_meta.Uniq_Patient_id.isin(upids[left_idx:right_idx])
        data_meta.loc[meta_idx, 'DL_set'] = 'set%s'%(s+1)
        sub_data = data[data.Uniq_Sample_id.isin(data_meta.Uniq_Sample_id[meta_idx])]
        sub_data = sub_data.groupby('Uniq_Sample_id', sort=False, observed=True)[seq_features].apply(lambda df: [tuple(x) for x in df.values])
        sub_data.to_pickle('{}_set{}.pkl.gz'.format(prefix, s+1), protocol=4)

    data_meta = data_meta[['Uniq_Sample_id','Uniq_Patient_id','Label','DL_set']].drop_duplicates().set_index('Uniq_Sample_id')
    data_meta.to_csv(meta)
    meta_cc = data_meta.groupby(['Label', 'DL_set']).size().reset_index(name='count')
    data_info = pd.pivot_table(meta_cc, index='Label', columns='DL_set', values='count')
    data_info['total'] = data_info.sum(1)
    data_info.to_csv(info)
    return data_info, data_meta, prefix


###############################################################################
## Global variable for basic animo acid propoties

AA_LIST = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '*']
AA_CHEM = {'A': [-0.591, -1.302, -0.733, 1.57, -0.146],
           'C': [-1.343, 0.465, -0.862, -1.02, -0.255],
           'D': [1.05, 0.302, -3.656, -0.259, -3.242],
           'E': [1.357, -1.453, 1.477, 0.113, -0.837],
           'F': [-1.006, -0.59, 1.891, -0.397, 0.412],
           'G': [-0.384, 1.652, 1.33, 1.045, 2.064],
           'H': [0.336, -0.417, -1.673, -1.474, -0.078],
           'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
           'K': [1.831, -0.561, 0.533, -0.277, 1.648],
           'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
           'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
           'N': [0.945, 0.828, 1.299, -0.169, 0.933],
           'P': [0.189, 2.081, -1.628, 0.421, -1.392],
           'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
           'R': [1.538, -0.055, 1.502, 0.44, 2.897],
           'S': [-0.228, 1.399, -4.76, 0.67, -2.647],
           'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
           'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
           'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
           'Y': [0.26, 0.83, 3.097, -0.838, 1.512],
           '*': [0, 0, 0, 0, 0]}


###############################################################################
## Helper Functions for Preparing the Input Data for Further Formatting

def to_list(df):
    if len(df.shape) == 1:
        return df.tolist()
    else:
        return [tuple(x) for x in df.values]



###############################################################################
## A Set of Functions for Mapping Patient Id to Clinical Annotations

def pid2label(pid, meta, use_subtype=True):
    """ Map patient id to tumor type """
    prefix = ''
    if pid.startswith('Normal_'):
        prefix = 'Normal of '
        pid = pid.replace('Normal_', '')
    if pid not in meta.index:  ## no entry
        return 'Unknown Sample'
    cancer = meta.loc[pid, 'Label']
    if pd.isnull(cancer):  ## no record
        return 'Unknown Cancer'
    return prefix + cancer


def pid2survival(pid, meta, use_log_ratio=True):
    """ Map patient id to overall survival """
    if pid not in meta:
        return None
    os = meta[pid].get('OS_days', np.nan)
    event = meta[pid].get('OS_event', np.nan)
    if pd.isnull(os) or pd.isnull(event):
        return None
    os = float(os)
    event = int(event)  ## 1 means death and 0 means living
    if use_log_ratio:
        ## Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394368/
        os = np.log2(os + 1)
    os = max(os, 1e-7)  ## ensure time > 0 so sign-based event encoding is unambiguous
    return os if event == 1 else -os


def pid2survival_meta(pid, meta, clin, exps, cancer_types, use_log_ratio=False):
    """ Map patient id to overall survival and target scores """
    if pid not in meta or meta[pid].get('Label') not in cancer_types:
        return None
    if pid not in clin:
        return None
    os = clin[pid].get('OS_days', np.nan)
    event = clin[pid].get('OS_event', np.nan)
    if pd.isnull(os) or pd.isnull(event):
        return None
    os = float(os)
    event = int(event)  # 1 means death and 0 means living
    if use_log_ratio:
        # Reference: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2394368/
        os = np.log2(os + 1)

    clinical_features = np.concatenate([meta[pid].get('Patient_age_norm'),
                                        meta[pid].get('Patient_gender_encode'),
                                        meta[pid].get('Disease_stage_encode'),
                                        meta[pid].get('Disease_name_encode')], axis=None)

    exps_features = exps.loc[pid].values
    return np.array([os, event, cancer_types.index(meta[pid].get('Disease_name'))]), \
           clinical_features, exps_features


###############################################################################
# A Set of Functions for Getting the Data for Mapping Patient Ids

def read_scores(score_file=None, genes=None, out_targets=False):
    """ Read the scoring matrix """
    if score_file is None:
        score_file = '../data/TCGA/target_zscore.csv.gz'  # default table file
    if out_targets:
        tab = pd.read_csv(score_file, index_col=0, nrows=3)
    elif genes is not None:
        try:
            tab = pd.read_csv(score_file, index_col=0, usecols=['Uniq_Sample_id'] + genes)
        except ValueError:
            try:
                tab = pd.read_csv(score_file, index_col=0, usecols=['Patient_id'] + genes)
            except ValueError:
                tab = pd.read_csv(score_file, index_col=0, usecols=['Unnamed: 0'] + genes)
    else:
        tab = pd.read_csv(score_file, index_col=0)
    if genes is None:
        genes = tab.columns.values.tolist()
    if out_targets:
        return genes
    print('Get scores from', len(tab.columns), 'targets, such as', tab.columns[0])
    return tab[genes]  # re-order


###############################################################################

class RepertoireDataset(Dataset):
    """ Load general B cell receptor repertoire dataset. """

    def __init__(self, data, transform=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data = data.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = {'idx': idx, 'seqs': self.data[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class TCGA_TRUST4(RepertoireDataset):
    """
    Load the TRUST4 BCR outputs of the TCGA dataset.
    data: CDR sequence data
    map_id: Get the labels & features
    transform: transform to be applied to a sample
    survival: if survival prediction, return target & clinical features
    """

    def __init__(self, data, map_id, transform=None, survival=False):
        self.data = data
        self.map_id = map_id
        self.transform = transform
        self.survival = survival

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pid = self.data.index[idx]
        if self.survival:
            label, clinical_features, exps_scores = self.map_id(pid)
            sample = {'pid': pid, 'targets': label, 'seqs': self.data[idx], 'clinical_features': clinical_features,
                      'exps_scores': exps_scores}
        else:
            label = self.map_id(pid)
            sample = {'pid': pid, 'targets': label, 'seqs': self.data[idx]}
        if self.transform:
            sample = self.transform(sample)
        return sample


class MapPID(object):
    def __init__(self, to='label', labels=None, meta=None, exps=None, clin=None, task_list=None, cancer_types=None):
        self.to = to
        self.labels = labels
        self.meta = meta
        self.exps = exps
        self.clin = clin
        self.task_list = task_list
        self.cancer_types = cancer_types

    def __call__(self, pid):
        if self.to == 'label':
            label = pid2label(pid=pid, meta=self.meta)
            try:
                label_idx = self.labels.index(label)
            except ValueError:
                print('{} from {} is not in {}'.format(label, pid, self.labels))
                return None
            return label_idx
        elif self.to == 'exps':
            try:
                return self.exps.loc[pid].values
            except KeyError:
                print('{} is not in the expression data'.format(pid))
                return None
        elif self.to == 'survival':
            try:
                return pid2survival(pid, self.clin)
            except KeyError:
                print('{} is not in the clinical data'.format(pid))
                return None
        elif self.to == 'survival_meta':
            try:
                return pid2survival_meta(pid, self.meta, self.clin, self.exps, self.cancer_types)
            except KeyError:
                print('{} is not in the clinical data'.format(pid))
                return None
        elif self.to == 'multitask':
            label = pid2label(pid=pid, meta=self.meta)
            labels_concat = []

            for task in self.task_list:
                if task == 'tumors':
                    try:
                        label_idx = self.labels.index(label)
                    except ValueError:
                        label_idx = np.nan
                    labels_concat.append(np.array([label_idx]))

                elif task == 'targets':
                    try:
                        exps_val = self.exps.loc[pid].values
                    except KeyError:
                        exps_val = np.repeat(np.nan, self.exps.shape[1])
                    labels_concat.append(exps_val)

                elif task == 'risks':
                    clin_val = pid2survival(pid, self.clin)
                    clin_val = np.nan if clin_val is None else clin_val
                    labels_concat.append(np.array([clin_val]))

                else:
                    raise ValueError('Unknown task '+task)

            return np.concatenate(labels_concat, axis=None)
        else:
            raise ValueError('Unknown case ' + self.to)


# @jit
def count_kmer(rep, k=6, max_pool=None, k1=3, k2=3):
    info = {}
    for s in rep:
        if type(s) == float:  # invalid
            continue
        elif len(s) == 1:  # CDR3_aa
            s = s[0]
            for i in range(len(s) - k + 1):
                mer = s[i:(i + k)]
                if mer in info:
                    info[mer] += 1
                else:
                    info[mer] = 1
        elif len(s) == 2:  # CDR3_aa, Cgene
            seq, gene = s
            gene = gene.split('*')[0]
            if type(seq) == float:  # must have cdr3
                continue
            if len(seq) < k:
                seq += ''.join(['*'] * (k - len(seq)))  # padding
            for i in range(len(seq) - k + 1):
                mer = seq[i:(i + k)]
                cc = info.get(mer, {})
                cc[gene] = cc.get(gene, 0) + 1
                info[mer] = cc
        elif len(s) == 4:  # CDR1_aa, CDR2_aa, CDR3_aa, Cgene
            cdr1, cdr2, cdr3, gene = s
            if type(cdr3) == float:  # must have cdr3
                continue
            if type(cdr1) == float:
                cdr1 = ''
            if type(cdr2) == float:
                cdr2 = ''
            if len(cdr1) < k1:
                cdr1 += ''.join(['*'] * (k1 - len(cdr1)))  # padding
            if len(cdr2) < k2:
                cdr2 += ''.join(['*'] * (k2 - len(cdr2)))  # padding
            if len(cdr3) < k:
                cdr3 += ''.join(['*'] * (k - len(cdr3)))  # padding
            gene = gene.split('*')[0]
            for i1 in range(len(cdr1) - k1 + 1):
                for i2 in range(len(cdr2) - k2 + 1):
                    for i in range(len(cdr3) - k + 1):
                        mer = cdr1[i1:(i1 + k1)] + cdr2[i2:(i2 + k2)] + cdr3[i:(i + k)]
                        cc = info.get(mer, {})
                        cc[gene] = cc.get(gene, 0) + 1
                        info[mer] = cc
        else:
            raise RuntimeError('Unknown format for '+str(s))
        if max_pool is not None and len(info) > max_pool:
            break
    return info


def summary_count(seqs, info, gmap, new):
    if type(info[seqs[0]]) == int:
        return np.array([info[s] for s in seqs])
    cc = np.zeros((len(seqs), len(new)))
    for i, mer in enumerate(seqs):
        counts = info[mer]
        for g in counts:
            if type(g) == str and g in gmap:
                iso = gmap[g]
            else:
                iso = 'Others'
            cc[i, new.index(iso)] += counts[g]
    return cc


class Rep2Kmer(object):
    """ Break the repertoire into k-mers """

    def __init__(self, kmer=3, max_pool_size=None):
        self.kmer = kmer
        gmap = {'IGHM': 'IGHM|IGHD',
                'IGHD': 'IGHM|IGHD',
                'IGHG1': 'IGHG1',
                'IGHG2': 'IGHG2/4',
                'IGHG3': 'IGHG3',
                'IGHG4': 'IGHG2/4',
                'IGHA1': 'IGHA1/2',
                'IGHA2': 'IGHA1/2',
                'IGK': 'IGK',
                'IGL': 'IGL'}
        new = sorted(set(gmap.values())) + ['Others']
        self.gmap = gmap
        self.new = new
        self.max_pool_size = max_pool_size

    def __call__(self, sample, seq_ratio=2):
        seqs = sample['seqs']  # clones
        if self.max_pool_size is not None:
            rnd_idx = np.random.permutation(len(seqs))
            seqs = [seqs[i] for i in rnd_idx]

        if type(self.kmer) is list and len(self.kmer) == 3:
            info = count_kmer(seqs, k1=self.kmer[0], k2=self.kmer[1], k=self.kmer[2],
                              max_pool=seq_ratio * self.max_pool_size)
        else:
            info = count_kmer(seqs, k=self.kmer, max_pool=seq_ratio * self.max_pool_size)

        seqs = list(info.keys())  # k-mers
        if len(seqs) == 0:
            print('kmers are', self.kmer)
            raise RuntimeError('No enough k-mers! Please decrease the --kmer-size parameter.')

        if self.max_pool_size is not None:
            rnd_idx = np.random.permutation(len(seqs))
            rnd_idx = rnd_idx[:self.max_pool_size]  # double check
            seqs = [seqs[i] for i in rnd_idx]

        cc = summary_count(seqs, info, self.gmap, self.new)
        cc = np.array(cc > 0, dtype=float)  # to implement other counting methods

        sample['seqs'] = seqs
        sample['counts'] = cc
        return sample


# @jit
def encode_aa_index(seqs, n=None):
    if n is None:
        n = len(seqs)
    out = np.zeros((n, len(seqs[0])))
    for i, s in enumerate(seqs):
        if i == n: break
        for j, a in enumerate(s):
            out[i, j] = AA_LIST.index(a)
    return out


# @jit
def encode_aa_onehot(seqs, n=None):
    if n is None:
        n = len(seqs)
    out = np.zeros((n, len(seqs[0]), len(AA_LIST)))
    for i, s in enumerate(seqs):
        if i == n: break
        for j, a in enumerate(s):
            out[i, j, AA_LIST.index(a)] = 1
    return np.reshape(out, (n, len(seqs[0]) * len(AA_LIST)))


# @jit
def encode_aa_atchley(seqs, n=None):
    if n is None:
        n = len(seqs)
    out = np.zeros((n, len(seqs[0]), len(AA_CHEM['A'])))
    for i, s in enumerate(seqs):
        if i == n: break
        for j, a in enumerate(s):
            out[i, j, :] = np.array(AA_CHEM[a])
    return np.reshape(out, (n, len(seqs[0]) * len(AA_CHEM['A'])))


# @jit
def fill_zero_counts(cc, n=None):
    cc = np.array(cc)
    if n is None:
        return cc
    if len(cc.shape) == 1:
        zz = np.zeros(n)
        m = min(n, cc.shape[0])
        zz[:m] = cc[:m]
    elif len(cc.shape) == 2:
        zz = np.zeros((n, cc.shape[1]))
        m = min(n, cc.shape[0])
        zz[:m, :] = cc[:m, :]
    return zz


class EncodeKmer(object):
    """ Encoding amino acid sequences into a pytorch tensor """

    def __init__(self, encode_fun=encode_aa_index, max_pool_size=2):
        self.encode_fun = encode_fun
        self.max_pool_size = max_pool_size

    def __call__(self, sample):
        seqs = sample['seqs']
        cc = sample['counts']
        sample['seqs'] = self.encode_fun(seqs, self.max_pool_size)
        sample['counts'] = fill_zero_counts(cc, self.max_pool_size)
        return sample


def make_tumor_weights(data, nclasses):
    count = np.zeros(nclasses)
    for item in data:
        tag = item['targets']
        if tag >= 0:
            count[tag] += 1
    N = count.sum()
    weight_per_class = N / (count + 1.0)
    weight = np.zeros(len(data))
    for idx, item in enumerate(data):
        tag = item['targets']
        if tag >= 0:
            weight[idx] = weight_per_class[tag]
    return weight


###############################################################################
## Test Functions for Different Deep Learning Models

def get_tumor_type(clinical, add_BRCA_subtype=True, combine_COAD_READ=True):
    """ Get the cancer types from the tcga_bcr data package """

    meta = clinical.copy()
    meta['Disease'] = meta['Disease_name']

    if add_BRCA_subtype:
        idx = (meta['Disease_name'] == 'Breast Invasive Carcinoma')
        meta.loc[idx, 'Disease'] = meta.loc[idx, 'Disease_name'] + ' ' + meta.loc[idx, 'Disease_subtype']

    if combine_COAD_READ:
        # REF: https://www.nature.com/articles/nature11252
        idx = meta['Disease_name'].isin(['Colon Adenocarcinoma', 'Rectum Adenocarcinoma'])
        meta.loc[idx, 'Disease'] = 'Colorectal Adenocarcinoma'

    return meta['Disease']


def get_label_map(meta, data=None, min_count=1):
    """ Get cancer names which have enough patient samples """
    count = {}
    if data is None:
        data = meta
    for i in data:
        label = pid2label(i, meta)
        count[label] = count.get(label, 0) + 1
    count = sorted([(count[i], i) for i in count])
    print('Sample counts are:')
    out = []
    total = 0
    for j, i in count:
        if j >= min_count:
            print(i, '\t', j)
            total += j
            out.append(i)
    print('Total\t', total)
    return out[::-1]


def get_test_environment(data_file='../data/TCGA/trust4_tcga.v20191001.txt.gz',
                         test_file='../data/TCGA/trust4_tcga_test.txt.gz',
                         work_path='../work/test_tcga_lung',
                         test_cancers=('LUSC', 'LUAD'),
                         test_min_num=100000,
                         balance_labels=True,
                         idmap=None,
                         encode_fun=encode_aa_index,
                         features=['CDR3_aa', 'Cgene']):
    """ Create a small test set
    """
    if not os.path.exists(test_file):
        outs = []
        total_size = {cancer: 0 for cancer in test_cancers}
        for chunk in pd.read_csv(data_file, chunksize=500000, sep='\t'):
            chunk['Sample_type'] = chunk.Sample_id.str.slice(13, 15)
            chunk['Disease'] = chunk['Project_id'].str.slice(start=5)
            for cancer in test_cancers:
                tmp = chunk[(chunk['Disease'] == cancer) & (chunk['Sample_type'] == '01') & \
                            (chunk['Chain'] == 'IGH') & (chunk['CDR3_score'] >= 0)]
                if tmp.shape[0] >= 1 and total_size[cancer] < test_min_num:
                    total_size[cancer] += tmp.shape[0]
                    outs.append(tmp)
            print(total_size)
            if sum([i < test_min_num for i in total_size.values()]) == 0:  # done
                break
        trust_tab = pd.concat(outs, axis=0)
        if 'File_index_start1' in list(trust_tab):
            del trust_tab['File_index_start1']
        del trust_tab['Project_id']
        trust_tab.to_csv(test_file, sep='\t', index=False, compression='gzip')
    else:
        trust_tab = pd.read_csv(test_file, sep='\t')

    # special changes for running the new version of codes
    trust_tab = trust_tab.rename(columns={'PID': 'Uniq_Patient_id', 'Disease': 'Disease_name'})
    trust_tab['Project_id'] = 'test'
    trust_tab['Disease_subtype'] = ''
    trust_tab['Sample_type'] = 'tumor'
    meta = get_tumor_type(trust_tab)
    sample_ids, data = extract_bcr(trust_tab, features=features)
    tumors = get_label_map(meta, data.index.tolist())

    if not os.path.exists(work_path):
        os.mkdir(work_path)

    random = np.random.RandomState(seed=0)
    data = data[random.permutation(data.shape[0])]

    if idmap is None:
        idmap = MapPID(to='label', labels=tumors, meta=meta)
    valid_idx = np.array([idmap(pid) is not None for pid in data.index])
    data = data[valid_idx]
    print(data.shape)

    cut = data.shape[0] // 2
    data_train = data.iloc[:cut, ]
    data_test = data.iloc[cut:, ]

    trans = transforms.Compose([Rep2Kmer(kmer=3, max_pool_size=100),
                                EncodeKmer(encode_fun=encode_fun, max_pool_size=100)])
    ds_train = TCGA_TRUST4(data=data_train, map_id=idmap, transform=trans)
    trans = transforms.Compose([Rep2Kmer(kmer=3, max_pool_size=5000),
                                EncodeKmer(encode_fun=encode_fun, max_pool_size=5000)])
    ds_test = TCGA_TRUST4(data=data_test, map_id=idmap, transform=trans)

    if balance_labels:
        weights = make_tumor_weights(ds_train, len(tumors))
        sampler = WeightedRandomSampler(weights, len(weights))
    else:
        sampler = RandomSampler(ds_train)
    train_loader = DataLoader(ds_train, batch_size=10, sampler=sampler, num_workers=4)
    test_loader = DataLoader(ds_test, batch_size=5000, num_workers=4)

    return tumors, train_loader, test_loader


def test_labels_model(test_file, work_path, epochs=10):
    tumors, train_loader, test_loader = get_test_environment(test_file=test_file,
                                                             work_path=work_path,
                                                             encode_fun=encode_aa_atchley,
                                                             features=['CDR3_aa'])

    model = TwoLayerModel(num_labels=len(tumors), save_path=work_path)
    model.train_batch(train_loader, epochs=epochs, save_step=epochs // 10)
    model.test_batch(train_loader)
    model.test_batch(test_loader)


def test_encode_model(test_file, work_path, epochs=10):
    tumors, train_loader, test_loader = get_test_environment(test_file=test_file,
                                                             work_path=work_path,
                                                             features=['CDR3_aa'])

    ei = np.array([AA_CHEM[i] for i in AA_LIST], dtype='float')
    model = EncodeLayerModel(model_name='EncodeModel', num_labels=len(tumors), encode_init=ei, save_path=work_path)
    model.train_batch(train_loader, epochs=epochs, save_step=epochs // 10)
    model.test_batch(train_loader)
    model.test_batch(test_loader)


def test_isotype_model(test_file, work_path, epochs=10):
    tumors, train_loader, test_loader = get_test_environment(test_file=test_file,
                                                             work_path=work_path,
                                                             features=['CDR3_aa', 'Cgene'])

    model = IsotypeModelFast(model_name='IsotypeModel', num_labels=len(tumors), encode_init=(len(AA_LIST), 10),
                             save_path=work_path)
    model.train_batch(train_loader, epochs=epochs, save_step=epochs // 10)
    model.test_batch(train_loader)
    model.test_batch(test_loader)


def test_antibody_model(test_file, work_path, epochs=10):
    tumors, train_loader, test_loader = get_test_environment(test_file=test_file,
                                                             work_path=work_path,
                                                             features=['CDR1_aa', 'CDR2_aa', 'CDR3_aa', 'Cgene'])

    model = IsotypeModelFast(model_name='AntibodyModel', num_labels=len(tumors), encode_init=(len(AA_LIST), 10),
                             save_path=work_path)
    model.train_batch(train_loader, epochs=epochs, save_step=epochs // 10)
    model.test_batch(train_loader)
    model.test_batch(test_loader)


def test_expression_model(test_file, work_path, exp_table, epochs=10):
    genes = ['T1', 'T2']
    idmap = MapPID(to='exps', exps=read_expression(exp_table=exp_table, genes=genes))
    tumors, train_loader, test_loader = get_test_environment(test_file=test_file, work_path=work_path,
                                                             balance_labels=False, idmap=idmap)

    model = PhialBCR_batch(model_name='DeepAntigen', num_labels=-len(genes), encode_init=(len(AA_LIST), 10),
                           save_path=work_path)
    model.train_batch(train_loader, epochs=epochs, save_step=epochs // 10)
    model.test_batch(train_loader)
    model.test_batch(test_loader)


def test_survival_model(test_file, work_path, cli_table, epochs=10):
    idmap = MapPID(to='survival', clin=read_clinical(cli_table))
    tumors, train_loader, test_loader = get_test_environment(test_file=test_file, work_path=work_path,
                                                             balance_labels=False, idmap=idmap)

    model = PhialBCR(model_name='DeepSurvival', num_labels=-1, encode_init=(len(AA_LIST), 10), save_path=work_path)
    model.train_batch(train_loader, epochs=epochs, save_step=epochs // 10)
    model.test_batch(train_loader)
    model.test_batch(test_loader)


if __name__ == '__main__':
    import optparse

    parser = optparse.OptionParser()

    parser.add_option("--workpath", type=str, default='../work')
    parser.add_option("--datapath", type=str, default='../data')
    args, _ = parser.parse_args()
    test_file = os.path.abspath(os.path.join(args.datapath, 'TCGA/trust4_tcga_test.txt.gz'))
    work_path = os.path.abspath(args.workpath)
    exp_table = os.path.abspath(os.path.join(args.datapath, 'TCGA/target_zscore.csv.gz'))
    cli_table = os.path.abspath(os.path.join(args.datapath, 'TCGA/tcga_clinical_CC.csv'))

    from datetime import datetime

    START_TIME = datetime.now()

    max_epoch = 10

    test_labels_model(test_file=test_file, work_path=work_path, epochs=max_epoch)
    test_encode_model(test_file=test_file, work_path=work_path, epochs=max_epoch)
    test_isotype_model(test_file=test_file, work_path=work_path, epochs=max_epoch)
    test_antibody_model(test_file=test_file, work_path=work_path, epochs=max_epoch)
    test_expression_model(test_file=test_file, work_path=work_path, exp_table=exp_table, epochs=max_epoch)
    test_survival_model(test_file=test_file, work_path=work_path, cli_table=cli_table, epochs=max_epoch)

    FINISH_TIME = datetime.now()
    print('Start  at', START_TIME)
    print('Finish at', FINISH_TIME)
    print("Time Cost", FINISH_TIME - START_TIME)
