"""
Train PhialBCR models on the complete TCGA data with different parameter settings
"""

from __future__ import print_function
import argparse
parser = argparse.ArgumentParser()

from phial_bcr_mtl import PhialBCR_MTL
from tcga_bcr import *

from datetime import datetime

START_TIME = datetime.now()

##############################################################################################

parser.add_argument("--workpath", type=str, default='../work')
parser.add_argument("--datapath", type=str, default='../data')
parser.add_argument("--outpath", type=str, default=None)
parser.add_argument("--dataset", type=str, default='MDS7')
parser.add_argument("--ignore-label-index", type=int, default=None)
parser.add_argument("--model", type=str, default='PhialBCR_MTL')
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--num-motifs", type=int, default=50)
parser.add_argument("--batch-size", type=int, default=250)
parser.add_argument("--max-num-kmer", type=int, default=25)
parser.add_argument("--kmer-size", type=int, default=5)  # kmer on CDR3
parser.add_argument("--kmer-size1", type=int, default=5)  # kmer on CDR1
parser.add_argument("--kmer-size2", type=int, default=5)  # kmer on CDR2
parser.add_argument("--encode-size", type=int, default=10)
parser.add_argument("--learning-rate", type=float, default=0.01)
parser.add_argument("--lr-decay", type=int, default=10)
parser.add_argument("--max-epoch", type=int, default=10)
parser.add_argument("--save-step", type=int, default=1)
parser.add_argument("--save-one-model-only", type=bool, default=False)
parser.add_argument("--suffix-label", type=str, default=None)
parser.add_argument("--test-dup", type=int, default=10)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--score-file", type=str, default=None)
parser.add_argument("--clinical-file", type=str, default=None)
parser.add_argument("--device", type=str, default='cpu')

args, _ = parser.parse_known_args()

if args.model.startswith('Transfer'):
    parser.add_argument("--model-transfer", type=str, default='PhialBCR_batch')
    parser.add_argument("--transfer-epoch", type=int, default=None)
    parser.add_argument("--grad-fn", type=str, default='')

if args.model.startswith('PhialBCR_MTL'):
    parser.add_argument("--multitask", type=str, default='tumors-targets-risks')
    parser.add_argument("--lambda-param1", type=float, default=0.2)
    parser.add_argument("--lambda-param2", type=float, default=0.2)

args = parser.parse_args()

data_info, meta, data_prefix = load_data(args.dataset, datapath=args.datapath, rawdata=args.datapath)
data_train = pd.read_pickle(data_prefix + '_set1.pkl.gz')
data_test = pd.read_pickle(data_prefix + '_set2.pkl.gz')
tumors = data_info.index.tolist()

if args.ignore_label_index is not None:
    ignore_sid = meta.index[meta.Label == tumors[args.ignore_label_index]].tolist()
    ignore_idx = data_train.index.isin(ignore_sid)
    data_train = data_train[~ignore_idx]  # only modify the training set
    print(ignore_idx.sum(), 'samples are ignored in training')


if args.model.startswith('PhialBCR_MTL'):
    para_list = [args.dataset,
                 str(args.num_motifs),
                 str(args.batch_size),
                 str(args.max_num_kmer),
                 str(args.encode_size),
                 str(args.kmer_size1) + str(args.kmer_size2) + str(args.kmer_size),
                 str(args.multitask), str(args.device)]
else:
    para_list = [args.dataset,
                 str(args.num_motifs),
                 str(args.batch_size),
                 str(args.max_num_kmer),
                 str(args.encode_size),
                 str(args.kmer_size1) + str(args.kmer_size2) + str(args.kmer_size),
                 str(args.device)]

if args.suffix_label is not None:  # compatible with the old format
    para_list.append(args.suffix_label)

run_name = '-'.join(para_list)

work_path = os.path.abspath(os.path.join(args.workpath, run_name))

if args.num_workers > 1:
    torch.set_num_threads(args.num_workers)
else:
    import multiprocessing

    torch.set_num_threads(multiprocessing.cpu_count())  # this is more accurate than pytorch estimation
print('Pytorch number of threads:', torch.get_num_threads())

if not os.path.exists(work_path):
    os.mkdir(work_path)

if args.device not in ['cpu','cuda']:
    raise ValueError('Unknown device. Need to be either cpu or cuda')

EI = (len(AA_LIST), args.encode_size if args.encode_size > 0 else 10)

t1 = Rep2Kmer(kmer=[args.kmer_size1, args.kmer_size2, args.kmer_size], max_pool_size=args.max_num_kmer)
t2 = EncodeKmer(encode_fun=encode_aa_index, max_pool_size=args.max_num_kmer)

if args.model == 'EncodeLayerModel':
    idmap = MapPID(to='label', labels=tumors, meta=meta)
    trans = transforms.Compose([t1, t2])
    model = EncodeLayerModel(num_motifs=args.num_motifs, num_labels=len(tumors), encode_init=EI, save_path=work_path,
                             device=args.device)

elif args.model == 'PhialBCR':
    idmap = MapPID(to='label', labels=tumors, meta=meta)
    trans = transforms.Compose([t1, t2])
    model = PhialBCR(num_motifs=args.num_motifs, num_labels=len(tumors), encode_init=EI, save_path=work_path,
                     device=args.device)

elif args.model == 'PhialBCR_batch':
    idmap = MapPID(to='label', labels=tumors, meta=meta)
    trans = transforms.Compose([t1, t2])
    model = PhialBCR_batch(num_motifs=args.num_motifs, num_labels=len(tumors), encode_init=EI, save_path=work_path,
                           device=args.device)

elif args.model.startswith('TransferExpression'):
    if '_' in args.model:
        paras = args.model.split('_')
        genes = paras[1].split('-')
        if len(genes) == 1:
            N = 1
        else:
            N = -len(genes)
    else:
        genes = read_scores(out_targets=True, score_file=args.score_file)
        N = -len(genes)

    exps = read_scores(genes=genes, score_file=args.score_file)
    idmap = MapPID(to='exps', exps=exps)

    valid_pid = exps.index
    data_train = data_train[data_train.index.isin(valid_pid)]
    data_test = data_test[data_test.index.isin(valid_pid)]

    trans = transforms.Compose([t1, t2])
    if args.model_transfer == 'PhialBCR_batch':
        model = PhialBCR_batch(num_motifs=args.num_motifs, num_labels=N, encode_init=EI, save_path=work_path,
                               model_name=args.model, device=args.device)
    elif args.model_transfer == 'PhialBCR_MTL':
        model = PhialBCR_batch(num_motifs=args.num_motifs, num_labels=N, encode_init=EI, save_path=work_path,
                               model_name=args.model, device=args.device)

    else:
        raise ValueError('Unknown model name ' + args.model_transfer)
    model.load_hidden_params = model.load_model(model_name=args.model_transfer, return_file=True)

elif args.model.startswith('PhialBCR_MTL'):

    trans = transforms.Compose([t1, t2])
    task_list = args.multitask.split('-')
    lambda_param = []
    num_labels_mtl = []
    class_tumors = None
    class_exps = None
    class_clin = None

    for i, task in enumerate(task_list):
        i = i+1

        if 'tumors' == task:
            class_tumors = tumors
            num_labels_mtl.append(len(tumors))
        elif 'targets' == task:
            genes = read_scores(out_targets=True, score_file=args.score_file)
            class_exps = read_scores(genes=genes, score_file=args.score_file)
            num_labels_mtl.append(-len(genes))
        elif 'risks' == task:
            class_clin = read_clinical(clinical_file=args.clinical_file)
            num_labels_mtl.append(-1)
        else:
            raise ValueError('Unknown task '+task)

        if i == len(task_list):
            lambda_param.append(1 - sum(lambda_param))
        elif i == 1:
            lambda_param.append(args.lambda_param1)
        elif i == 2:
            lambda_param.append(args.lambda_param2)

    print('lambdas are', list(zip(task_list, num_labels_mtl, lambda_param)))
    idmap = MapPID(to='multitask', labels=class_tumors, meta=meta, exps=class_exps, clin=class_clin, task_list=task_list)
    model = PhialBCR_MTL(num_motifs=args.num_motifs, num_labels=num_labels_mtl, lambda_param=lambda_param,
                         encode_init=EI, save_path=work_path, device=args.device, model_name=args.model)

elif args.model == 'TransferSurvival':
    clin = read_clinical(clinical_file=args.clinical_file)
    idmap = MapPID(to='survival', meta=meta, clin=clin)

    valid_pid = [pid for pid in clin.keys() if idmap(pid) is not None]
    data_train = data_train[data_train.index.isin(valid_pid)]
    data_test = data_test[data_test.index.isin(valid_pid)]

    trans = transforms.Compose([t1, t2])
    if args.model_transfer == 'PhialBCR_batch':
        model = PhialBCR_batch(num_motifs=args.num_motifs, num_labels=-1, encode_init=EI, save_path=work_path,
                               model_name=args.model, device=args.device)
    elif args.model_transfer == 'PhialBCR_MTL':
        model = PhialBCR_batch(num_motifs=args.num_motifs, num_labels=-1, encode_init=EI, save_path=work_path,
                               model_name=args.model, device=args.device)
    else:
        raise ValueError('Unknown model name ' + args.model_transfer)
    model.load_hidden_params = model.load_model(model_name=args.model_transfer, return_file=True)

else:
    raise ValueError('Unknown model name ' + args.model)


print('Train size is', len(data_train), 'and test size is', len(data_test))
ds_train = TCGA_TRUST4(data=data_train, map_id=idmap, transform=trans)
ds_test = TCGA_TRUST4(data=data_test, map_id=idmap, transform=trans)


if not args.model.startswith('Transfer') and not args.model.startswith('PhialBCR_MTL'):
    weights = make_tumor_weights(ds_train, len(tumors))
    sampler = WeightedRandomSampler(weights, len(weights))
else:
    sampler = RandomSampler(ds_train)


print('Start training...')

def worker_init_fn(x):
    np.random.seed() # use a random seed for each worker

train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

if args.model.startswith('TransferExpression'):
    if args.transfer_epoch is not None:
        args.max_epoch = args.transfer_epoch
    model.train_batch(train_loader, learning_rate=args.learning_rate, epochs=args.max_epoch, save_step=args.save_step,
                      save_one=args.save_one_model_only, lr_decay=args.lr_decay, grad_fn=args.grad_fn)
else:
    model.train_batch(train_loader, learning_rate=args.learning_rate, epochs=args.max_epoch, save_step=args.save_step,
                      save_one=args.save_one_model_only, lr_decay=args.lr_decay, grad_fn=None)

if args.model.startswith('PhialBCR_MTL'):
    res_path = [None] * len(num_labels_mtl)
    mlt = args.multitask.split('-')
    for i in range(len(num_labels_mtl)):
        res_path[i] = os.path.join(work_path, model.model_name + '/result_task_' + str(mlt[i]) + '/')
        if not os.path.exists(res_path[i]):
            os.mkdir(res_path[i])
else:
    res_path = os.path.join(work_path, model.model_name + '/result/')
    if not os.path.exists(res_path):
        os.mkdir(res_path)

print('Start evaluation...')
eva_cases = [('train', ds_train), ('test', ds_test)] # default values to be compatable with old versions, will remove in future

if args.model.startswith('PhialBCR_MTL'):
    eva_cases = []
    for dataset in ['set1','set2','set3']:
        dataset_file = data_prefix+'_'+dataset+'.pkl.gz'
        if not os.path.exists(dataset_file):
            continue
        data_raw = pd.read_pickle(dataset_file)
        ds_data = TCGA_TRUST4(data=data_raw, map_id=idmap, transform=trans, survival=False)
        eva_cases.append((dataset, ds_data))

for case, ds_data in eva_cases:

    data_loader = DataLoader(ds_data, batch_size=args.batch_size, num_workers=args.num_workers, worker_init_fn=worker_init_fn)

    if args.model.startswith('PhialBCR_MTL'):
        idx_target = 0
        for i, label_idx in enumerate(num_labels_mtl):
            # get ref data
            save_ref = os.path.join(res_path[i], case + '.ref.csv.gz')
            if not os.path.exists(save_ref):
                if label_idx > 1:
                    ref = [(d['pid'], d['targets'][idx_target]) for d in ds_data]
                    ref_task = pd.DataFrame(ref, columns=['pid', 'labels'])
                elif label_idx == -1:
                    ref = [(d['pid'], d['targets'][idx_target]) for d in ds_data]
                    ref_task = pd.DataFrame(ref, columns=['pid', 'risks'])
                elif label_idx == 1 or label_idx < -1:
                    ref = [[d['pid']] + d['targets'][idx_target:idx_target + abs(label_idx)].tolist() for d in ds_data]
                    ref_task = pd.DataFrame(ref, columns=['pid'] + genes)
                ref_task.to_csv(save_ref, index=False)

            if label_idx < -1:
                idx_target += abs(label_idx)
            else:
                idx_target += 1

        all_predicted = None
        for i, label_idx in enumerate(num_labels_mtl):
            # get pred
            save_pred = os.path.join(res_path[i], case + '.Epoch_{}.csv.gz'.format(args.max_epoch))
            if not os.path.exists(save_pred):
                if all_predicted is None:
                    # let's predict all
                    all_predicted = []
                    for _ in range(args.test_dup):
                        obj, eva, out = model.test_batch(data_loader)
                        all_predicted.append(out)

                pred_out = 0
                for out in all_predicted:
                    pred_out += out[i]
                pred_out /= float(args.test_dup)

                if label_idx == 1 or label_idx < -1:
                    pred_out = pd.DataFrame(pred_out, columns=genes)
                elif label_idx > 1:
                    pred_out = pd.DataFrame(pred_out, columns=tumors)
                elif label_idx == -1:
                    pred_out = pd.DataFrame(pred_out, columns=['Risks'])
                pred_out.to_csv(save_pred, index=False)
        all_predicted = None # free up memory

        for i, label_idx in enumerate(num_labels_mtl):
            # evaluate
            save_ref = os.path.join(res_path[i], case + '.ref.csv.gz')
            save_pred = os.path.join(res_path[i], case + '.Epoch_{}.csv.gz'.format(args.max_epoch))
            ref_task = pd.read_csv(save_ref)
            pred_out = pd.read_csv(save_pred)

            ref_mat = ref_task.iloc[:, 1:].values
            prd_mat = pred_out.values

            valid_idx = ~np.any(np.isnan(ref_mat), axis=1)
            ref_mat = ref_mat[valid_idx, :]
            prd_mat = prd_mat[valid_idx, :]

            if label_idx > 1:
                loss, mcc = evaluate_classifier(ref_mat, prd_mat, class_labels=len(tumors))
                print("Dataset {} MCC: {:f}".format(case, mcc))
                res = pd.DataFrame([[args.model, run_name, case, args.max_epoch, loss, mcc]],
                                   columns=['Model', 'Key', 'Case', 'Epoch', 'Loss', 'MCC'])
            elif label_idx == 1 or label_idx < -1:
                res = pd.DataFrame()
                res['Target'] = list(ref_task)[1:]
                res['PCC'], res['PCC_Pval'] = colwise_pearson(ref_mat, prd_mat)
                res['SPC'], res['SPC_Pval'] = colwise_spearman(ref_mat, prd_mat)
                print("Dataset {} Correlation: {:f}".format(case, 0.5*(res.PCC.mean()+res.SPC.mean())))
            elif label_idx == -1:
                loss, cidx = evaluate_survival(ref_mat, prd_mat)
                print("Dataset {} C-index: {:f}".format(case, cidx))
                res = pd.DataFrame([[args.model, run_name, case, args.max_epoch, loss, cidx]],
                                   columns=['Model', 'Key', 'Case', 'Epoch', 'Loss', 'C-index'])
            save_res = os.path.join(res_path[i], case + '.Epoch_{}.res.csv'.format(args.max_epoch))
            res.to_csv(save_res, index=False)

    else: ## other than PhialBCR-MTL
        save = os.path.join(res_path, case + '.ref.csv.gz')
        if not os.path.exists(save):
            if args.model.startswith('TransferExpression'):
                ref = [[d['pid']] + d['targets'].tolist() for d in ds_data]
                ref = pd.DataFrame(ref, columns=['pid'] + genes)
            else:
                ref = [(d['pid'], d['targets']) for d in ds_data]
                ref = pd.DataFrame(ref, columns=['pid', 'target'])
            ref.to_csv(save, index=False)
        else:
            ref = pd.read_csv(save)

        save = os.path.join(res_path, case + '.Epoch_{}.csv.gz'.format(args.max_epoch))
        if not os.path.exists(save):
            pred = 0
            for i in range(args.test_dup):
                obj, eva, p = model.test_batch(data_loader)
                pred += p
            pred /= float(args.test_dup)
            if args.model.startswith('TransferExpression'):
                pred = pd.DataFrame(pred, columns=genes)
            elif args.model == 'TransferSurvival':
                pred = pd.DataFrame(pred, columns=['Year_' + str(a) for a in range(1, args.n_intervals + 1)])
            else:
                pred = pd.DataFrame(pred, columns=tumors)
            pred.to_csv(save, index=False)
        else:
            pred = pd.read_csv(save)

        save = os.path.join(res_path, case + '.Epoch_{}.res.csv'.format(args.max_epoch))
        if not os.path.exists(save):
            ref_mat = ref.iloc[:, 1:].values
            prd_mat = pred.values
            if args.model.startswith('TransferExpression'):
                res = pd.DataFrame()
                res['Target'] = list(ref)[1:]
                res['PCC'], res['PCC_Pval'] = colwise_pearson(ref_mat, prd_mat)
                res['SPC'], res['SPC_Pval'] = colwise_spearman(ref_mat, prd_mat)
                print("Correlation: {:.4f}".format(0.5*(res.PCC.mean()+res.SPC.mean())), end='')
            elif args.model == 'TransferSurvival':
                loss, cidx = evaluate_survival(ref_mat, prd_mat)
                res = pd.DataFrame([[args.model, run_name, case, args.max_epoch, loss, cidx]],
                                   columns=['Model', 'Key', 'Case', 'Epoch', 'Loss', 'C-index'])
            else:
                loss, mcc = evaluate_classifier(ref_mat, prd_mat)
                print("MCC: {:.3f}, Average {} loss: {}".format(mcc, case, loss))
                res = pd.DataFrame([[args.model, run_name, case, args.max_epoch, loss, mcc]],
                                   columns=['Model', 'Key', 'Case', 'Epoch', 'Loss', 'MCC'])
            res.to_csv(save, index=False)
        else:
            res = pd.read_csv(save)
            if not args.model.startswith('Transfer'):
                print("MCC: {:.3f}, Average {} loss: {}".format(res.loc[0, 'MCC'], res.loc[0, 'Case'],
                                                                res.loc[0, 'Loss']))


##############################################################################################
import sys, json

with open(os.path.join(work_path, model.model_name + '/train_parameters.json'), 'w') as f:
    params = dict(args.__dict__)
    params['command'] = ' '.join(sys.argv)
    json.dump(params, f, indent=2)

FINISH_TIME = datetime.now()
print('Parameters:', vars(args))
print('Start  at', START_TIME)
print('Finish at', FINISH_TIME)
print("Time Cost", FINISH_TIME - START_TIME)

if args.outpath is not None:  # mainly for AWS SageMaker Spot runs
    from shutil import copytree, ignore_patterns

    copytree(work_path, os.path.join(args.outpath, run_name), ignore=ignore_patterns('*sagemaker-uploaded'))
