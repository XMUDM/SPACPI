import argparse
import os
import numpy as np
from torch.nn import functional as F

def parser():
    ap = argparse.ArgumentParser(description='MRGNN testing for the recommendation dataset')
    ap.add_argument('--device', default='cuda:6')
    ap.add_argument('--epoch', type=int, default=200, help='Number of epochs. Default is 200.')
    ap.add_argument('--patience', type=int, default=50, help='Patience. Default is 50.')
    ap.add_argument('--batch_size', type=int, default=64, help='Batch size. Default is 64.')
    ap.add_argument('--lr', default=0.0001)
    ap.add_argument('--weight_decay', default=1e-4)
    ap.add_argument('--dropout_rate', default=0.2)
    ap.add_argument('--num_workers', default=0, type=int)
    ap.add_argument('--cpi_dataset', default='human', type=str)
    ap.add_argument('--use_fp', type=bool, default=True,help='Choose whether to use molecular fingerprints')
    ap.add_argument('--perturbation', type=bool, default=True,help='Choose whether to add perturbation')
    ap.add_argument('--perturbation_iters', type=int, default=3)
    ap.add_argument('--Cmax', default=200, type=int)
    ap.add_argument('--Pmax', default=545, type=int)
    ap.add_argument('--dim', default=32, type=int)
    ap.add_argument('--data_dir',
                    default='./cpi_datasets/{}/{}.csv',help='Postfix for the dataset.')
    ap.add_argument('--save_dir',
                    default='./results/{}/batch_size{}_fp_{}_per_{}/',
                    help='Postfix for the saved model.')
    ap.add_argument('--only_test', default=False, type=bool)
    args = ap.parse_args()
    return args


def extract_set(set,indexs):
    new_set = []
    for index in indexs:
        new_set.append(set[index])
    return new_set

def make_dir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp, exist_ok=True)

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def stable_kl(logit, target, epsilon=1e-6, reduce=True):
        logit = logit.squeeze(-1)
        target = target.squeeze(-1)
        logit = logit.view(-1, logit.size(-1)).float()
        target = target.view(-1, target.size(-1)).float()

        bs = logit.size(0)
        p = F.log_softmax(logit, 1).exp()
        # y = F.log_softmax(target, 1).exp()
        y = target.exp()
        rp = -(1.0 / (p + epsilon) - 1 + epsilon).detach().log()
        ry = -(1.0 / (y + epsilon) - 1 + epsilon).detach().log()
        if reduce:
            return (p * (rp - ry) * 2).sum() / bs
        else:
            return (p * (rp - ry) * 2).sum()
