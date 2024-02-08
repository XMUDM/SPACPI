import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from utils.tools import parser,make_dir
import torch
import torch.nn as nn
from common import Trainer,Tester
from model.SPACPI import SPACPI
from utils.dataset import load_dataset,dataset_segmentation
from utils.dataloader import dataloader

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

setup_seed(1)

def run_model(args):
    drug_smiles, protein_seqs, cpi_labels, compouns_set, N_fingerprints = load_dataset(args.data_dir.format(str(args.cpi_dataset), str(args.cpi_dataset)+'_preprocess'))

    # network
    model = SPACPI(N_fingerprints, args)
    model.to(args.device)

    # Trainer
    trainer = Trainer(model, args.lr, args.weight_decay, args.batch_size)

    # Tester
    tester = Tester(model)

    if not args.only_test:
        dataset = list(zip(drug_smiles, protein_seqs, cpi_labels, compouns_set))
        dataset_train,dataset_dev,dataset_test = dataset_segmentation(dataset,args.cpi_dataset)

        print('------------------------')
        print(' CPI | Train_pairs:', len(dataset_train), ' | Dev_pairs:', len(dataset_dev), ' | Test_pairs:',
              len(dataset_test), ' | Neg_Times:', args.neg_times, ' | Dataset:', args.cpi_dataset)
        print('------------------------')

        train_loader = dataloader(dataset_train,batch_size=args.batch_size, shuffle=True)
        dev_loader = dataloader(dataset_train,batch_size=args.batch_size, shuffle=True)
        test_loader = dataloader(dataset_train,batch_size=1, shuffle=False)

    best_acc,best_auc,best_aupr,best_recall,best_precision = 0,0,0,0,0
    counter = 0
    if os.path.exists(args.save_dir.format(args.cpi_dataset,args.batch_size,args.use_fp,args.perturbation) + 'best_checkpoint.pt'):
                # pass
                print('Load model weights from best_checkpoint.pt')
                model.load_state_dict(torch.load(args.save_dir.format(args.cpi_dataset,args.batch_size,args.use_fp,args.perturbation) + 'best_checkpoint.pt', map_location=args.device),strict=False)
    for epoch in range(args.epoch):
        loss_train,acc,auc,aupr, precision, recall = trainer.train(train_loader)
        loss_dev,acc_dev, AUC_dev, PRC_dev, precision_dev, recall_dev = tester.test(dev_loader)
        loss_test,acc_test, AUC_test, PRC_test, precision_test, recall_test = tester.test(test_loader)

        str = 'Epoch {:d} | loss {:.6f} | Train| acc {:.4f} | precision {:.4f} | recall {:.4f} | auc {:.4f} | aupr {:.4f} '
        print(str.format(epoch, loss_train, acc, precision, recall, auc, aupr))
        print(str.format(epoch,loss_dev,acc_dev, precision_dev, recall_dev, AUC_dev, PRC_dev))
        print(str.format(epoch, loss_test,acc_test, precision_test, recall_test, AUC_test, PRC_test))

        # early stopping
        if (best_auc < AUC_test):
            best_acc = acc_test
            best_auc = AUC_test
            best_aupr = PRC_test
            best_precision = precision_test
            best_recall = recall_test

            print('Save model...\n')
            make_dir(args.save_dir.format(args.cpi_dataset, args.batch_size,args.use_fp,args.perturbation))
            torch.save(model.state_dict(), args.save_dir.format(args.cpi_dataset, args.batch_size,args.use_fp,args.perturbation) + 'best_checkpoint.pt')
            counter = 0
        else:
            counter += 1
            print('Patience:',counter,'\n')
        if counter >= args.patience:
            print('Early stopping!')
            break
    print('ACC:',best_acc)
    print('AUC:',best_auc)
    print('PRC:', best_aupr)
    print('Precision:', best_precision)
    print('Recall:', best_recall)

if __name__ == '__main__':
    args = parser()
    run_model(args)

