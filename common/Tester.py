import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score,recall_score,precision_score
from utils.tools import parser
import warnings
warnings.filterwarnings("ignore")
args = parser()

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        pred_val = []
        y_true_s = []
        val_loss = 0
        with torch.no_grad():
            for i, (p,compound_set,fp,cpi_y_true) in enumerate(dataset):
                    p = torch.Tensor(p).to(args.device)
                    cpi_y_true = torch.tensor(cpi_y_true).long().to(args.device)
                    fp = torch.LongTensor(fp).to(args.device)
                    output, loss = self.model((p,compound_set,fp,cpi_y_true))

                    pred_val.append(output)
                    y_true_s.append(cpi_y_true)
                    val_loss += loss.item()

            val_pred = torch.cat(pred_val)
            y_true = torch.cat(y_true_s)
            val_pred = val_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()

            acc = accuracy_score(y_true, np.argmax(val_pred, axis=1))
            recall = recall_score(y_true, np.argmax(val_pred, axis=1))
            precision = precision_score(y_true, np.argmax(val_pred, axis=1))
            auc = roc_auc_score(y_true,val_pred[:,1])
            aupr = average_precision_score(y_true, val_pred[:,1])

        return val_loss,acc,auc,aupr, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)