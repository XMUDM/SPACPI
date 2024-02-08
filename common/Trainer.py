import torch
import numpy as np
from model.PerturbationAttention import PerturbationAttention
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score,recall_score,precision_score
from utils.tools import parser
import warnings
warnings.filterwarnings("ignore")
args = parser()

class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=lr, weight_decay=weight_decay)
        self.adv_optimizer = torch.optim.Adam(self.model.parameters(),
                                              lr=lr, weight_decay=weight_decay)
        self.batch = batch
        self.l2 = torch.nn.MSELoss(reduction='sum')

    def train(self, dataset):
        self.model.train()
        loss_total = 0
        pred_val = []
        y_true_s = []

        self.optimizer.zero_grad()
        for i, (p, compound_set, fp, cpi_y_train) in enumerate(dataset):
            p = torch.Tensor(p).to(args.device)
            cpi_y_train = torch.tensor(cpi_y_train, dtype=torch.long).to(args.device)
            fp = torch.LongTensor(fp).to(args.device)

            output, loss = self.model((p, cpi_y_train, fp, compound_set))
            loss.backward()

            if args.perturbation == True:
                perturb_model = PerturbationAttention(self.model).to(args.device)
                # Initial embedding
                embeds_init = perturb_model.lookup_emb(input=list(zip(*compound_set[:])))
                # Perturbation initialization
                delta = perturb_model.generate_noise(embeds_init)

                steps = args.perturbation_iters
                for astep in range(steps):
                    delta.requires_grad_()

                    '''Add perturbation to compound embedding'''
                    input_embeds = delta + embeds_init
                    if astep == steps - 1:
                        pertubation_attention = perturb_model.cal_attn(delta)
                        adv_output, adv_loss = self.model((p, compound_set, fp, cpi_y_train), emb_init=input_embeds,
                                                          pertubation_attention=pertubation_attention)
                    else:
                        adv_output, adv_loss = self.model((p, compound_set, fp, cpi_y_train), emb_init=input_embeds)

                    self.optimizer.zero_grad()
                    adv_loss.backward(retain_graph=True)

                    delta_grad = delta.grad.clone().detach()

                    # norm
                    norm = torch.norm(delta_grad)

                    # Update perturbation
                    adv_lr = 1e1
                    adv_max_norm = 3e-1
                    delta = delta + adv_lr * delta_grad / norm
                    if adv_max_norm > 0:
                        delta = torch.clamp(delta, -adv_max_norm, adv_max_norm).detach()

            # loss.backward()
            self.optimizer.step()
            if args.perturbation == True:
                loss_total += adv_loss.item()
            else:
                loss_total += loss.item()
            pred_val.append(output)
            y_true_s.append(cpi_y_train)
        with torch.no_grad():
            val_pred = torch.cat(pred_val)
            y_true = torch.cat(y_true_s)
            val_pred = val_pred.cpu().numpy()
            y_true = y_true.cpu().numpy()

            acc = accuracy_score(y_true, np.argmax(val_pred, axis=1))
            recall = recall_score(y_true, np.argmax(val_pred, axis=1))
            precision = precision_score(y_true, np.argmax(val_pred, axis=1))
            auc = roc_auc_score(y_true, val_pred[:, 1])
            aupr = average_precision_score(y_true, val_pred[:, 1])

        return loss_total, acc, auc, aupr, precision, recall