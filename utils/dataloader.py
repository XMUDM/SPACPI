import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from torch.utils.data import Dataset
from utils.dataset import protein2emb_encoder
from torch.utils.data import DataLoader
from utils.tools import parser
args = parser()

class mydatasetcpi(Dataset):
    def __init__(self,dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        smile = self.dataset[index][0]
        seq = self.dataset[index][1]
        cpi_y = self.dataset[index][2]
        compound_set = self.dataset[index][-1]

        mol = Chem.MolFromSmiles(smile)

        # FingerPrint
        fp = []
        fp_maccs = AllChem.GetMACCSKeysFingerprint(mol)
        fp_phaErGfp = AllChem.GetErGFingerprint(mol, fuzzIncrement=0.3, maxPath=21, minPath=1)
        fp.extend(fp_maccs)
        fp.extend(fp_phaErGfp)
        fp = np.array(fp)
        max_length = args.Cmax
        fp_idx = np.nonzero(fp)[0]
        if len(fp_idx) >= max_length:
            fp_idx = fp_idx[:max_length]
        else:
            fp_idx = np.pad(fp_idx, (0, max_length - len(fp_idx)), 'constant', constant_values=0)
        # Protein
        v_p = protein2emb_encoder(seq)
        return v_p[0],cpi_y,fp_idx,compound_set

class collate_fc_cpi(object):
    def __init__(self,device):
        self.device = device

    def collate_func(self, batch_list):
        p = [v for v,_,_,_ in batch_list]
        cpi_y_true = [v for _,v,_,_ in batch_list]
        fp = [v for _,_,v,_ in batch_list]
        compound_set = [v for _,_,_,v in batch_list]

        return p,compound_set,fp,cpi_y_true


def dataloader(dataset_,batch_size, shuffle):
    dataset = mydatasetcpi(dataset_)  # return p,cpi_y,fp,compound_set
    collate = collate_fc_cpi(device=args.device)
    dataloadercpi = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                              num_workers=args.num_workers, drop_last=True,
                              collate_fn=collate.collate_func)
    return dataloadercpi



