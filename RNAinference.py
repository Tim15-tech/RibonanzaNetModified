# code source is https://www.kaggle.com/code/shujun717/ribonanzanet-2d-structure-inference

import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import sys
from Network import *
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

import csv

class RNA2D_Dataset(Dataset):
    def __init__(self,data):
        self.data=data
        self.tokens={nt:i for i,nt in enumerate('ACGU')}

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence=[self.tokens[nt] for nt in self.data.loc[idx,'sequence']]
        sequence=np.array(sequence)
        sequence=torch.tensor(sequence)

        return {'sequence':sequence}

test_data=pd.read_csv("input/RNAinference_input.csv")
test_dataset=RNA2D_Dataset(test_data)
test_dataset[0]

print(len(test_data))

sys.path.append("input/ribonanzanet2d-final")

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)
        self.entries=entries

    def print(self):
        print(self.entries)

def load_config_from_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return Config(**config)

class finetuned_RibonanzaNet(RibonanzaNet):
    def __init__(self, config):
        config.dropout=0.3
        super(finetuned_RibonanzaNet, self).__init__(config)

        self.dropout=nn.Dropout(0.0)
        self.ct_predictor=nn.Linear(64,1)

    def forward(self,src):

        #with torch.no_grad():
        _, pairwise_features=self.get_embeddings(src, torch.ones_like(src).long().to(src.device))

        pairwise_features=pairwise_features+pairwise_features.permute(0,2,1,3) #symmetrize

        output=self.ct_predictor(self.dropout(pairwise_features)) #predict

        return output.squeeze(-1)

model=finetuned_RibonanzaNet(load_config_from_yaml("configs/pairwise.yaml")).cuda()
print(model.load_state_dict(torch.load("input/ribonanzanet-weights/RibonanzaNet-SS.pt",map_location='cpu')))

test_preds=[]
model.eval()
for i in tqdm(range(len(test_dataset))):
    example=test_dataset[i]
    sequence=example['sequence'].cuda().unsqueeze(0)

    with torch.no_grad():
        test_preds.append(model(sequence).sigmoid().cpu().numpy())

with open('predictions.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for index in range(0, len(test_preds)):
        writer.writerow(test_preds[index][0])
        
plt.imshow(test_preds[0][0]) #also plt.show() doesnt help - it's not shown
plt.savefig('test_preds.png', dpi=100) # this has to be added manually
plt.close()

#create dummy arnie config
with open('arnie_file.txt','w+') as f:
    f.write("linearpartition: . \nTMP: /tmp")
    
os.environ['ARNIEFILE'] = 'arnie_file.txt'

from arnie.pk_predictors import _hungarian #do not move this up - this has to be here as it may depen ond dummy file

def mask_diagonal(matrix, mask_value=0):
    matrix=matrix.copy()
    n = len(matrix)
    for i in range(n):
        for j in range(n):
            if abs(i - j) < 4:
                matrix[i][j] = mask_value
    return matrix

test_preds_hungarian=[]
hungarian_structures=[]
hungarian_bps=[]
for i in range(len(test_preds)):
    s,bp=_hungarian(mask_diagonal(test_preds[i][0]),theta=0.5,min_len_helix=1) #best theta based on val is 0.5
    hungarian_bps.append(bp)
    ct_matrix=np.zeros((len(s),len(s)))
    for b in bp:
        ct_matrix[b[0],b[1]]=1
    ct_matrix=ct_matrix+ct_matrix.T
    test_preds_hungarian.append(ct_matrix)
    hungarian_structures.append(s)
    #break

print(hungarian_structures[0])

with open('hungarian_structures.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for index in range(0, len(hungarian_structures)):
        writer.writerow([hungarian_structures[index]]) # writerow takes iterable, so list-conversion is necessary to avoid spaces between charakters