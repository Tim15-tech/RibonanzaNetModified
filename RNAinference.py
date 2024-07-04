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

test_data=pd.read_csv("input/casp15.csv")
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
        print("test")
        writer.writerow(hungarian_structures[index])

def dedupe_lists(list_of_lists):
    # Step 1: Convert each sublist to a sorted tuple
    tuple_set = {tuple(sorted(sublist)) for sublist in list_of_lists}
    
    # Step 2: Convert the set of tuples back to a list of lists
    deduped_list = [list(tup) for tup in tuple_set]
    
    return deduped_list

def detect_crossed_pairs(bp_list):
    """
    Detect crossed base pairs in a list of base pairs in RNA secondary structure.

    Args:
    bp_list (list of tuples): List of base pairs, where each tuple (i, j) represents a base pair.
    
    Returns:
    list of tuples: List of crossed base pairs.
    """
    crossed_pairs_set = set()
    crossed_pairs = []
    # Iterate through each pair of base pairs
    for i in range(len(bp_list)):
        for j in range(i+1, len(bp_list)):
            bp1 = bp_list[i]
            bp2 = bp_list[j]

            # Check if they are crossed
            if (bp1[0] < bp2[0] < bp1[1] < bp2[1]) or (bp2[0] < bp1[0] < bp2[1] < bp1[1]):
                crossed_pairs.append(bp1)
                crossed_pairs.append(bp2)
                crossed_pairs_set.add(bp1[0])
                crossed_pairs_set.add(bp1[1])
                crossed_pairs_set.add(bp2[0])
                crossed_pairs_set.add(bp2[1])
    return dedupe_lists(crossed_pairs), crossed_pairs_set

eF1s=[]
for pred, bp_list in zip(test_preds,hungarian_bps):
    bpp=pred[0]
    
    crossed_pairs,crossed_pairs_set=detect_crossed_pairs(bp_list)
    global_confidence=np.mean([bpp[j, k] for j, k in bp_list])
    cross_pair_confidence=np.mean([bpp[j, k] for j, k in crossed_pairs])
    
#     global_ef1=3.66*global_confidence-2.7
#     crossed_pair_ef1=6.2*cross_pair_confidence-5.17
    
    global_ef1=2.21*global_confidence-1.25
    crossed_pair_ef1=2.02*cross_pair_confidence-1.27
    
    
    
    
    eF1s.append({'global_ef1':global_ef1,"crossed_pair_ef1":crossed_pair_ef1})
    
    
print(eF1s)



def dotbrackte2bp(structure):
    stack={'(':[],
           '[':[],
           '<':[],
           '{':[]}
    pop={')':'(',
         ']':'[',
         '>':"<",
         '}':'{'}       
    bp_list=[]
    matrix=np.zeros((len(structure),len(structure)))
    for i,s in enumerate(structure):
        if s in stack:
            stack[s].append((i,s))
        elif s in pop:
            forward_bracket=stack[pop[s]].pop()
            #bp_list.append(str(forward_bracket[0])+'-'+str(i))
            #bp_list.append([forward_bracket[0],i])
            bp_list.append([forward_bracket[0],i])

    return bp_list  


def calculate_f1_score_with_pseudoknots(true_pairs, predicted_pairs):
    true_pairs=[f"{i}-{j}" for i,j in true_pairs]
    predicted_pairs=[f"{i}-{j}" for i,j in predicted_pairs]
    
    true_pairs=set(true_pairs)
    predicted_pairs=set(predicted_pairs)

    # Calculate TP, FP, and FN
    TP = len(true_pairs.intersection(predicted_pairs))
    FP = len(predicted_pairs)-TP
    FN = len(true_pairs)-TP

    # Calculate Precision, Recall, and F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1_score

from ast import literal_eval

F1s=[]
crossed_pair_F1s=[]
for true_bp, predicted_bp in zip(test_data['bp'],hungarian_bps):

    true_bp=literal_eval(true_bp)
    crossed_pairs,crossed_pairs_set=detect_crossed_pairs(true_bp)
    predicted_crossed_pairs,predicted_crossed_pairs_set=detect_crossed_pairs(predicted_bp)
    
    _,_,f1=calculate_f1_score_with_pseudoknots(true_bp, predicted_bp)
    F1s.append(f1)
    

    if len(crossed_pairs)>0:
        _,_,crossed_pair_f1=calculate_f1_score_with_pseudoknots(crossed_pairs, predicted_crossed_pairs)
        crossed_pair_F1s.append(crossed_pair_f1)
    elif len(crossed_pairs)==0 and len(predicted_crossed_pairs)>0:
        crossed_pair_F1s.append(0)
    else:
        crossed_pair_F1s.append(np.nan)
    
    
print('global F1 mean',np.mean(F1s))
print('global F1 median',np.median(F1s))
print('crossed pair F1 mean',np.nanmean(crossed_pair_F1s))
print('crossed pair F1 median',np.nanmedian(crossed_pair_F1s))

plt.boxplot([F1s,np.array(crossed_pair_F1s)[~np.isnan(crossed_pair_F1s)]])
plt.xticks([1, 2], ['F1', 'Crossed Pair F1'])
plt.savefig('F1.png', dpi=100)
plt.show()
plt.close()

corr,p=pearsonr(F1s,[i['global_ef1'] for i in eF1s])

plt.title(f"Pearsonr: {corr:.2f}")
plt.scatter(F1s,[i['global_ef1'] for i in eF1s])
plt.xlabel('global F1')
plt.ylabel('global eF1')
plt.savefig('Pearson_1.png', dpi=100)
plt.show()
plt.close()

crossed_pair_F1s=np.array(crossed_pair_F1s)
crossed_pair_eF1s=np.array([i['crossed_pair_ef1'] for i in eF1s])

#mask nans only keep cases where both crossed_pair_F1s and crossed_pair_eF1s are defined
mask=np.isnan(crossed_pair_F1s)+np.isnan(np.array(crossed_pair_eF1s)) 
crossed_pair_F1s=np.array(crossed_pair_F1s)

corr,p=pearsonr(crossed_pair_F1s[~mask],crossed_pair_eF1s[~mask])

plt.title(f"Pearsonr: {corr:.2f}")
plt.scatter(crossed_pair_F1s,crossed_pair_eF1s)
plt.xlabel('Crossed Pair F1')
plt.ylabel('Crossed Pair eF1')
plt.savefig('Crossed_Pair.png', dpi=100)
plt.show()
plt.close()