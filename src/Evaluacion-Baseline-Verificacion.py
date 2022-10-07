#!/usr/bin/env python
# coding: utf-8

# # Verificación - BaseLine

# In[ ]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from sentence_transformers import SentenceTransformer

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random
from tqdm import trange

from itertools import islice as take

import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from os.path import join
import datetime
import time


# In[2]:


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# In[3]:


#from torch.multiprocessing import Pool, Process, set_start_method
#try:
#     set_start_method('spawn')
#except RuntimeError:
#    pass


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[119]:


def timestamp(fmt='%y%m%dT%H%M%S'):
    """Regresa la marca de tiempo."""
    return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)


# In[4]:


def carga_embeddings():
    print("Inicia carga embeddings...")
    archivo_emb=open("word_emb/glove.6B.300d.txt","r")
    embeddings={}
    for linea in archivo_emb:
        items=linea.split(" ")
        array=np.array(items[1:], dtype=np.float32)
        
        embeddings[items[0]]=array
        
    archivo_emb.close()
    
    print("Finaliza carga embeddings [OK]")
    
    return embeddings


# In[5]:


word_emb = carga_embeddings()


# In[6]:


len(word_emb)


# In[128]:


def get_embedding_word(word):
    emb=word_emb.get(word, None)
    
    if emb is not None:
        emb = np.concatenate((emb, np.ones(768-len(emb))))
        
    return emb

def get_embedding_sentence(sentence, padding):
    items=sentence.split(" ")
    s_embeddings=[]
    size_emb=768
    for item in items:
        if len(s_embeddings)==padding:
            break;
        w_embedding=get_embedding_word(item.lower())
        
        if w_embedding is None:
            continue;
        
        size_emb=len(w_embedding)
        s_embeddings.append(w_embedding)
    
    while padding>0 and len(s_embeddings)<padding:
        w_embedding=np.zeros(size_emb)
        s_embeddings.append(w_embedding)
    
    return s_embeddings


# In[8]:


sent_trans = SentenceTransformer('bert-base-nli-mean-tokens')


# In[38]:


# ### Modelo

# In[109]:


class FakeClassifierLSTM(nn.Module):
    def __init__(self):
        super(FakeClassifierLSTM, self).__init__()
        self.dense1 = nn.Sequential(
            nn.Linear(1536, 1),
            nn.Tanh(),
            nn.Softmax(dim=1)
        )
        
        self.lstm = nn.LSTM(input_size=768, batch_first=True, bidirectional=True, hidden_size=384)
        
        self.dense2 = nn.Sequential(
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
    def forward(self, claim_, evidence_):
        claim_ = claim_.unsqueeze(1).repeat(1, evidence_.shape[1], 1)
        
        in_dense1 = torch.cat([claim_, evidence_], dim=2)
        
        out_dense1 = self.dense1(in_dense1.float()).squeeze(2)
        
        out_dense1 = out_dense1.unsqueeze(1)
        
        out_lstm, _ = self.lstm(evidence_.float())
        
        atn_lstm = out_dense1 @ out_lstm
        atn_lstm = atn_lstm.squeeze(1)
        
        out_dense2 = self.dense2(atn_lstm)
        
        return out_dense2


# In[113]:


relevant_arr = []

archivo = open("relevance_out/relevance_prediction_dev_bl.json","r")
for linea in archivo:
    item=json.loads(linea)
    relevant_arr.append(item)
archivo.close()

print(len(relevant_arr))
print(relevant_arr[0])


model_fk = FakeClassifierLSTM()
ckpt = torch.load(os.path.join('..','trained','siam-atn', 'weights_fakeDetection_baseline.pth'), map_location=device)
state_dict = ckpt['model_state_dict']
model_fk.load_state_dict(state_dict)
model_fk.eval()
model_fk = model_fk.to(device)



ventana = 5

pred_labels = []

for item in tqdm(relevant_arr):
    claim = item['claim']
    evidence = [e[2] for index, e in enumerate(item['evidence']) if index < ventana]
    
    tag = "NOT ENOUGH INFO"
    
    ev = []
    
    if len(evidence) > 0:
        claim_vect = 
        
        evidence_vect = [get_embedding_sentence()]

        ids = ids.to(device)
        mask = mask.to(device)
        ids2 = ids2.to(device)
        mask2 = mask2.to(device)

        with torch.no_grad():
            prob = model_fk(ids, mask, ids2, mask2)
            
        #print(prob)

        max_ = torch.argmax(prob, 1)
        
        #print(max_)
        
        #predicted=torch.mode(max_)[0].item()
        
        #print(predicted)
        
        '''
        if predicted == 1:
            tag = "SUPPORTS"
        elif predicted == 0:
            tag = "REFUTES"
        else:
            tag = "NOT ENOUGH INFO"'''
        
        if 1 in max_:
            tag = "SUPPORTS"
        elif 0 in max_:
            tag = "REFUTES"
        else:
            tag = "NOT ENOUGH INFO"
            
        #ev = [[e[0], e[1]] for e in item['evidence']]
    
    pred_obj = {"claim":claim, "predicted_label":tag}
    pred_labels.append(pred_obj)
    


# In[9]:


salida = open("verificacion_out_test/tag_prediction_r1_v1.jsonl","w")
for pred in pred_labels:
    salida.write(json.dumps(pred)+"\n")
salida.close()

