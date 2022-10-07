#!/usr/bin/env python
# coding: utf-8

# # Evaluación de modelos

# In[1]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import json
import re
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer, BertTokenizerFast
from transformers import AutoModel, AutoTokenizer

import modelos as m


from fever.scorer import fever_score



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# In[4]:


tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/stsb-roberta-base-v2')

# In[5]:


relevant_arr = []

archivo = open("relevance_out_test/relevance_prediction_test_r1.json","r")
for linea in archivo:
    item=json.loads(linea)
    relevant_arr.append(item)
archivo.close()

print(len(relevant_arr))
print(relevant_arr[0])




model_fk = m.FakeClassifierSiamAtnMulShared()
ckpt = torch.load(os.path.join('..','trained','siam-atn', 'weights_fakeDetection_mul_shared_roberta.pth'), map_location=device)
state_dict = ckpt['model_state_dict']
model_fk.load_state_dict(state_dict)
model_fk.evaluacion()
model_fk = model_fk.to(device)


ventana = 5

pred_labels = []

for item in tqdm(relevant_arr):
    claim = item['claim']
    evidence = [e[2] for index, e in enumerate(item['evidence']) if index < ventana]
    
    tag = "NOT ENOUGH INFO"
    
    ev = []
    
    if len(evidence) > 0:
        inputs = tokenizer([claim]*len(evidence), max_length=128, padding='max_length', truncation=True, return_tensors="pt")
        inputs2 = tokenizer(evidence, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
        ids, mask = inputs['input_ids'], inputs['attention_mask']
        ids2, mask2 = inputs2['input_ids'], inputs2['attention_mask']

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



