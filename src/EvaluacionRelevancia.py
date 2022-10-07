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


from fever.scorer import fever_score

import modelos as m


# In[2]:


#archivo = open('train_min.jsonl', 'r')
#archivo_retrieved = open('Athene/data/retrieved_docs.jsonl', 'r')

#archivo_out = open('Athene/data/retrieved_docs_min.jsonl', 'w', 100000)

#lineas = archivo_retrieved.readlines()

#for index, linea in enumerate(archivo):
#    item = json.loads(linea)
#    item_ret = json.loads(lineas[index])
#    
#    if item["id"] != item_ret["id"]:
#        print("error")
        
#    archivo_out.write(lineas[index])

#archivo.close()
#archivo_retrieved.close()
#archivo_out.close()
#lineas= None


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


# ## Referencia

# In[4]:


ROOT = "../wiki-pages"

pattern = re.compile("\n\d\d?")

directory = os.fsencode(ROOT)

wiki_dict = {}

for file in os.listdir(directory):
    
    filename = os.path.join(ROOT, os.fsdecode(file))
    
    archivo = open(filename, "r")
    
    for linea in archivo:
        item = json.loads(linea)
        lines = item["lines"]
        items = pattern.split(lines)
        items = [it.replace('0\t','').replace('\n','').replace('\t', ' ').strip().replace('"','\'') for it in items]
        #items = [it for it in items if len(it) > 0]
        
        wiki_dict[item["id"]] = items

    archivo.close()


# In[5]:


len(wiki_dict)


# In[6]:



# ## Predicción relevancia




# In[8]:


model = m.RelevanceClassifierSiamAtnShared()
ckpt = torch.load(os.path.join('..','trained','siam-atn', 'weights_relevanceClassifier_cosine_shared.pth'), map_location=device)
state_dict = ckpt['model_state_dict']
model.load_state_dict(state_dict)
model.evaluacion()
model = model.to(device)


# In[9]:


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')


# In[16]:

docs_retrieved = open('../Athene/data/retrieved_docs_test.jsonl')
#docs_retrieved = open('../Athene/data/retrieved_docs_dev.jsonl')
#docs_retrieved = open('Athene/data/retrieved_docs_train.jsonl')

relevant_arr = []

idx =0
for linea in tqdm(docs_retrieved):
    #idx+=1
	
    #if idx < 16270:
    #    continue

    item = json.loads(linea)
    
    probs = []
    sents = []
    docs = []
    idxs = []
    
    sentences_claim = []
    
    idx_sents = []
    docs_predicted = []
    
    for predicted in item["predicted_pages"]:
        sentences = wiki_dict.get(predicted, None)
        
        if sentences is None:
            continue
        
        idx_sentences = list(range(len(sentences)))
        idx_sentences = [i for i, s in zip(idx_sentences, sentences) if len(s)>0]  
        
        sentences = [predicted+" "+s for s in sentences if len(s)>0]
        
        sentences_claim.extend(sentences)
        idx_sents.extend(idx_sentences)
        docs_predicted.extend([predicted]*len(sentences))
    
    if len(sentences_claim)<1:
        relevant_arr.append({"claim":item["claim"], "evidence":[]})
        continue

    inputs = tokenizer([item["claim"]]*len(sentences_claim), max_length=128, padding='max_length', truncation=True, return_tensors="pt")
    inputs2 = tokenizer(sentences_claim, max_length=128, padding='max_length', truncation=True, return_tensors="pt")
    #ids, mask = torch.squeeze(inputs['input_ids'], dim=0), torch.squeeze(inputs['attention_mask'], dim=0)
    batch = (inputs['input_ids'], inputs['attention_mask'], inputs2['input_ids'], inputs2['attention_mask'])
    
    batch = tuple(t.detach().to(device) for t in batch)
    ids, mask, ids2, mask2 = batch

    with torch.no_grad():
        prob = model(ids, mask, ids2, mask2)

    y_hat = prob[:,1]

    probs.extend([y for y in y_hat if y >= 0])
    sents.extend([sent for sent, y in zip(sentences_claim, y_hat) if y >= 0])
    idxs.extend([i for i, y in zip(idx_sents, y_hat) if y >= 0])
    docs.extend([predicted for predicted,y in zip(docs_predicted, y_hat) if y >= 0])
        
        
    if len(probs) <1:
        relevant_arr.append({"claim":item["claim"], "evidence":[]})
        continue

    top_k = 20 if len(probs)>=20 else len(probs)

    indices = torch.topk(torch.tensor(probs), top_k)[1]
    
    evidence = []
    for indice in indices:
        ev_item = [docs[indice], idxs[indice], sents[indice]]
        evidence.append(ev_item)
    
    obj = {"claim":item["claim"], "evidence":evidence}
    
    relevant_arr.append(obj)
    
    torch.cuda.empty_cache()
    
    #idx += 1
    #if idx>10:
    #    break

docs_retrieved.close()

salida = open("relevance_out_test/relevance_prediction_test_r3.json","w")
for rel in relevant_arr:
    salida.write(json.dumps(rel)+"\n")
salida.close()


#del model
#del wiki_dict

torch.cuda.empty_cache()


# In[ ]:


#relevant_arr


# In[ ]:


#wiki_dict['Soul_Food']


# ## Predicción etiqueta
