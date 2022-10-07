import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from sentence_transformers import SentenceTransformer

import re

import json
from tqdm import tqdm

import torch

from scipy.spatial.distance import cosine
from operator import itemgetter


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device

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
    

model = SentenceTransformer('bert-base-nli-mean-tokens')
model = model.to(device)


relevant_arr = []

#docs_retrieved = open('../Athene/data/retrieved_docs_dev.jsonl')
docs_retrieved = open('../Athene/data/retrieved_docs_train.jsonl')

for linea in tqdm(docs_retrieved):
    item = json.loads(linea)
    claim = item["claim"]
    
    evidencia = []
    
    for predicted in item["predicted_pages"]:
        sentences = wiki_dict.get(predicted, None)
        
        if sentences is None:
            continue
            
        idx_sentences = list(range(len(sentences)))
        idx_sentences = [i for i, s in zip(idx_sentences, sentences) if len(s)>0]  
        
        sentences = [predicted+" "+s for s in sentences if len(s)>0]
        
        sentences_input = [claim]
        sentences_input.extend(sentences)
        vectors = model.encode(sentences_input)
        
        vector_claim = vectors[0]
        vectors = vectors[1:]
        
        for sentence, idx_sentence, vector in zip(sentences, idx_sentences, vectors):
            score = cosine(vector_claim, vector)
            ev = (predicted, idx_sentence, sentence, score)
            evidencia.append(ev)
            
    evidencia = sorted(evidencia, key=itemgetter(3), reverse=True)
    
    evidencia = evidencia[:20]
    
    evidence = [[e[0], e[1], e[2]] for e in evidencia]
            
    obj = {"claim":item["claim"], "evidence":evidence}
    
    relevant_arr.append(obj)
    
docs_retrieved.close()

salida = open("relevance_out/relevance_prediction_train_bl.json","w")
for rel in relevant_arr:
    salida.write(json.dumps(rel)+"\n")
salida.close()